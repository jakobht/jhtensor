use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{Mutex, RwLock},
};

use crate::tensor::{Activation, Backend, DType, Shape, TensorError};

pub struct MetalBackend;

#[repr(C)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    activation: u32,
}

#[repr(C)]
struct TransposeParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
struct SumAxisParams {
    rows: u32,
    cols: u32,
    axis: u32,
}

#[repr(C)]
struct BroadcastParams {
    rows: u32,
    cols: u32,
    axis: u32,
}

impl Backend for MetalBackend {
    type Storage = Retained<ProtocolObject<dyn MTLBuffer>>;

    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: Shape,
        b: &Self::Storage,
        shape_b: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
        activation: Activation,
    ) -> Result<(), TensorError> {
        let ctx = get_metal_context()?;

        with_compute_encoder(&ctx, |encoder| unsafe {
            let pipeline = ctx
                .get_pipeline(&dtype.pipeline_name("mat_mul"))
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = MatMulParams {
                m: shape_a[0] as u32,
                n: shape_b[1] as u32,
                k: shape_a[1] as u32,
                activation: activation.to_shader_flag(),
            };

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 2);

            set_params(encoder, &mut params, 3);

            let threadgroup_size = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            let grid_width = (shape_b[1] + 15) / 16;
            let grid_height = (shape_a[0] + 15) / 16;

            let threadgroups_per_grid = MTLSize {
                width: grid_width,
                height: grid_height,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threadgroup_size);
        })
    }

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError> {
        dispatch_elementwise("add_arrays", a, b, dest, shape, dtype)
    }

    fn mul_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError> {
        dispatch_elementwise("mul_arrays", a, b, dest, shape, dtype)
    }

    fn sub_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError> {
        dispatch_elementwise("sub_arrays", a, b, dest, shape, dtype)
    }

    fn transpose_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
    ) -> Result<(), TensorError> {
        let ctx = get_metal_context()?;

        with_compute_encoder(&ctx, |encoder| unsafe {
            let pipeline = ctx
                .get_pipeline(&dtype.pipeline_name("transpose"))
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = TransposeParams {
                rows: shape[0] as u32,
                cols: shape[1] as u32,
            };

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 1);

            set_params(encoder, &mut params, 2);

            let threadgroup_size = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            let grid_width = (shape[1] + 15) / 16;
            let grid_height = (shape[0] + 15) / 16;

            let threadgroups_per_grid = MTLSize {
                width: grid_width,
                height: grid_height,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threadgroup_size);
        })
    }

    fn sum_axis_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError> {
        let ctx = get_metal_context()?;

        with_compute_encoder(&ctx, |encoder| unsafe {
            let pipeline = ctx
                .get_pipeline(&dtype.pipeline_name("sum_axis"))
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = SumAxisParams {
                rows: shape[0] as u32,
                cols: shape[1] as u32,
                axis: axis as u32,
            };

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 1);
            set_params(encoder, &mut params, 2);

            let dest_size = shape[if axis == 0 { 1 } else { 0 }];

            let max_threads_per_group = 256;
            let threadgroup_width = std::cmp::min(dest_size, max_threads_per_group);

            let threadgroup_size = MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            };

            let grid_width = (dest_size + threadgroup_width - 1) / threadgroup_width;

            let threadgroups_per_grid = MTLSize {
                width: grid_width,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threadgroup_size);
        })
    }

    fn broadcast_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dest_shape: Shape,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError> {
        assert!(shape.ndim() == 1, "Shape must be 1 for broadcast");
        assert!(dest_shape.ndim() == 2, "Destination shape must be 2 for broadcast");
        assert!(axis == 0 || axis == 1, "Axis must be 0 or 1 for broadcast");
        assert!(
            axis == 1 && shape[0] == dest_shape[0] || axis == 0 && shape[0] == dest_shape[1],
            "Shape must match destination shape"
        );

        let ctx = get_metal_context()?;

        with_compute_encoder(&ctx, |encoder| unsafe {
            let pipeline = ctx
                .get_pipeline(&dtype.pipeline_name("broadcast"))
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = BroadcastParams {
                rows: dest_shape[0] as u32,
                cols: dest_shape[1] as u32,
                axis: axis as u32,
            };

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 1);
            set_params(encoder, &mut params, 2);

            let threadgroup_size = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            let grid_width = (dest_shape[1] + 15) / 16;
            let grid_height = (dest_shape[0] + 15) / 16;

            let threadgroups_per_grid = MTLSize {
                width: grid_width,
                height: grid_height,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threadgroup_size);
        })
    }

    #[inline]
    fn allocate_empty(size: usize, dtype: DType) -> Result<Self::Storage, TensorError> {
        let ctx = get_metal_context()?;
        Ok(ctx
            .device
            .newBufferWithLength_options(size * dtype.byte_size(), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| TensorError::BackendFailure("Failed to allocate GPU buffer".into()))?)
    }

    #[inline]
    fn from_slice<T: Copy>(data: &[T]) -> Result<Self::Storage, TensorError> {
        unsafe {
            let ctx = get_metal_context()?;
            let buffer_size = data.len() * std::mem::size_of::<T>();

            let buffer = ctx
                .device
                .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| TensorError::BackendFailure("Failed to allocate GPU buffer".into()))?;

            let raw_ptr = buffer.contents().as_ptr().cast::<T>();
            let slice = std::slice::from_raw_parts_mut(raw_ptr, data.len());
            slice.copy_from_slice(data);

            Ok(buffer)
        }
    }

    #[inline]
    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Result<Vec<T>, TensorError> {
        unsafe {
            let raw_ptr = storage.contents().as_ptr().cast::<T>();
            let slice = std::slice::from_raw_parts(raw_ptr, num_elements);
            Ok(slice.to_vec())
        }
    }
}

#[allow(clippy::missing_safety_doc)]
unsafe fn set_params<T: Sized>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, params: &mut T, index: usize) {
    encoder.setBytes_length_atIndex(
        NonNull::new(std::ptr::from_mut(params).cast::<std::ffi::c_void>())
            .expect("Invalid params pointer"),
        std::mem::size_of::<T>(),
        index,
    );
}

fn with_compute_encoder<F>(ctx: &MetalContext, body: F) -> Result<(), TensorError>
where
    F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
{
    let command_buffer = ctx
        .command_queue
        .commandBuffer()
        .ok_or_else(|| TensorError::BackendFailure("Failed to create command buffer".into()))?;

    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| TensorError::BackendFailure("Failed to create compute encoder".into()))?;

    body(&encoder);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    Ok(())
}

fn dispatch_elementwise(
    prefix: &str,
    a: &<MetalBackend as Backend>::Storage,
    b: &<MetalBackend as Backend>::Storage,
    dest: &mut <MetalBackend as Backend>::Storage,
    shape: Shape,
    dtype: DType,
) -> Result<(), TensorError> {
    let ctx = get_metal_context()?;
    let array_length = shape.product();

    with_compute_encoder(&ctx, |encoder| unsafe {
        let pipeline = ctx
            .get_pipeline(&dtype.pipeline_name(prefix))
            .expect(&format!("Failed to get pipeline for {:?}", dtype));

        encoder.setComputePipelineState(&pipeline);
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(dest), 0, 2);

        let thread_execution_width = pipeline.maxTotalThreadsPerThreadgroup();
        let grid_width = if thread_execution_width > array_length {
            array_length
        } else {
            thread_execution_width
        };

        let grid_size = MTLSize {
            width: array_length,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: grid_width,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    })
}

/// Holds our heavy, reusable GPU states for all supported data types
struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RwLock<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl MetalContext {
    /// Retrieves a cached pipeline or compiles and caches it if missing.
    fn get_pipeline(&self, name: &str) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, TensorError> {
        if let Some(pipeline) = self.pipelines.read().unwrap().get(name).cloned() {
            return Ok(pipeline);
        }

        let mut cache = self.pipelines.write().unwrap();
        if let Some(pipeline) = cache.get(name).cloned() {
            return Ok(pipeline);
        }

        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .expect(&format!("Shader function '{}' not found", name));
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .expect(&format!("Failed to create pipeline for '{}'", name));

        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}

/// Fetches the GPU state, initializing all variations exactly once
fn get_metal_context() -> Result<&'static MetalContext, TensorError> {
    static CONTEXT: Mutex<Option<&'static MetalContext>> = Mutex::new(None);

    let mut lock = CONTEXT.lock().expect("Failed to lock Metal context");

    if let Some(metal_context) = *lock {
        return Ok(metal_context);
    }

    let device =
        MTLCreateSystemDefaultDevice().ok_or_else(|| TensorError::BackendFailure("No Metal device found".into()))?;

    let source = NSString::from_str(concat!(
        include_str!("../../src/shaders/mat_mul_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/add_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/mul_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/sub_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/transpose_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/sum_axis_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/broadcast_func.metal"),
    ));

    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("Failed to compile Metal shader source at runtime");

    let command_queue = device
        .newCommandQueue()
        .ok_or_else(|| TensorError::BackendFailure("Failed to create command queue".into()))?;

    let metal_context = MetalContext {
        device,
        command_queue,
        library,
        pipelines: RwLock::new(HashMap::new()),
    };

    // It's fine to leak the box here because the context is static and will never be deallocated.
    let metal_context = Box::leak(Box::new(metal_context));
    *lock = Some(metal_context);
    Ok(metal_context)
}
