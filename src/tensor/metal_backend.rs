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

use crate::tensor::{Activation, Backend, DType, TensorError};

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

impl Backend for MetalBackend {
    type Storage = Retained<ProtocolObject<dyn MTLBuffer>>;

    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: &[usize],
        b: &Self::Storage,
        shape_b: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
        activation: Activation,
    ) -> Result<(), TensorError> {
        unsafe {
            let ctx = get_metal_context()?;
            let command_buffer = ctx
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = command_buffer
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            let pipeline = ctx
                .get_pipeline(match dtype {
                    DType::Float32 => "mat_mul_f32",
                    DType::Int32 => "mat_mul_i32",
                    DType::Int16 => "mat_mul_i16",
                })
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

            encoder.setBytes_length_atIndex(
                NonNull::new(std::ptr::from_mut(&mut params).cast::<std::ffi::c_void>())
                    .expect("Invalid params pointer"),
                std::mem::size_of::<MatMulParams>(),
                3,
            );

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
            encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: &[usize],
        dtype: DType,
    ) -> Result<(), TensorError> {
        unsafe {
            let ctx = get_metal_context()?;
            let array_length = shape.iter().product();

            let command_buffer = ctx
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let compute_encoder = command_buffer
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            let pipeline = ctx
                .get_pipeline(match dtype {
                    DType::Float32 => "add_arrays_f32",
                    DType::Int32 => "add_arrays_i32",
                    DType::Int16 => "add_arrays_i16",
                })
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            compute_encoder.setComputePipelineState(&pipeline);

            compute_encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            compute_encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
            compute_encoder.setBuffer_offset_atIndex(Some(dest), 0, 2);

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

            compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
            compute_encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    fn transpose_inplace(
        a: &Self::Storage,
        shape: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
    ) -> Result<(), TensorError> {
        unsafe {
            let ctx = get_metal_context()?;
            let command_buffer = ctx
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = command_buffer
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            let pipeline = ctx
                .get_pipeline(match dtype {
                    DType::Float32 => "transpose_f32",
                    DType::Int32 => "transpose_i32",
                    DType::Int16 => "transpose_i16",
                })
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = TransposeParams {
                rows: shape[0] as u32,
                cols: shape[1] as u32,
            };

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 1);

            encoder.setBytes_length_atIndex(
                NonNull::new(std::ptr::from_mut(&mut params).cast::<std::ffi::c_void>())
                    .expect("Invalid params pointer"),
                std::mem::size_of::<TransposeParams>(),
                2,
            );

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
            encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    fn sum_axis_inplace(
        a: &Self::Storage,
        shape: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError> {
        unsafe {
            let ctx = get_metal_context()?;

            let pipeline = ctx
                .get_pipeline(match dtype {
                    DType::Float32 => "sum_axis_f32",
                    DType::Int32 => "sum_axis_i32",
                    DType::Int16 => "sum_axis_i16",
                })
                .expect(&format!("Failed to get pipeline for {:?}", dtype));

            let mut params = SumAxisParams {
                rows: shape[0] as u32,
                cols: shape[1] as u32,
                axis: axis as u32,
            };

            let params_ptr = NonNull::new(std::ptr::from_mut(&mut params).cast::<std::ffi::c_void>())
                .expect("Invalid params pointer");

            let command_buffer = ctx
                .command_queue
                .commandBuffer()
                .ok_or_else(|| TensorError::BackendFailure("Failed to create command buffer".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| TensorError::BackendFailure("Failed to create compute encoder".into()))?;

            encoder.setComputePipelineState(&pipeline);
            encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dest), 0, 1);
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<SumAxisParams>(), 2);

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
            encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }
        Ok(())
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
        include_str!("../../src/shaders/transpose_func.metal"),
        "\n\n",
        include_str!("../../src/shaders/sum_axis_func.metal"),
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
