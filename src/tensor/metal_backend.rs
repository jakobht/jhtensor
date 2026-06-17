use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{OnceLock, RwLock},
};

use crate::tensor::{Activation, Backend, DType};

pub struct MetalBackend;

impl MetalBackend {
    fn new_empty(num_elements: usize, dtype: DType) -> <Self as Backend>::Storage {
        let ctx = get_metal_context();
        let buffer_size = num_elements * dtype.byte_size();
        ctx.device
            .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
            .unwrap()
    }
}

#[repr(C)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    activation: u32,
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
    ) {
        unsafe {
            let ctx = get_metal_context();

            let command_buffer = ctx.command_queue.commandBuffer().unwrap();
            let encoder = command_buffer.computeCommandEncoder().unwrap();

            let pipeline = match dtype {
                DType::Float32 => ctx.get_pipeline("mat_mul_f32"),
                DType::Int32 => ctx.get_pipeline("mat_mul_i32"),
                DType::Int16 => ctx.get_pipeline("mat_mul_i16"),
            };

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
                NonNull::new(std::ptr::from_mut(&mut params).cast::<std::ffi::c_void>()).unwrap(),
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
    }

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: &[usize],
        dtype: DType,
    ) {
        unsafe {
            let ctx = get_metal_context();
            let array_length = shape.iter().product();

            let command_buffer = ctx.command_queue.commandBuffer().unwrap();
            let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

            let pipeline = match dtype {
                DType::Float32 => ctx.get_pipeline("add_arrays_f32"),
                DType::Int32 => ctx.get_pipeline("add_arrays_i32"),
                DType::Int16 => ctx.get_pipeline("add_arrays_i16"),
            };
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
    }

    fn allocate_empty(size: usize, dtype: DType) -> Self::Storage {
        let ctx = get_metal_context();
        ctx.device
            .newBufferWithLength_options(size * dtype.byte_size(), MTLResourceOptions::StorageModeShared)
            .unwrap()
    }

    fn from_slice<T: Copy>(data: &[T]) -> Self::Storage {
        unsafe {
            let ctx = get_metal_context();
            let buffer_size = data.len() * std::mem::size_of::<T>();

            let buffer = ctx
                .device
                .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
                .unwrap();

            let raw_ptr = buffer.contents().as_ptr().cast::<T>();
            let slice = std::slice::from_raw_parts_mut(raw_ptr, data.len());
            slice.copy_from_slice(data);

            buffer
        }
    }

    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Vec<T> {
        unsafe {
            let raw_ptr = storage.contents().as_ptr().cast::<T>();
            let slice = std::slice::from_raw_parts(raw_ptr, num_elements);
            slice.to_vec()
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
    fn get_pipeline(&self, name: &str) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        if let Some(pipeline) = self.pipelines.read().unwrap().get(name) {
            return pipeline.clone();
        }

        let mut cache = self.pipelines.write().unwrap();

        // Check if the pipeline was already inserted into the cache (if not we are responsible for inserting it)
        if let Some(pipeline) = cache.get(name) {
            return pipeline.clone();
        }

        let function = self.library.newFunctionWithName(&NSString::from_str(name)).unwrap();
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        cache.insert(name.to_string(), pipeline.clone());

        pipeline
    }
}

/// Fetches the GPU state, initializing all variations exactly once
fn get_metal_context() -> &'static MetalContext {
    static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

    CONTEXT.get_or_init(|| {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");

        let source = NSString::from_str(concat!(
            include_str!("../../src/shaders/mat_mul_func.metal"),
            "\n\n",
            include_str!("../../src/shaders/add_func.metal"),
        ));

        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Failed to compile Metal shader source at runtime");

        let command_queue = device.newCommandQueue().unwrap();

        MetalContext {
            device,
            command_queue,
            library,
            pipelines: RwLock::new(HashMap::new()),
        }
    })
}
