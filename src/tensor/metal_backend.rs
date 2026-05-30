use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::tensor::{Backend, DType};

pub struct MetalBackend;

impl Backend for MetalBackend {
    type Storage = Retained<ProtocolObject<dyn MTLBuffer>>;

    fn add_arrays(a: &Self::Storage, b: &Self::Storage, shape: &[usize], dtype: DType) -> Self::Storage {
        unsafe {
            let ctx = get_metal_context();

            let array_length = shape.iter().product();
            let buffer_size = array_length * dtype.byte_size();

            let buffer_result = ctx
                .device
                .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
                .unwrap();

            let command_buffer = ctx.command_queue.commandBuffer().unwrap();
            let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

            let pipeline = match dtype {
                DType::Float32 => &ctx.add_f32_pipeline,
                DType::Int32 => &ctx.add_i32_pipeline,
                DType::Int16 => &ctx.add_i16_pipeline,
            };
            compute_encoder.setComputePipelineState(pipeline);

            compute_encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            compute_encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
            compute_encoder.setBuffer_offset_atIndex(Some(&buffer_result), 0, 2);

            let thread_execution_width = pipeline.maxTotalThreadsPerThreadgroup();
            let grid_with = if thread_execution_width > array_length {
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
                width: grid_with,
                height: 1,
                depth: 1,
            };

            compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
            compute_encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            buffer_result
        }
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

use std::sync::OnceLock;

/// Holds our heavy, reusable GPU states for all supported data types
struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    add_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_i32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_i16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// Fetches the GPU state, initializing all variations exactly once
fn get_metal_context() -> &'static MetalContext {
    static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

    CONTEXT.get_or_init(|| {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");

        let source = NSString::from_str(include_str!("../../src/shaders/add_func.metal"));

        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Failed to compile Metal shader source at runtime");

        // Extract and compile the Float32 entry point
        let f32_fn = library
            .newFunctionWithName(&NSString::from_str("add_arrays_f32"))
            .unwrap();
        let add_f32_pipeline = device.newComputePipelineStateWithFunction_error(&f32_fn).unwrap();

        // Extract and compile the Int32 entry point
        let i32_fn = library
            .newFunctionWithName(&NSString::from_str("add_arrays_i32"))
            .unwrap();
        let add_i32_pipeline = device.newComputePipelineStateWithFunction_error(&i32_fn).unwrap();

        // Extract and compile the Float16 entry point
        let i16_fn = library
            .newFunctionWithName(&NSString::from_str("add_arrays_i16"))
            .unwrap();
        let add_i16_pipeline = device.newComputePipelineStateWithFunction_error(&i16_fn).unwrap();

        let command_queue = device.newCommandQueue().unwrap();

        MetalContext {
            device,
            command_queue,
            add_f32_pipeline,
            add_i32_pipeline,
            add_i16_pipeline,
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Backend, DType, MetalBackend};

    mod add_arrays {
        use super::*;

        macro_rules! test_type {
            ($t:ident, $dtype:expr) => {
                #[test]
                fn $t() {
                    let a = MetalBackend::from_slice(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t]);
                    let b = MetalBackend::from_slice(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t]);

                    let result_bytes = MetalBackend::add_arrays(&a, &b, &[5], $dtype);

                    let result_vec = MetalBackend::to_vec::<$t>(&result_bytes, 5);

                    assert_eq!(
                        result_vec,
                        vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]
                    );
                }
            };
        }
        test_type!(f32, DType::Float32);
        test_type!(i32, DType::Int32);
        test_type!(i16, DType::Int16);
    }
}
