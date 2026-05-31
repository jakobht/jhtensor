mod cpu_backend;
mod dtype;
mod metal_backend;
mod tensor;
mod tensor_error;
pub use cpu_backend::CPUBackend;
pub use dtype::{DType, TensorDType};
pub use metal_backend::MetalBackend;
pub use tensor::Tensor;
pub use tensor_error::TensorError;

// Backend trait for all supported backends
pub trait Backend: Sized {
    type Storage;

    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: &[usize],
        b: &Self::Storage,
        shape_b: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
    );

    fn add_arrays(a: &Self::Storage, b: &Self::Storage, shape: &[usize], dtype: DType) -> Self::Storage;

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: &[usize],
        dtype: DType,
    );

    fn from_slice<T: Copy>(data: &[T]) -> Self::Storage;
    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Vec<T>;
}
