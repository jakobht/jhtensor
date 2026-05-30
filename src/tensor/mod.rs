mod cpu_backend;
mod dtype;
mod metal_backend;
mod tensor;
pub use cpu_backend::CPUBackend;
pub use dtype::{DType, TensorDType};
pub use metal_backend::MetalBackend;
pub use tensor::{Tensor, TensorError};

pub trait Backend: Sized {
    type Storage;

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
