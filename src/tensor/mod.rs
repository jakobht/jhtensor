mod cpu_backend;
mod dtype;
mod metal_backend;
mod opts;
mod shape;
mod tensor;
mod tensor_error;

pub use cpu_backend::CPUBackend;
pub use dtype::{DType, TensorDType};
pub use metal_backend::MetalBackend;
pub use opts::Activation;
pub use shape::Shape;
pub use tensor::Tensor;
pub use tensor_error::TensorError;

/// Backend trait for all supported backends
pub trait Backend: Sized {
    /// The underlying storage type for this backend.
    type Storage;

    /// Performs matrix multiplication in-place: dest = A * B + activation
    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: Shape,
        b: &Self::Storage,
        shape_b: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
        activation: Activation,
    ) -> Result<(), TensorError>;

    /// Performs element-wise addition in-place: dest = A + B
    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError>;

    /// Performs element-wise multiplication in-place: dest = A * B
    fn mul_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError>;

    /// Performs element-wise subtraction in-place: dest = A - B
    fn sub_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError>;

    /// Performs matrix transposition in-place: dest = A^T
    fn transpose_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
    ) -> Result<(), TensorError>;

    /// Performs sum reduction over the axis
    fn sum_axis_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError>;

    /// Performs broadcasting of the tensor to the destination shape
    fn broadcast_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dest_shape: Shape,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError>;

    /// Allocates a new empty storage buffer of the given size and dtype.
    fn allocate_empty(size: usize, dtype: DType) -> Result<Self::Storage, TensorError>;

    /// Converts a slice of typed data into backend storage.
    fn from_slice<T: Copy>(data: &[T]) -> Result<Self::Storage, TensorError>;

    /// Converts backend storage back into a typed vector.
    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Result<Vec<T>, TensorError>;
}
