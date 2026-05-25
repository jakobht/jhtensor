use objc2_metal::{MTLTensorDataType};

mod cpu_backend;
mod metal_backend;
pub use cpu_backend::CPUBackend;
pub use metal_backend::MetalBackend;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float32,
    Int16,
    Int32,
}

impl DType {
    fn to_metal(self) -> MTLTensorDataType {
        match self {
            DType::Float32 => MTLTensorDataType::Float32,
            DType::Int16 => MTLTensorDataType::Int16,
            DType::Int32 => MTLTensorDataType::Int32,
        }
    }

    /// Returns the size of a single element in bytes
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Float32 => std::mem::size_of::<f32>(),
            DType::Int32 => std::mem::size_of::<i32>(),
            DType::Int16 => std::mem::size_of::<i16>(),
        }
    }
}

pub trait TensorDType: Copy + 'static {
    fn dtype() -> DType;
}

impl TensorDType for f32 {
    fn dtype() -> DType {
        DType::Float32
    }
}

impl TensorDType for i32 {
    fn dtype() -> DType {
        DType::Int32
    }
}

impl TensorDType for i16 {
    fn dtype() -> DType {
        DType::Int16
    }
}

pub trait Backend: Sized {
    type Storage;

    /// The backend takes the raw storage, plus the shape and dtype metadata
    fn add_arrays(a: &Self::Storage, b: &Self::Storage, shape: &[usize], dtype: DType) -> Self::Storage;

    fn from_slice<T: Copy>(data: &[T]) -> Self::Storage;
    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Vec<T>;
}

pub struct Tensor<B: Backend> {
    data: B::Storage,
    shape: Vec<usize>,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for addition");
        assert_eq!(self.dtype, other.dtype, "Tensor dtypes must match");

        let result_storage = B::add_arrays(&self.data, &other.data, &self.shape, self.dtype);

        Tensor {
            data: result_storage,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    pub fn new<T: TensorDType>(data: &[T], shape: Vec<usize>) -> Self {
        Tensor {
            data: B::from_slice(data),
            shape,
            dtype: T::dtype(),
        }
    }

    pub fn to_vec<T: TensorDType>(&self) -> Vec<T> {
        assert_eq!(self.dtype, T::dtype(), "Type mismatch");

        let num_elements = self.shape.iter().product();
        B::to_vec::<T>(&self.data, num_elements)
    }
}
