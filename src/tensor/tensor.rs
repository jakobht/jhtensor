use crate::tensor::{Backend, DType, TensorDType};

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
