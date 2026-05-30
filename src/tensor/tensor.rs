use crate::tensor::{Backend, DType, TensorDType};

#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    TypeMismatch { expected: DType, got: DType },
    DataLengthMismatch { expected_len: usize, got_len: usize },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => write!(f, "Tensor shapes must match for addition: {:?} != {:?}", expected, got),
            TensorError::TypeMismatch { expected, got } => write!(f, "Tensor dtypes must match: {:?} != {:?}", expected, got),
            TensorError::DataLengthMismatch { expected_len, got_len } => write!(f, "Data length must match shape: {} != {}", expected_len, got_len),
        }
    }
}

impl std::error::Error for TensorError {}

pub struct Tensor<B: Backend> {
    data: B::Storage,
    shape: Vec<usize>,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch { expected: self.shape.clone(), got: other.shape.clone() });
        }
        if self.dtype != other.dtype {
            return Err(TensorError::TypeMismatch { expected: self.dtype, got: other.dtype });
        }

        let result_storage = B::add_arrays(&self.data, &other.data, &self.shape, self.dtype);
        Ok(Tensor {
            data: result_storage,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn new<T: TensorDType>(data: &[T], shape: Vec<usize>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product() {
            return Err(TensorError::DataLengthMismatch { expected_len: shape.iter().product::<usize>(), got_len: data.len() });
        }

        Ok(Tensor {
            data: B::from_slice(data),
            shape,
            dtype: T::dtype(),
        })
    }

    pub fn to_vec<T: TensorDType>(&self) -> Result<Vec<T>, TensorError> {
        if self.dtype != T::dtype() {
            return Err(TensorError::TypeMismatch { expected: self.dtype, got: T::dtype() });
        }
        Ok(B::to_vec::<T>(&self.data, self.shape.iter().product()))
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{CPUBackend, DType, MetalBackend, Tensor, TensorError};


    #[test]
    fn test_tensor_new_shape_mismatch() {
        let result = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![4]);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), TensorError::DataLengthMismatch { expected_len: 4, got_len: 5 });
    }

    #[test]
    fn test_tensor_to_vec_type_mismatch() {
        let tensor = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let result = tensor.to_vec::<i32>();
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), TensorError::TypeMismatch { expected: DType::Float32, got: DType::Int32 });
    }

    mod add {
        use super::*;

        macro_rules! test_add_for {
            ($backend:ident, $t:ident) => {
                #[test]
                fn $t() {
                    let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();
                    let b = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();

                    let result = a.add(&b).unwrap();
                    let result_vec = result.to_vec::<$t>().unwrap();

                    assert_eq!(result_vec, vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]);
                }
            };
        }

        mod metal {
            use super::*;

            test_add_for!(MetalBackend, f32);
            test_add_for!(MetalBackend, i32);
            test_add_for!(MetalBackend, i16);
        }

        mod cpu {
            use super::*;

            test_add_for!(CPUBackend, f32);
            test_add_for!(CPUBackend, i32);
            test_add_for!(CPUBackend, i16);
        }

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

            let result = a.add(&b);
            assert!(result.is_err());
            assert_eq!(result.err().unwrap(), TensorError::ShapeMismatch { expected: vec![5], got: vec![4] });
        }
    }
}
