use crate::tensor::tensor_error::TensorError;
use crate::tensor::{Activation, Backend, DType, TensorDType};
pub struct Tensor<B: Backend> {
    data: B::Storage,
    shape: Vec<usize>,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn mat_mul_inplace(&self, other: &Self, dest: &mut Self, activation: Activation) -> Result<(), TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 || dest.shape.len() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![2, 2, 2],
                got: vec![self.shape.len(), other.shape.len(), dest.shape.len()],
            });
        }

        if self.shape[1] != other.shape[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape[0], other.shape[1]],
                got: vec![self.shape[1], other.shape[0]],
            });
        }
        if dest.shape != vec![self.shape[0], other.shape[1]] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape[0], other.shape[1]],
                got: dest.shape.clone(),
            });
        }
        if self.dtype != other.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: other.dtype,
            });
        }
        if dest.dtype != self.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: dest.dtype,
            });
        }
        B::mat_mul_inplace(
            &self.data,
            &self.shape,
            &other.data,
            &other.shape,
            &mut dest.data,
            self.dtype,
            activation,
        );
        Ok(())
    }

    pub fn add_inplace(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        if dest.shape != self.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: dest.shape.clone(),
            });
        }
        if self.dtype != other.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: other.dtype,
            });
        }
        if dest.dtype != self.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: dest.dtype,
            });
        }
        B::add_arrays_inplace(&self.data, &other.data, &mut dest.data, &self.shape, self.dtype);
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        if self.dtype != other.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: other.dtype,
            });
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
            return Err(TensorError::DataLengthMismatch {
                expected_len: shape.iter().product::<usize>(),
                got_len: data.len(),
            });
        }

        Ok(Tensor {
            data: B::from_slice(data),
            shape,
            dtype: T::dtype(),
        })
    }

    pub fn to_vec<T: TensorDType>(&self) -> Result<Vec<T>, TensorError> {
        if self.dtype != T::dtype() {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: T::dtype(),
            });
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
        assert_eq!(
            result.err().unwrap(),
            TensorError::DataLengthMismatch {
                expected_len: 4,
                got_len: 5
            }
        );
    }

    #[test]
    fn test_tensor_to_vec_type_mismatch() {
        let tensor = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let result = tensor.to_vec::<i32>();
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            TensorError::TypeMismatch {
                expected: DType::Float32,
                got: DType::Int32
            }
        );
    }

    mod mat_mul {
        use crate::tensor::{Activation, DType, MetalBackend, Tensor, TensorError};

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 25], vec![5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![2, 2, 2],
                    got: vec![1, 2, 2]
                }
            );
        }

        #[test]
        fn test_dest_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 25], vec![5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32
                }
            );
        }

        #[test]
        fn test_dest_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 16], vec![4, 4]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![5, 5],
                    got: vec![4, 4]
                }
            );
        }

        #[test]
        fn test_other_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<i32>(&[1, 2, 3, 4, 5], vec![1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 25], vec![5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32
                }
            );
        }

        #[test]
        fn test_other_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 10], vec![2, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![2, 1],
                    got: vec![2, 5]
                }
            );
        }
    }

    mod add {
        use super::*;

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();

            let result = a.add(&b);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![5],
                    got: vec![4]
                }
            );
        }

        #[test]
        fn test_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<i32>(&[1, 2, 3, 4, 5], vec![5]).unwrap();

            let result = a.add(&b);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32
                }
            );
        }
    }

    mod add_inplace {
        use super::*;

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 5], vec![5]).unwrap();

            let result = a.add_inplace(&b, &mut dest);
            assert_eq!(
                result.unwrap_err(),
                TensorError::ShapeMismatch {
                    expected: vec![5],
                    got: vec![4]
                }
            );
        }

        #[test]
        fn test_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<i32>(&[1, 2, 3, 4, 5], vec![5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 5], vec![5]).unwrap();

            let result = a.add_inplace(&b, &mut dest);
            assert_eq!(
                result.unwrap_err(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32
                }
            );
        }

        #[test]
        fn test_dest_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 4], vec![4]).unwrap();

            let result = a.add_inplace(&b, &mut dest);
            assert_eq!(
                result.unwrap_err(),
                TensorError::ShapeMismatch {
                    expected: vec![5],
                    got: vec![4]
                }
            );
        }

        #[test]
        fn test_dest_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 5], vec![5]).unwrap();

            let result = a.add_inplace(&b, &mut dest);
            assert_eq!(
                result.unwrap_err(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32
                }
            );
        }
    }
}
