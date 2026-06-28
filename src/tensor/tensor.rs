use crate::tensor::tensor_error::TensorError;
use crate::tensor::{Activation, Backend, DType, Shape, TensorDType};

pub struct Tensor<B: Backend> {
    data: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn mat_mul_inplace(&self, other: &Self, dest: &mut Self, activation: Activation) -> Result<(), TensorError> {
        if self.shape.ndim() != 2 || other.shape.ndim() != 2 || dest.shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.shape.ndim(),
            });
        }
        if self.shape[1] != other.shape[0] {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[self.shape[0], other.shape[1]]),
                got: Shape::new(&[self.shape[0], self.shape[1]]),
            });
        }
        if dest.shape != Shape::new(&[self.shape[0], other.shape[1]]) {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[self.shape[0], other.shape[1]]),
                got: dest.shape,
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
            self.shape,
            &other.data,
            other.shape,
            &mut dest.data,
            self.dtype,
            activation,
        )?;
        Ok(())
    }

    fn validate_elementwise(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape,
                got: other.shape,
            });
        }
        if dest.shape != self.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape,
                got: dest.shape,
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
        Ok(())
    }

    pub fn add_inplace(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
        self.validate_elementwise(other, dest)?;
        B::add_arrays_inplace(&self.data, &other.data, &mut dest.data, self.shape, self.dtype)?;
        Ok(())
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.shape.ndim(),
            });
        }
        let result = B::allocate_empty(self.shape.product(), self.dtype)?;

        let result_shape = Shape::new(&[self.shape[1], self.shape[0]]);

        let mut result_tensor = Tensor {
            data: result,
            shape: result_shape,
            dtype: self.dtype,
        };
        self.transpose_inplace(&mut result_tensor)?;
        Ok(result_tensor)
    }

    pub fn transpose_inplace(&self, dest: &mut Self) -> Result<(), TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.shape.ndim(),
            });
        }
        if dest.shape != Shape::new(&[self.shape[1], self.shape[0]]) {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[self.shape[1], self.shape[0]]),
                got: dest.shape,
            });
        }
        if dest.dtype != self.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: dest.dtype,
            });
        }
        B::transpose_inplace(&self.data, self.shape, &mut dest.data, self.dtype)?;
        Ok(())
    }

    pub fn mul_inplace(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
        self.validate_elementwise(other, dest)?;
        B::mul_arrays_inplace(&self.data, &other.data, &mut dest.data, self.shape, self.dtype)?;
        Ok(())
    }

    pub fn sub_inplace(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
        self.validate_elementwise(other, dest)?;
        B::sub_arrays_inplace(&self.data, &other.data, &mut dest.data, self.shape, self.dtype)?;
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        let result = B::allocate_empty(self.shape.iter().product(), self.dtype)?;
        let mut result_tensor = Tensor {
            data: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        };
        self.add_inplace(other, &mut result_tensor)?;
        Ok(result_tensor)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        let result = B::allocate_empty(self.shape.iter().product(), self.dtype)?;
        let mut result_tensor = Tensor {
            data: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        };
        self.mul_inplace(other, &mut result_tensor)?;
        Ok(result_tensor)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        let result = B::allocate_empty(self.shape.iter().product(), self.dtype)?;
        let mut result_tensor = Tensor {
            data: result,
            shape: self.shape.clone(),
            dtype: self.dtype,
        };
        self.sub_inplace(other, &mut result_tensor)?;
        Ok(result_tensor)
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.shape.ndim(),
            });
        }
        if axis >= self.shape.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: axis + 1,
                got: self.shape.ndim(),
            });
        }

        let dest_size = self.shape.dim(if axis == 0 { 1 } else { 0 }).unwrap();
        let result = B::allocate_empty(dest_size, self.dtype)?;
        let mut result_tensor = Tensor {
            data: result,
            shape: Shape::new(&[dest_size]),
            dtype: self.dtype,
        };

        self.sum_axis_inplace(axis, &mut result_tensor)?;
        Ok(result_tensor)
    }

    pub fn sum_axis_inplace(&self, axis: usize, dest: &mut Self) -> Result<(), TensorError> {
        if self.shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: self.shape.ndim(),
            });
        }
        if axis >= self.shape.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: axis + 1,
                got: self.shape.ndim(),
            });
        }
        let dest_size = self.shape.dim(if axis == 0 { 1 } else { 0 }).unwrap();
        if dest.shape != Shape::new(&[dest_size]) {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[dest_size]),
                got: dest.shape,
            });
        }
        if dest.dtype != self.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: dest.dtype,
            });
        }
        B::sum_axis_inplace(&self.data, self.shape, &mut dest.data, self.dtype, axis)?;
        Ok(())
    }

    pub fn broadcast(&self, shape: impl AsRef<[usize]>, axis: usize) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);
        self.validate_broadcast_params(shape, axis)?;

        let dest_size = shape.product();
        let result = B::allocate_empty(dest_size, self.dtype)?;
        let mut result_tensor = Tensor {
            data: result,
            shape: shape,
            dtype: self.dtype,
        };

        self.broadcast_inplace(axis, &mut result_tensor)?;

        return Ok(result_tensor);
    }

    pub fn broadcast_inplace(&self, axis: usize, dest: &mut Self) -> Result<(), TensorError> {
        if self.dtype != dest.dtype {
            return Err(TensorError::TypeMismatch {
                expected: self.dtype,
                got: dest.dtype,
            });
        }
        self.validate_broadcast_params(dest.shape, axis)?;

        B::broadcast_inplace(&self.data, self.shape, &mut dest.data, dest.shape, self.dtype, axis)?;
        Ok(())
    }

    fn validate_broadcast_params(&self, shape: Shape, axis: usize) -> Result<(), TensorError> {
        if self.shape.ndim() != 1 {
            return Err(TensorError::DimensionMismatch {
                expected: 1,
                got: self.shape.ndim(),
            });
        }
        if shape.ndim() != 2 {
            return Err(TensorError::DimensionMismatch {
                expected: 2,
                got: shape.ndim(),
            });
        }
        if axis >= shape.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: shape.ndim(),
                got: axis,
            });
        }
        if axis == 0 && self.shape[0] != shape[1] {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[shape[0], self.shape[0]]),
                got: shape,
            });
        }
        if axis == 1 && self.shape[0] != shape[0] {
            return Err(TensorError::ShapeMismatch {
                expected: Shape::new(&[self.shape[0], shape[1]]),
                got: shape,
            });
        }
        Ok(())
    }

    pub fn new<T: TensorDType>(data: &[T], shape: impl AsRef<[usize]>) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);

        if data.len() != shape.product() {
            return Err(TensorError::DataLengthMismatch {
                expected_len: shape.product(),
                got_len: data.len(),
            });
        }

        Ok(Tensor {
            data: B::from_slice(data)?,
            shape: shape,
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
        Ok(B::to_vec::<T>(&self.data, self.shape.iter().product())?)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{DType, MetalBackend, Tensor, TensorError};

    #[test]
    fn test_tensor_new_shape_mismatch() {
        let result = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [4]);
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
        let tensor = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
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
        use crate::tensor::{Activation, DType, MetalBackend, Shape, Tensor, TensorError};

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5, 1]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 25], [5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 1 }
            );
        }

        #[test]
        fn test_dest_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 25], [5, 5]).unwrap();

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
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 16], [4, 4]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([5, 5]),
                    got: Shape::new([4, 4])
                }
            );
        }

        #[test]
        fn test_other_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<i32>(&[1, 2, 3, 4, 5], [1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 25], [5, 5]).unwrap();

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
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5, 1]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 10], [2, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest, Activation::None);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([2, 1]),
                    got: Shape::new([2, 2])
                }
            );
        }
    }

    macro_rules! elementwise_op_tests {
        ($non_inplace_mod:ident, $inplace_mod:ident, $method:ident, $inplace_method:ident) => {
            mod $non_inplace_mod {
                use crate::tensor::Shape;

                use super::*;

                #[test]
                fn test_tensor_shape_mismatch() {
                    let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [4]).unwrap();

                    let result = a.$method(&b);
                    assert!(result.is_err());
                    assert_eq!(
                        result.err().unwrap(),
                        TensorError::ShapeMismatch {
                            expected: Shape::new([5]),
                            got: Shape::new([4])
                        }
                    );
                }

                #[test]
                fn test_tensor_type_mismatch() {
                    let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let b = Tensor::<MetalBackend>::new::<i32>(&[1, 2, 3, 4, 5], [5]).unwrap();

                    let result = a.$method(&b);
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

            mod $inplace_mod {
                use crate::tensor::Shape;

                use super::*;

                #[test]
                fn test_dest_shape_mismatch() {
                    let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 4], [4]).unwrap();

                    let result = a.$inplace_method(&b, &mut dest);
                    assert_eq!(
                        result.unwrap_err(),
                        TensorError::ShapeMismatch {
                            expected: Shape::new([5]),
                            got: Shape::new([4])
                        }
                    );
                }

                #[test]
                fn test_dest_type_mismatch() {
                    let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
                    let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 5], [5]).unwrap();

                    let result = a.$inplace_method(&b, &mut dest);
                    assert_eq!(
                        result.unwrap_err(),
                        TensorError::TypeMismatch {
                            expected: DType::Float32,
                            got: DType::Int32
                        }
                    );
                }
            }
        };
    }

    elementwise_op_tests!(add, add_inplace, add, add_inplace);
    elementwise_op_tests!(mul, mul_inplace, mul, mul_inplace);
    elementwise_op_tests!(sub, sub_inplace, sub, sub_inplace);

    mod transpose {
        use crate::tensor::Shape;

        use super::*;

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let result = a.transpose();
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 1 },
            );
        }

        #[test]
        fn test_tensor_shape_mismatch_inplace() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 6], [2, 3]).unwrap();
            let result = a.transpose_inplace(&mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 1 },
            );
        }

        #[test]
        fn test_tensor_dest_shape_inline_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 6], [2, 3]).unwrap();
            let result = a.transpose_inplace(&mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([3, 2]),
                    got: Shape::new([2, 3]),
                },
            );
        }

        #[test]
        fn test_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 6], [3, 2]).unwrap();
            let result = a.transpose_inplace(&mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32,
                },
            );
        }
    }

    mod sum_axis {
        use super::*;

        #[test]
        fn test_tensor_wrong_dimension() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let result = a.sum_axis(0);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 1 },
            );
        }

        #[test]
        fn test_axis_out_of_bounds() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let result = a.sum_axis(2);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 3, got: 2 },
            );
        }
    }

    mod sum_axis_inplace {
        use crate::tensor::Shape;

        use super::*;

        #[test]
        fn test_tensor_wrong_dimension() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 5], [5]).unwrap();
            let result = a.sum_axis_inplace(0, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 1 },
            );
        }

        #[test]
        fn test_axis_out_of_bounds() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 2], [2]).unwrap();
            let result = a.sum_axis_inplace(2, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 3, got: 2 },
            );
        }

        #[test]
        fn test_dest_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 4], [4]).unwrap();
            let result = a.sum_axis_inplace(0, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([2]),
                    got: Shape::new([4]),
                },
            );
        }

        #[test]
        fn test_dest_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 2], [2]).unwrap();
            let result = a.sum_axis_inplace(0, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32,
                },
            );
        }
    }

    mod broadcast {
        use crate::tensor::Shape;

        use super::*;

        #[test]
        fn test_tensor_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 4], [2, 2]).unwrap();
            let result = a.broadcast_inplace(0, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::TypeMismatch {
                    expected: DType::Float32,
                    got: DType::Int32,
                },
            );
        }

        #[test]
        fn test_self_wrong_dimension() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [2, 5])
                .unwrap();
            let result = a.broadcast([5, 5], 0);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 1, got: 2 },
            );
        }

        #[test]
        fn test_shape_wrong_dimension() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]).unwrap();
            let result = a.broadcast([5, 5, 5], 0);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 3 },
            );
        }

        #[test]
        fn test_axis_out_of_bounds() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [4]).unwrap();
            let result = a.broadcast([4, 4], 2);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::DimensionMismatch { expected: 2, got: 2 },
            );
        }

        #[test]
        fn test_shape_mismatch_axis_0() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [4]).unwrap();
            let result = a.broadcast([5, 3], 0);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([5, 4]),
                    got: Shape::new([5, 3])
                },
            );
        }

        #[test]
        fn test_shape_mismatch_axis_1() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0], [4]).unwrap();
            let result = a.broadcast([5, 5], 1);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: Shape::new([4, 5]),
                    got: Shape::new([5, 5])
                },
            );
        }
    }
}
