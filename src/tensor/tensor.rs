use crate::tensor::tensor_error::TensorError;
use crate::tensor::{Backend, DType, TensorDType};
pub struct Tensor<B: Backend> {
    data: B::Storage,
    shape: Vec<usize>,
    dtype: DType,
}

impl<B: Backend> Tensor<B> {
    pub fn mat_mul_inplace(&self, other: &Self, dest: &mut Self) -> Result<(), TensorError> {
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
        use crate::tensor::{CPUBackend, DType, MetalBackend, Tensor, TensorError};

        #[test]
        fn test_tensor_shape_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 25], vec![5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![2,2,2],
                    got: vec![1, 2, 2]
                }
            );
        }

        #[test]
        fn test_dest_type_mismatch() {
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<i32>(&[0; 25], vec![5, 5]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest);
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
            let a = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5,1]).unwrap();
            let b = Tensor::<MetalBackend>::new::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1,5]).unwrap();
            let mut dest = Tensor::<MetalBackend>::new::<f32>(&[0.0; 16], vec![4, 4]).unwrap();

            let result = a.mat_mul_inplace(&b, &mut dest);
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

            let result = a.mat_mul_inplace(&b, &mut dest);
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

            let result = a.mat_mul_inplace(&b, &mut dest);
            assert!(result.is_err());
            assert_eq!(
                result.err().unwrap(),
                TensorError::ShapeMismatch {
                    expected: vec![2, 1],
                    got: vec![2, 5]
                }
            );
        }

        macro_rules! test_fuzzy {
            ($backend:ident, $t:ident) => {
                mod $t {
                    use rand::{RngExt, SeedableRng};
                        use rand_chacha::ChaCha8Rng;

                    use super::*;

                    fn run_test(
                        m_range: std::ops::Range<usize>,
                        n_range: std::ops::Range<usize>,
                        k_range: std::ops::Range<usize>,
                    ) {
                        let mut rng = ChaCha8Rng::seed_from_u64(42);

                        let m = rng.random_range(m_range);
                        let n = rng.random_range(n_range);
                        let k = rng.random_range(k_range);

                        let size_a = m * k;
                        let size_b = k * n;
                        let size_dest = m * n;

                        // Fill vectors with dynamic data
                        let a_data: Vec<$t> = (0..size_a).map(|_| rng.random_range(1..10) as $t).collect();
                        let b_data: Vec<$t> = (0..size_b).map(|_| rng.random_range(1..10) as $t).collect();

                        // Ground truth from CPU
                        let cpu_a = Tensor::<CPUBackend>::new::<$t>(&a_data, vec![m, k]).unwrap();
                        let cpu_b = Tensor::<CPUBackend>::new::<$t>(&b_data, vec![k, n]).unwrap();
                        let mut cpu_dest =
                            Tensor::<CPUBackend>::new::<$t>(&vec![0 as $t; size_dest], vec![m, n]).unwrap();
                        cpu_a.mat_mul_inplace(&cpu_b, &mut cpu_dest).unwrap();

                        // Backend under test
                        let metal_a = Tensor::<$backend>::new::<$t>(&a_data, vec![m, k]).unwrap();
                        let metal_b = Tensor::<$backend>::new::<$t>(&b_data, vec![k, n]).unwrap();
                        let mut backend_dest =
                            Tensor::<$backend>::new::<$t>(&vec![0 as $t; size_dest], vec![m, n]).unwrap();
                        metal_a.mat_mul_inplace(&metal_b, &mut backend_dest).unwrap();

                        assert_eq!(
                            backend_dest.to_vec::<$t>().unwrap(),
                            cpu_dest.to_vec::<$t>().unwrap(),
                        );
                    }

                    #[test]
                    fn run_small_test() {
                        run_test(1..16, 1..16, 1..16);
                    }

                    #[test]
                    fn run_medium_test() {
                        run_test(16..64, 16..64, 16..64);
                    }

                    #[test]
                    fn run_large_test() {
                        run_test(64..128, 64..128, 64..128);
                    }
                }
            };
        }

        mod metal_fuzzy {
            use super::*;

            test_fuzzy!(MetalBackend, f32);
            test_fuzzy!(MetalBackend, i32);
            test_fuzzy!(MetalBackend, i16);
        }

        macro_rules! test_mat_mul_for {
            ($backend:ident, $t:ident) => {
                mod $t {
                    use super::*;

                    #[test]
                    fn test_mat_mul_inplace_dot_product() {
                        let a =
                            Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![1, 5])
                                .unwrap();
                        let b =
                            Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5, 1])
                                .unwrap();
                        let mut dest = Tensor::<$backend>::new::<$t>(&[0 as $t; 1], vec![1, 1]).unwrap();

                        a.mat_mul_inplace(&b, &mut dest).unwrap();

                        assert_eq!(dest.to_vec::<$t>().unwrap(), vec![55 as $t]);
                    }

                    #[test]
                    fn test_mat_mul_inplace() {
                        let a = Tensor::<$backend>::new::<$t>(
                            &[
                                1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t, 7 as $t, 8 as $t, 9 as $t,
                                10 as $t, 11 as $t, 12 as $t, 13 as $t, 14 as $t, 15 as $t,
                            ],
                            vec![5, 3],
                        )
                        .unwrap();
                        let b = Tensor::<$backend>::new::<$t>(
                            &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t],
                            vec![3, 2],
                        )
                        .unwrap();
                        let mut dest = Tensor::<$backend>::new::<$t>(&[0 as $t; 10], vec![5, 2]).unwrap();

                        a.mat_mul_inplace(&b, &mut dest).unwrap();

                        assert_eq!(
                            dest.to_vec::<$t>().unwrap(),
                            vec![
                                22 as $t, 28 as $t, 49 as $t, 64 as $t, 76 as $t, 100 as $t, 103 as $t, 136 as $t,
                                130 as $t, 172 as $t
                            ]
                        );
                    }
                }
            };
        }

        mod metal {
            use super::*;

            test_mat_mul_for!(MetalBackend, f32);
            test_mat_mul_for!(MetalBackend, i32);
            test_mat_mul_for!(MetalBackend, i16);
        }

        mod cpu {
            use super::*;

            test_mat_mul_for!(CPUBackend, f32);
            test_mat_mul_for!(CPUBackend, i32);
            test_mat_mul_for!(CPUBackend, i16);
        }
    }

    mod add {
        use super::*;

        macro_rules! test_add_for {
            ($backend:ident, $t:ident) => {
                #[test]
                fn $t() {
                    let a =
                        Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();
                    let b =
                        Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();

                    let result = a.add(&b).unwrap();
                    let result_vec = result.to_vec::<$t>().unwrap();

                    assert_eq!(
                        result_vec,
                        vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]
                    );
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

        macro_rules! test_add_inplace_for {
            ($backend:ident, $t:ident) => {
                #[test]
                fn $t() {
                    let mut a =
                        Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();
                    let b =
                        Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();

                    let mut result = Tensor::<$backend>::new::<$t>(&[0 as $t; 5], vec![5]).unwrap();
                    a.add_inplace(&b, &mut result).unwrap();
                    let result_vec = result.to_vec::<$t>().unwrap();
                    assert_eq!(
                        result_vec,
                        vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]
                    );
                }
            };
        }

        mod metal {
            use super::*;

            test_add_inplace_for!(MetalBackend, f32);
            test_add_inplace_for!(MetalBackend, i32);
            test_add_inplace_for!(MetalBackend, i16);
        }

        mod cpu {
            use super::*;

            test_add_inplace_for!(CPUBackend, f32);
            test_add_inplace_for!(CPUBackend, i32);
            test_add_inplace_for!(CPUBackend, i16);
        }

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
