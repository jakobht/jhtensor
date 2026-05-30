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

#[cfg(test)]
mod tests {
    use crate::tensor::{CPUBackend, MetalBackend, Tensor};

    mod add {
        use super::*;

        macro_rules! test_add_for {
            ($backend:ident, $t:ident) => {
                #[test]
                fn $t() {
                    let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]);
                    let b = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]);

                    let result = a.add(&b);
                    let result_vec = result.to_vec::<$t>();

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
    }
}
