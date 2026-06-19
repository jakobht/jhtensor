use crate::tensor::{Activation, Backend, DType, TensorError};

pub struct CPUBackend;

impl Backend for CPUBackend {
    type Storage = Vec<u8>;

    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: &[usize],
        b: &Self::Storage,
        shape_b: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
        activation: Activation,
    ) -> Result<(), TensorError> {
        assert!(
            dest.len() >= shape_a[0] * shape_b[1] * dtype.byte_size(),
            "Destination buffer too small for matmul"
        );

        macro_rules! compute_mat_mul {
            ($t:ty) => {{
                unsafe {
                    let a_ptr = a.as_ptr().cast::<$t>();
                    let b_ptr = b.as_ptr().cast::<$t>();
                    let dest_ptr = dest.as_mut_ptr().cast::<$t>();

                    let a_slice = std::slice::from_raw_parts(a_ptr, shape_a[0] * shape_a[1]);
                    let b_slice = std::slice::from_raw_parts(b_ptr, shape_b[0] * shape_b[1]);
                    let dest_slice = std::slice::from_raw_parts_mut(dest_ptr, shape_a[0] * shape_b[1]);

                    for i in 0..shape_a[0] {
                        for j in 0..shape_b[1] {
                            let mut sum = 0 as $t;
                            for k in 0..shape_a[1] {
                                sum += a_slice[i * shape_a[1] + k] * b_slice[k * shape_b[1] + j];
                            }
                            dest_slice[i * shape_b[1] + j] = match activation {
                                Activation::None => sum,
                                Activation::ReLU => sum.max(0 as $t),
                            };
                        }
                    }
                }
            }};
        }

        match dtype {
            DType::Float32 => compute_mat_mul!(f32),
            DType::Int32 => compute_mat_mul!(i32),
            DType::Int16 => compute_mat_mul!(i16),
        }
        Ok(())
    }

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: &[usize],
        dtype: DType,
    ) -> Result<(), TensorError> {
        let array_length = shape.iter().product();
        assert!(
            dest.len() >= array_length * dtype.byte_size(),
            "Destination buffer too small for addition"
        );

        macro_rules! compute_add {
            ($t:ty) => {{
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), array_length);
                    let b_slice = std::slice::from_raw_parts(b.as_ptr().cast::<$t>(), array_length);
                    let dest_slice = std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), array_length);

                    for i in 0..array_length {
                        dest_slice[i] = a_slice[i] + b_slice[i];
                    }
                }
            }};
        }

        match dtype {
            DType::Float32 => compute_add!(f32),
            DType::Int32 => compute_add!(i32),
            DType::Int16 => compute_add!(i16),
        }
        Ok(())
    }

    fn transpose_inplace(
        a: &Self::Storage,
        shape: &[usize],
        dest: &mut Self::Storage,
        dtype: DType,
    ) -> Result<(), TensorError> {
        assert!(
            dest.len() >= shape[0] * shape[1] * dtype.byte_size(),
            "Destination buffer too small for transpose"
        );

        macro_rules! compute_transpose {
            ($t:ty) => {{
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), shape[0] * shape[1]);
                    let dest_slice =
                        std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), shape[0] * shape[1]);

                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            dest_slice[j * shape[0] + i] = a_slice[i * shape[1] + j];
                        }
                    }
                }
            }};
        }

        match dtype {
            DType::Float32 => compute_transpose!(f32),
            DType::Int32 => compute_transpose!(i32),
            DType::Int16 => compute_transpose!(i16),
        }
        Ok(())
    }

    fn sum_axis_inplace(a: &Self::Storage, shape: &[usize], dest: &mut Self::Storage, dtype: DType, axis: usize) -> Result<(), TensorError> {
        unimplemented!()
    }

    #[inline(always)]
    fn allocate_empty(size: usize, dtype: DType) -> Result<Self::Storage, TensorError> {
        Ok(vec![0; size * dtype.byte_size()])
    }

    #[inline(always)]
    fn from_slice<T: Copy>(data: &[T]) -> Result<Self::Storage, TensorError> {
        unsafe {
            let total_bytes = data.len() * std::mem::size_of::<T>();
            Ok(std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), total_bytes).to_vec())
        }
    }

    #[inline(always)]
    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Result<Vec<T>, TensorError> {
        unsafe { Ok(std::slice::from_raw_parts(storage.as_ptr().cast::<T>(), num_elements).to_vec()) }
    }
}
