use crate::tensor::{Activation, Backend, DType};

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
    ) {
        // Naive implementation for now

        assert_eq!(shape_a.len(), 2);
        assert_eq!(shape_b.len(), 2);
        assert_eq!(shape_a[1], shape_b[0]);
        assert!(dest.len() >= shape_a[0] * shape_b[1] * dtype.byte_size());

        macro_rules! compute_mat_mul {
            ($t:ty) => {{
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), shape_a[0] * shape_a[1]);
                    let b_slice = std::slice::from_raw_parts(b.as_ptr().cast::<$t>(), shape_b[0] * shape_b[1]);
                    let dest_slice =
                        std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), shape_a[0] * shape_b[1]);

                    for i in 0..shape_a[0] {
                        for j in 0..shape_b[1] {
                            let mut sum = 0 as $t;
                            for k in 0..shape_a[1] {
                                sum += a_slice[i * shape_a[1] + k] * b_slice[k * shape_b[1] + j];
                            }
                            dest_slice[i * shape_b[1] + j] = match activation {
                                Activation::None => sum,
                                Activation::ReLU => {
                                    if sum > 0 as $t {
                                        sum
                                    } else {
                                        0 as $t
                                    }
                                }
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
    }

    fn add_arrays_inplace(
        a: &Self::Storage,
        b: &Self::Storage,
        dest: &mut Self::Storage,
        shape: &[usize],
        dtype: DType,
    ) {
        let array_length = shape.iter().product();

        macro_rules! compute_add {
            ($t:ty) => {
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), array_length);
                    let b_slice = std::slice::from_raw_parts(b.as_ptr().cast::<$t>(), array_length);
                    let dest_slice = std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), array_length);

                    for i in 0..array_length {
                        dest_slice[i] = a_slice[i] + b_slice[i];
                    }
                }
            };
        }

        match dtype {
            DType::Float32 => compute_add!(f32),
            DType::Int32 => compute_add!(i32),
            DType::Int16 => compute_add!(i16),
        }
    }

    fn transpose_inplace(tensor: &Self::Storage, shape: &[usize], dest: &mut Self::Storage) {
        unimplemented!()
    }

    fn allocate_empty(size: usize, dtype: DType) -> Self::Storage {
        vec![0; size * dtype.byte_size()]
    }

    fn from_slice<T: Copy>(data: &[T]) -> Self::Storage {
        unsafe {
            let total_bytes = data.len() * std::mem::size_of::<T>();
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), total_bytes).to_vec()
        }
    }

    fn to_vec<T: Copy>(storage: &Self::Storage, num_elements: usize) -> Vec<T> {
        unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<T>(), num_elements).to_vec() }
    }
}
