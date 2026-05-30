use crate::tensor::{Backend, DType};

pub struct CPUBackend;

impl CPUBackend {
    fn new_empty(num_elements: usize, dtype: DType) -> <Self as Backend>::Storage {
        vec![0; num_elements * dtype.byte_size()]
    }
}

impl Backend for CPUBackend {
    type Storage = Vec<u8>;

    fn add_arrays(a: &Self::Storage, b: &Self::Storage, shape: &[usize], dtype: DType) -> Self::Storage {
        let array_length = shape.iter().product();
        let mut dest = Self::new_empty(array_length, dtype);
        Self::add_arrays_inplace(a, b, &mut dest, shape, dtype);
        dest
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
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Backend, CPUBackend, DType};

    mod add_arrays {
        use super::*;

        macro_rules! test_type {
            ($t:ident, $dtype:expr) => {
                #[test]
                fn $t() {
                    let a = CPUBackend::from_slice(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t]);
                    let b = CPUBackend::from_slice(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t]);

                    let result_bytes = CPUBackend::add_arrays(&a, &b, &[5], $dtype);

                    let result_vec = CPUBackend::to_vec::<$t>(&result_bytes, 5);

                    assert_eq!(
                        result_vec,
                        vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]
                    );
                }
            };
        }
        test_type!(f32, DType::Float32);
        test_type!(i32, DType::Int32);
        test_type!(i16, DType::Int16);
    }
}
