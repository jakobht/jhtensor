use crate::tensor::{Backend, DType};

pub struct CPUBackend;

impl Backend for CPUBackend {
    type Storage = Vec<u8>;

    fn add_arrays(a: &Self::Storage, b: &Self::Storage, shape: &[usize], dtype: DType) -> Self::Storage {
        let array_length = shape.iter().product();

        macro_rules! compute_add {
            ($t:ty) => {
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), array_length);
                    let b_slice = std::slice::from_raw_parts(b.as_ptr().cast::<$t>(), array_length);

                    let result: Vec<$t> = a_slice.iter().zip(b_slice.iter()).map(|(&x, &y)| x + y).collect();

                    let total_bytes = array_length * std::mem::size_of::<$t>();
                    std::slice::from_raw_parts(result.as_ptr().cast::<u8>(), total_bytes).to_vec()
                }
            };
        }

        match dtype {
            DType::Float32 => compute_add!(f32),
            DType::Int32 => compute_add!(i32),
            DType::Int16 => compute_add!(i16),
        }
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
