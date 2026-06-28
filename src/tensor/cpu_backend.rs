use crate::tensor::{Activation, Backend, DType, Shape, TensorError};

pub struct CPUBackend;

impl Backend for CPUBackend {
    type Storage = Vec<u8>;

    fn mat_mul_inplace(
        a: &Self::Storage,
        shape_a: Shape,
        b: &Self::Storage,
        shape_b: Shape,
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
        shape: Shape,
        dtype: DType,
    ) -> Result<(), TensorError> {
        let array_length = shape.product();
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
        shape: Shape,
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

    fn sum_axis_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError> {
        let dest_size = shape[if axis == 0 { 1 } else { 0 }];
        assert!(
            dest.len() >= dest_size * dtype.byte_size(),
            "Destination buffer too small for sum axis"
        );

        macro_rules! compute_sum_axis {
            ($t:ty) => {{
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), shape.product());
                    let dest_slice = std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), dest_size);

                    if axis == 1 {
                        for i in 0..shape[0] {
                            let mut sum = 0 as $t;
                            for j in 0..shape[1] {
                                sum += a_slice[i * shape[1] + j];
                            }
                            dest_slice[i] = sum;
                        }
                    } else if axis == 0 {
                        dest_slice.fill(0 as $t);

                        for i in 0..shape[0] {
                            for j in 0..shape[1] {
                                dest_slice[j] += a_slice[i * shape[1] + j];
                            }
                        }
                    } else {
                        unreachable!("Invalid axis for sum_axis, should be validated upstream")
                    }
                }
            }};
        }

        match dtype {
            DType::Float32 => compute_sum_axis!(f32),
            DType::Int32 => compute_sum_axis!(i32),
            DType::Int16 => compute_sum_axis!(i16),
        }
        Ok(())
    }

    fn broadcast_inplace(
        a: &Self::Storage,
        shape: Shape,
        dest: &mut Self::Storage,
        dest_shape: Shape,
        dtype: DType,
        axis: usize,
    ) -> Result<(), TensorError> {
        assert!(shape.ndim() == 1, "Shape must be 1 for broadcast");
        assert!(dest_shape.ndim() == 2, "Destination shape must be 2 for broadcast");
        assert!(axis == 0 || axis == 1, "Axis must be 0 or 1 for broadcast");
        assert!(
            axis == 1 && shape[0] == dest_shape[0] || axis == 0 && shape[0] == dest_shape[1],
            "Shape must match destination shape"
        );
        assert!(
            dest.len() >= dest_shape.product() * dtype.byte_size(),
            "Destination buffer too small for broadcast"
        );

        macro_rules! compute_broadcast {
            ($t:ty) => {{
                unsafe {
                    let a_slice = std::slice::from_raw_parts(a.as_ptr().cast::<$t>(), shape[0]);
                    let dest_slice =
                        std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<$t>(), dest_shape.product());

                    if axis == 1 {
                        for i in 0..shape[0] {
                            for j in 0..dest_shape[1] {
                                dest_slice[i * dest_shape[1] + j] = a_slice[i];
                            }
                        }
                    } else if axis == 0 {
                        for i in 0..dest_shape[0] {
                            for j in 0..shape[0] {
                                dest_slice[i * dest_shape[1] + j] = a_slice[j];
                            }
                        }
                    } else {
                        unreachable!("Invalid axis for sum_axis, should be validated upstream")
                    }
                }
            }};
        }

        match dtype {
            DType::Float32 => compute_broadcast!(f32),
            DType::Int32 => compute_broadcast!(i32),
            DType::Int16 => compute_broadcast!(i16),
        }
        Ok(())
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
