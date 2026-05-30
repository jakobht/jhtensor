use crate::tensor::DType;

#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    TypeMismatch { expected: DType, got: DType },
    DataLengthMismatch { expected_len: usize, got_len: usize },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Tensor shapes must match for addition: {:?} != {:?}", expected, got)
            }
            TensorError::TypeMismatch { expected, got } => {
                write!(f, "Tensor dtypes must match: {:?} != {:?}", expected, got)
            }
            TensorError::DataLengthMismatch { expected_len, got_len } => {
                write!(f, "Data length must match shape: {} != {}", expected_len, got_len)
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_display {
        ($t:ident, $error:expr, $expected:expr) => {
            #[test]
            fn $t() {
                let error = $error;
                assert_eq!(format!("{}", error), $expected);
            }
        };
    }
    test_display!(test_shape_mismatch, TensorError::ShapeMismatch { expected: vec![1, 2, 3], got: vec![4, 5, 6] }, "Tensor shapes must match for addition: [1, 2, 3] != [4, 5, 6]");
    test_display!(test_type_mismatch, TensorError::TypeMismatch { expected: DType::Float32, got: DType::Int32 }, "Tensor dtypes must match: Float32 != Int32");
    test_display!(test_data_length_mismatch, TensorError::DataLengthMismatch { expected_len: 10, got_len: 20 }, "Data length must match shape: 10 != 20");
}
