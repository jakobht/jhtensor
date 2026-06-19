use crate::tensor::DType;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    TypeMismatch { expected: DType, got: DType },
    DataLengthMismatch { expected_len: usize, got_len: usize },
    DimensionMismatch { expected: usize, got: usize },
    BackendFailure(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Tensor shapes do not match: expected {:?}, got {:?}", expected, got)
            }
            TensorError::TypeMismatch { expected, got } => {
                write!(f, "Tensor dtypes do not match: expected {:?}, got {:?}", expected, got)
            }
            TensorError::DataLengthMismatch { expected_len, got_len } => {
                write!(
                    f,
                    "Data length does not match shape: expected {}, got {}",
                    expected_len, got_len
                )
            }
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "Invalid tensor dimensions: expected {}, got {}", expected, got)
            }
            TensorError::BackendFailure(msg) => {
                write!(f, "Backend operation failed: {}", msg)
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

    test_display!(
        test_shape_mismatch_display,
        TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![2, 4]
        },
        "Tensor shapes do not match: expected [2, 3], got [2, 4]"
    );

    test_display!(
        test_type_mismatch_display,
        TensorError::TypeMismatch {
            expected: DType::Float32,
            got: DType::Int32
        },
        "Tensor dtypes do not match: expected Float32, got Int32"
    );

    test_display!(
        test_data_length_mismatch_display,
        TensorError::DataLengthMismatch {
            expected_len: 10,
            got_len: 5
        },
        "Data length does not match shape: expected 10, got 5"
    );

    test_display!(
        test_dimension_mismatch_display,
        TensorError::DimensionMismatch { expected: 2, got: 3 },
        "Invalid tensor dimensions: expected 2, got 3"
    );

    test_display!(
        test_backend_failure_display,
        TensorError::BackendFailure("GPU out of memory".into()),
        "Backend operation failed: GPU out of memory"
    );
}
