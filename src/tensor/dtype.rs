#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float32,
    Int16,
    Int32,
}

impl DType {
    /// Returns the size of a single element in bytes
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Float32 => std::mem::size_of::<f32>(),
            DType::Int32 => std::mem::size_of::<i32>(),
            DType::Int16 => std::mem::size_of::<i16>(),
        }
    }
}

pub trait TensorDType: Copy + 'static {
    fn dtype() -> DType;
}

impl TensorDType for f32 {
    fn dtype() -> DType {
        DType::Float32
    }
}

impl TensorDType for i32 {
    fn dtype() -> DType {
        DType::Int32
    }
}

impl TensorDType for i16 {
    fn dtype() -> DType {
        DType::Int16
    }
}
