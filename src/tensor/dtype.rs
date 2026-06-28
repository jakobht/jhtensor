#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Returns a Metal pipeline name for the given operation prefix.
    pub fn pipeline_name(&self, prefix: &str) -> String {
        let suffix = match self {
            DType::Float32 => "f32",
            DType::Int32 => "i32",
            DType::Int16 => "i16",
        };
        format!("{}_{}", prefix, suffix)
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
