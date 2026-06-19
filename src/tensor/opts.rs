#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    ReLU,
}

impl Activation {
    pub fn to_shader_flag(&self) -> u32 {
        match self {
            Activation::None => 0,
            Activation::ReLU => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_shader_flag() {
        assert_eq!(Activation::None.to_shader_flag(), 0);
        assert_eq!(Activation::ReLU.to_shader_flag(), 1);
    }
}
