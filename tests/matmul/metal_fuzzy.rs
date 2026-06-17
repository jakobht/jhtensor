use crate::test_fuzzy;
use jhtensor::tensor::{Activation, MetalBackend};

mod linear {
    use super::*;

    test_fuzzy!(MetalBackend, f32, Activation::None);
    test_fuzzy!(MetalBackend, i32, Activation::None);
    test_fuzzy!(MetalBackend, i16, Activation::None);
}

mod relu {
    use super::*;

    test_fuzzy!(MetalBackend, f32, Activation::ReLU);
    test_fuzzy!(MetalBackend, i32, Activation::ReLU);
    test_fuzzy!(MetalBackend, i16, Activation::ReLU);
}
