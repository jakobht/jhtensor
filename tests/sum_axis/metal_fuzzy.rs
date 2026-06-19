use crate::test_fuzzy;
use jhtensor::tensor::{Activation, MetalBackend};

mod linear {
    use super::*;

    test_fuzzy!(MetalBackend, f32);
    test_fuzzy!(MetalBackend, i32);
    test_fuzzy!(MetalBackend, i16);
}

mod relu {
    use super::*;

    test_fuzzy!(MetalBackend, f32);
    test_fuzzy!(MetalBackend, i32);
    test_fuzzy!(MetalBackend, i16);
}
