use crate::test_fuzzy;
use jhtensor::tensor::{MetalBackend};

test_fuzzy!(MetalBackend, f32);
test_fuzzy!(MetalBackend, i32);
test_fuzzy!(MetalBackend, i16);
