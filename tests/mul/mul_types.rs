use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

macro_rules! test_mul_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn small() {
                let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], [5]).unwrap();
                let b = Tensor::<$backend>::new::<$t>(&[2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t], [5]).unwrap();

                let result = a.mul(&b).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result_vec, vec![2 as $t, 6 as $t, 12 as $t, 20 as $t, 30 as $t]);
            }

            #[test]
            fn large() {
                let a = Tensor::<$backend>::new::<$t>(&[2 as $t; 2024], [2024]).unwrap();
                let b = Tensor::<$backend>::new::<$t>(&[3 as $t; 2024], [2024]).unwrap();

                let result = a.mul(&b).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result_vec, vec![6 as $t; 2024]);
            }
        }
    };
}

mod metal {
    use super::*;

    test_mul_for!(MetalBackend, f32);
    test_mul_for!(MetalBackend, i32);
    test_mul_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_mul_for!(CPUBackend, f32);
    test_mul_for!(CPUBackend, i32);
    test_mul_for!(CPUBackend, i16);
}
