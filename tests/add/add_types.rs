use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

macro_rules! test_add_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn small() {
                let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();
                let b = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();

                let result = a.add(&b).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result_vec, vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]);
            }

            #[test]
            fn large() {
                let a = Tensor::<$backend>::new::<$t>(&[1 as $t; 2024], vec![2024]).unwrap();
                let b = Tensor::<$backend>::new::<$t>(&[1 as $t; 2024], vec![2024]).unwrap();

                let result = a.add(&b).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result_vec, vec![2 as $t; 2024]);
            }
        }
    };
}

mod metal {
    use super::*;

    test_add_for!(MetalBackend, f32);
    test_add_for!(MetalBackend, i32);
    test_add_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_add_for!(CPUBackend, f32);
    test_add_for!(CPUBackend, i32);
    test_add_for!(CPUBackend, i16);
}
