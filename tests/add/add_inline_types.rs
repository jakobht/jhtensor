use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

macro_rules! test_add_inplace_for {
    ($backend:ident, $t:ident) => {
        #[test]
        fn $t() {
            let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();
            let b = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5]).unwrap();

            let mut result = Tensor::<$backend>::new::<$t>(&[0 as $t; 5], vec![5]).unwrap();
            a.add_inplace(&b, &mut result).unwrap();
            let result_vec = result.to_vec::<$t>().unwrap();
            assert_eq!(result_vec, vec![2 as $t, 4 as $t, 6 as $t, 8 as $t, 10 as $t]);
        }
    };
}

mod metal {
    use super::*;

    test_add_inplace_for!(MetalBackend, f32);
    test_add_inplace_for!(MetalBackend, i32);
    test_add_inplace_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_add_inplace_for!(CPUBackend, f32);
    test_add_inplace_for!(CPUBackend, i32);
    test_add_inplace_for!(CPUBackend, i16);
}
