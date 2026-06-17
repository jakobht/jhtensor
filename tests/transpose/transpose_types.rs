use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

macro_rules! test_transpose_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn test_transpose() {
                let a = Tensor::<$backend>::new::<$t>(
                    &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t],
                    vec![2, 3],
                )
                .unwrap();

                let result = a.transpose().unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result.shape(), vec![3, 2]);
                assert_eq!(
                    result_vec,
                    vec![1 as $t, 4 as $t, 2 as $t, 5 as $t, 3 as $t, 6 as $t]
                );
            }
        }
    };
}

mod metal {
    use super::*;

    test_transpose_for!(MetalBackend, f32);
    test_transpose_for!(MetalBackend, i32);
    test_transpose_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_transpose_for!(CPUBackend, f32);
    test_transpose_for!(CPUBackend, i32);
    test_transpose_for!(CPUBackend, i16);
}
