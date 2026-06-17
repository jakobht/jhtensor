use jhtensor::tensor::{Activation, CPUBackend, MetalBackend, Tensor};

macro_rules! test_mat_mul_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn test_mat_mul_inplace_dot_product() {
                let a =
                    Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![1, 5])
                        .unwrap();
                let b =
                    Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t], vec![5, 1])
                        .unwrap();
                let mut dest = Tensor::<$backend>::new::<$t>(&[0 as $t; 1], vec![1, 1]).unwrap();

                a.mat_mul_inplace(&b, &mut dest, Activation::None).unwrap();

                assert_eq!(dest.to_vec::<$t>().unwrap(), vec![55 as $t]);
            }

            #[test]
            fn test_mat_mul_inplace() {
                let a = Tensor::<$backend>::new::<$t>(
                    &[
                        1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t, 7 as $t, 8 as $t, 9 as $t,
                        10 as $t, 11 as $t, 12 as $t, 13 as $t, 14 as $t, 15 as $t,
                    ],
                    vec![5, 3],
                )
                .unwrap();
                let b = Tensor::<$backend>::new::<$t>(
                    &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t],
                    vec![3, 2],
                )
                .unwrap();
                let mut dest = Tensor::<$backend>::new::<$t>(&[0 as $t; 10], vec![5, 2]).unwrap();

                a.mat_mul_inplace(&b, &mut dest, Activation::None).unwrap();

                assert_eq!(
                    dest.to_vec::<$t>().unwrap(),
                    vec![
                        22 as $t, 28 as $t, 49 as $t, 64 as $t, 76 as $t, 100 as $t, 103 as $t, 136 as $t,
                        130 as $t, 172 as $t
                    ]
                );
            }
        }
    };
}

mod metal {
    use super::*;

    test_mat_mul_for!(MetalBackend, f32);
    test_mat_mul_for!(MetalBackend, i32);
    test_mat_mul_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_mat_mul_for!(CPUBackend, f32);
    test_mat_mul_for!(CPUBackend, i32);
    test_mat_mul_for!(CPUBackend, i16);
}
