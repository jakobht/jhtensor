use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

macro_rules! test_sum_axis_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn test_sum_axis_0() {
                let a = Tensor::<$backend>::new::<$t>(
                    &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t],
                    vec![2, 3],
                )
                .unwrap();

                let result = a.sum_axis(0).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result.shape(), vec![3]);
                assert_eq!(result_vec, vec![5 as $t, 7 as $t, 9 as $t]);
            }

            #[test]
            fn test_sum_axis_1() {
                let a = Tensor::<$backend>::new::<$t>(
                    &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t],
                    vec![2, 3],
                )
                .unwrap();

                let result = a.sum_axis(1).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result.shape(), vec![2]);
                assert_eq!(result_vec, vec![6 as $t, 15 as $t]);
            }
        }
    };
}

mod metal {
    use super::*;

    test_sum_axis_for!(MetalBackend, f32);
    test_sum_axis_for!(MetalBackend, i32);
    test_sum_axis_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_sum_axis_for!(CPUBackend, f32);
    test_sum_axis_for!(CPUBackend, i32);
    test_sum_axis_for!(CPUBackend, i16);
}
