use jhtensor::tensor::{CPUBackend, MetalBackend, Shape, Tensor};

macro_rules! test_broadcast_for {
    ($backend:ident, $t:ident) => {
        mod $t {
            use super::*;

            #[test]
            fn test_broadcast_0() {
                let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t], [3]).unwrap();

                let result = a.broadcast([4, 3], 0).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                assert_eq!(result.shape(), Shape::new([4, 3]));
                assert_eq!(result_vec, vec![1 as $t, 2 as $t, 3 as $t].repeat(4));
            }

            #[test]
            fn test_broadcast_1() {
                let a = Tensor::<$backend>::new::<$t>(&[1 as $t, 2 as $t, 3 as $t], [3]).unwrap();

                let result = a.broadcast([3, 4], 1).unwrap();
                let result_vec = result.to_vec::<$t>().unwrap();

                let expected_vec: Vec<$t> = [1 as $t, 2 as $t, 3 as $t]
                    .iter()
                    .flat_map(|&val| std::iter::repeat(val).take(4))
                    .collect::<Vec<$t>>();

                assert_eq!(result.shape(), Shape::new([3, 4]));
                assert_eq!(result_vec, expected_vec);
            }
        }
    };
}

mod metal {
    use super::*;

    test_broadcast_for!(MetalBackend, f32);
    test_broadcast_for!(MetalBackend, i32);
    test_broadcast_for!(MetalBackend, i16);
}

mod cpu {
    use super::*;

    test_broadcast_for!(CPUBackend, f32);
    test_broadcast_for!(CPUBackend, i32);
    test_broadcast_for!(CPUBackend, i16);
}
