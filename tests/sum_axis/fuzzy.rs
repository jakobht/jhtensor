#[macro_export]
macro_rules! test_fuzzy {
    ($backend:ident, $t:ident) => {
        mod $t {
            use jhtensor::tensor::{CPUBackend, Tensor};
            use rand::{RngExt, SeedableRng};
            use rand_chacha::ChaCha8Rng;

            use super::*;

            fn run_test(m_range: std::ops::Range<usize>, n_range: std::ops::Range<usize>, axis: usize) {
                let mut rng = ChaCha8Rng::seed_from_u64(42);

                let m = rng.random_range(m_range);
                let n = rng.random_range(n_range);

                let size_a = m * n;
                let size_dest = if axis == 0 { n } else { m };

                // Fill vectors with dynamic data
                let a_data: Vec<$t> = (0..size_a).map(|_| rng.random_range(-10..10) as $t).collect();

                // Ground truth from CPU
                let cpu_a = Tensor::<CPUBackend>::new::<$t>(&a_data, vec![m, n]).unwrap();
                let cpu_result = cpu_a.sum_axis(axis).unwrap();

                // Backend under test
                let metal_a = Tensor::<$backend>::new::<$t>(&a_data, vec![m, n]).unwrap();
                let metal_result = metal_a.sum_axis(axis).unwrap();

                assert_eq!(cpu_result.shape(), vec![size_dest]);
                assert_eq!(metal_result.shape(), vec![size_dest]);

                assert_eq!(
                    metal_result.to_vec::<$t>().unwrap(),
                    cpu_result.to_vec::<$t>().unwrap(),
                );
            }

            #[test]
            fn run_small_test_axis_0() {
                run_test(8..16, 8..16, 0);
            }

            #[test]
            fn run_small_test_axis_1() {
                run_test(8..16, 8..16, 1);
            }

            #[test]
            fn run_medium_test_axis_0() {
                run_test(16..64, 16..64, 0);
            }

            #[test]
            fn run_medium_test_axis_1() {
                run_test(16..64, 16..64, 1);
            }

            #[test]
            fn run_large_test_axis_0() {
                run_test(64..128, 64..128, 0);
            }

            #[test]
            fn run_large_test_axis_1() {
                run_test(64..128, 64..128, 1);
            }
        }
    };
}
