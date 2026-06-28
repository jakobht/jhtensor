#[macro_export]
macro_rules! test_fuzzy {
    ($backend:ident, $t:ident) => {
        mod $t {
            use jhtensor::tensor::{CPUBackend, Tensor};
            use rand::{RngExt, SeedableRng};
            use rand_chacha::ChaCha8Rng;

            use super::*;

            fn run_test(m_range: std::ops::Range<usize>, n_range: std::ops::Range<usize>) {
                let mut rng = ChaCha8Rng::seed_from_u64(42);

                let m = rng.random_range(m_range);
                let n = rng.random_range(n_range);

                let size_mat = m * n;

                // Fill vectors with dynamic data
                let matrix_data: Vec<$t> = (0..size_mat).map(|_| rng.random_range(-10..10) as $t).collect();

                // Ground truth from CPU
                let cpu_matrix = Tensor::<CPUBackend>::new::<$t>(&matrix_data, [m, n]).unwrap();
                let cpu_transposed = cpu_matrix.transpose().unwrap();

                // Backend under test
                let backend_matrix = Tensor::<$backend>::new::<$t>(&matrix_data, [m, n]).unwrap();
                let backend_transposed = backend_matrix.transpose().unwrap();

                assert_eq!(
                    backend_transposed.to_vec::<$t>().unwrap(),
                    cpu_transposed.to_vec::<$t>().unwrap(),
                );
            }

            #[test]
            fn run_small_test() {
                run_test(8..16, 8..16);
            }

            #[test]
            fn run_medium_test() {
                run_test(16..64, 16..64);
            }

            #[test]
            fn run_large_test() {
                run_test(64..128, 64..128);
            }
        }
    };
}
