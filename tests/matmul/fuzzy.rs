#[macro_export]
macro_rules! test_fuzzy {
    ($backend:ident, $t:ident, $activation:expr) => {
        mod $t {
            use jhtensor::tensor::{CPUBackend, Tensor};
            use rand::{RngExt, SeedableRng};
            use rand_chacha::ChaCha8Rng;

            use super::*;

            fn run_test(
                m_range: std::ops::Range<usize>,
                n_range: std::ops::Range<usize>,
                k_range: std::ops::Range<usize>,
            ) {
                let mut rng = ChaCha8Rng::seed_from_u64(42);

                let m = rng.random_range(m_range);
                let n = rng.random_range(n_range);
                let k = rng.random_range(k_range);

                let size_a = m * k;
                let size_b = k * n;
                let size_dest = m * n;

                // Fill vectors with dynamic data
                let a_data: Vec<$t> = (0..size_a).map(|_| rng.random_range(-10..10) as $t).collect();
                let b_data: Vec<$t> = (0..size_b).map(|_| rng.random_range(-10..10) as $t).collect();

                // Ground truth from CPU
                let cpu_a = Tensor::<CPUBackend>::new::<$t>(&a_data, vec![m, k]).unwrap();
                let cpu_b = Tensor::<CPUBackend>::new::<$t>(&b_data, vec![k, n]).unwrap();
                let mut cpu_dest = Tensor::<CPUBackend>::new::<$t>(&vec![0 as $t; size_dest], vec![m, n]).unwrap();
                cpu_a.mat_mul_inplace(&cpu_b, &mut cpu_dest, $activation).unwrap();

                // Backend under test
                let metal_a = Tensor::<$backend>::new::<$t>(&a_data, vec![m, k]).unwrap();
                let metal_b = Tensor::<$backend>::new::<$t>(&b_data, vec![k, n]).unwrap();
                let mut backend_dest = Tensor::<$backend>::new::<$t>(&vec![0 as $t; size_dest], vec![m, n]).unwrap();
                metal_a
                    .mat_mul_inplace(&metal_b, &mut backend_dest, $activation)
                    .unwrap();

                assert_eq!(
                    backend_dest.to_vec::<$t>().unwrap(),
                    cpu_dest.to_vec::<$t>().unwrap(),
                );
            }

            #[test]
            fn run_small_test() {
                run_test(8..16, 8..16, 8..16);
            }

            #[test]
            fn run_medium_test() {
                run_test(16..64, 16..64, 16..64);
            }

            #[test]
            fn run_large_test() {
                run_test(64..128, 64..128, 64..128);
            }
        }
    };
}
