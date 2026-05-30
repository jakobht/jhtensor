use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};

fn bench_tensor_add(c: &mut Criterion) {
    let size = 1_000_000_000;

    // Create vector with random values
    let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
    let shape = vec![size];

    let cpu_a = Tensor::<CPUBackend>::new(&data, shape.clone()).unwrap();
    let cpu_b = Tensor::<CPUBackend>::new(&data, shape.clone()).unwrap();

    let metal_a = Tensor::<MetalBackend>::new(&data, shape.clone()).unwrap();
    let metal_b = Tensor::<MetalBackend>::new(&data, shape.clone()).unwrap();

    let mut group = c.benchmark_group("Add Tensors (1B elements)");

    group.bench_function("CPU Backend", |b| {
        b.iter(|| {
            let _result = cpu_a.add(black_box(&cpu_b)).unwrap();
        })
    });

    group.bench_function("Metal Backend", |b| {
        b.iter(|| {
            let _result = metal_a.add(black_box(&metal_b)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_tensor_add);
criterion_main!(benches);
