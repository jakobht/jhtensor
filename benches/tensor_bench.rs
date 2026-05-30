use criterion::{Criterion, criterion_group, criterion_main};
use jhtensor::tensor::{CPUBackend, MetalBackend, Tensor};
use std::hint::black_box;

fn bench_tensor_add(c: &mut Criterion) {
    let size = 1_000_000_000;

    // Create vector with random values
    let result: Vec<f32> = (0..size).map(|_| 0.0).collect();
    let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
    let shape = vec![size];

    let cpu_a = Tensor::<CPUBackend>::new(&data, shape.clone()).unwrap();
    let cpu_b = Tensor::<CPUBackend>::new(&data, shape.clone()).unwrap();
    let mut cpu_dest = Tensor::<CPUBackend>::new(&result, shape.clone()).unwrap();

    let metal_a = Tensor::<MetalBackend>::new(&data, shape.clone()).unwrap();
    let metal_b = Tensor::<MetalBackend>::new(&data, shape.clone()).unwrap();
    let mut metal_dest = Tensor::<MetalBackend>::new(&result, shape.clone()).unwrap();

    let mut group = c.benchmark_group("Add Tensors (1B elements)");

    group.bench_function("CPU Backend Add Inplace", |b| {
        b.iter(|| {
            cpu_a.add_inplace(black_box(&cpu_b), black_box(&mut cpu_dest)).unwrap();
        })
    });

    group.bench_function("Metal Backend Add Inplace", |b| {
        b.iter(|| {
            metal_a
                .add_inplace(black_box(&metal_b), black_box(&mut metal_dest))
                .unwrap();
        })
    });

    group.bench_function("CPU Backend Add", |b| {
        b.iter(|| {
            cpu_a.add(black_box(&cpu_b)).unwrap();
        })
    });

    group.bench_function("Metal Backend Add", |b| {
        b.iter(|| {
            metal_a.add(black_box(&metal_b)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_tensor_add);
criterion_main!(benches);
