use criterion::{Criterion, criterion_group, criterion_main};
use jhtensor::tensor::{Activation, CPUBackend, MetalBackend, Tensor};
use std::hint::black_box;

fn bench_tensor_add(c: &mut Criterion) {
    let size = 100_000_000;

    // Create vector with random values
    let result: Vec<f32> = (0..size).map(|_| 0.0).collect();
    let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
    let shape = [size];

    let cpu_a = Tensor::<CPUBackend>::new(&data, shape).unwrap();
    let cpu_b = Tensor::<CPUBackend>::new(&data, shape).unwrap();
    let mut cpu_dest = Tensor::<CPUBackend>::new(&result, shape).unwrap();

    let metal_a = Tensor::<MetalBackend>::new(&data, shape).unwrap();
    let metal_b = Tensor::<MetalBackend>::new(&data, shape).unwrap();
    let mut metal_dest = Tensor::<MetalBackend>::new(&result, shape).unwrap();

    let mut group = c.benchmark_group("Add Tensors (100M elements)");

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

fn bench_tensor_mat_mul(c: &mut Criterion) {
    let m = 300;
    let n = 300;
    let k = 300;

    let a_data: Vec<f32> = (0..m * k).map(|_| rand::random::<f32>()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|_| rand::random::<f32>()).collect();
    let result: Vec<f32> = (0..m * n).map(|_| 0.0).collect();

    let cpu_a = Tensor::<CPUBackend>::new(&a_data, [m, k]).unwrap();
    let cpu_b = Tensor::<CPUBackend>::new(&b_data, [k, n]).unwrap();
    let mut cpu_dest = Tensor::<CPUBackend>::new(&result, [m, n]).unwrap();

    let metal_a = Tensor::<MetalBackend>::new(&a_data, [m, k]).unwrap();
    let metal_b = Tensor::<MetalBackend>::new(&b_data, [k, n]).unwrap();
    let mut metal_dest = Tensor::<MetalBackend>::new(&result, [m, n]).unwrap();

    let mut group = c.benchmark_group("Mat Mul Tensors (300 x 300)");

    group.bench_function("CPU Backend Mat Mul Inplace", |b| {
        b.iter(|| {
            cpu_a
                .mat_mul_inplace(black_box(&cpu_b), black_box(&mut cpu_dest), Activation::None)
                .unwrap();
        })
    });

    group.bench_function("Metal Backend Mat Mul Inplace", |b| {
        b.iter(|| {
            metal_a
                .mat_mul_inplace(black_box(&metal_b), black_box(&mut metal_dest), Activation::None)
                .unwrap();
        })
    });
}

criterion_group!(benches, bench_tensor_mat_mul, bench_tensor_add);
criterion_main!(benches);
