mod tensor;

#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

fn main() {
    println!("Running on Metal backend, f32:");

    let tensor: tensor::Tensor<tensor::MetalBackend> = tensor::Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let tensor2: tensor::Tensor<tensor::MetalBackend> = tensor::Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let result = tensor.add(&tensor2);

    for i in 0..result.to_vec::<f32>().len() {
        println!(
            "{} + {} = {} (expected {})",
            tensor.to_vec::<f32>()[i],
            tensor2.to_vec::<f32>()[i],
            result.to_vec::<f32>()[i],
            tensor.to_vec::<f32>()[i] + tensor2.to_vec::<f32>()[i]
        );
    }

    println!("Running on Metal backend, i32:");
    let tensor: tensor::Tensor<tensor::MetalBackend> = tensor::Tensor::new(&[1, 2, 3, 4, 5], vec![5]);
    let tensor2: tensor::Tensor<tensor::MetalBackend> = tensor::Tensor::new(&[1, 2, 3, 4, 5], vec![5]);
    let result = tensor.add(&tensor2);

    for i in 0..result.to_vec::<i32>().len() {
        println!(
            "{} + {} = {} (expected {})",
            tensor.to_vec::<i32>()[i],
            tensor2.to_vec::<i32>()[i],
            result.to_vec::<i32>()[i],
            tensor.to_vec::<i32>()[i] + tensor2.to_vec::<i32>()[i]
        );
    }

    println!("Running on CPU backend, i16:");
    let tensor: tensor::Tensor<tensor::CPUBackend> = tensor::Tensor::new(&[1i16, 2, 3, 4, 5], vec![5]);
    let tensor2: tensor::Tensor<tensor::CPUBackend> = tensor::Tensor::new(&[1i16, 2, 3, 4, 5], vec![5]);
    let result = tensor.add(&tensor2);

    for i in 0..result.to_vec::<i16>().len() {
        println!(
            "{} + {} = {} (expected {})",
            tensor.to_vec::<i16>()[i],
            tensor2.to_vec::<i16>()[i],
            result.to_vec::<i16>()[i],
            tensor.to_vec::<i16>()[i] + tensor2.to_vec::<i16>()[i]
        );
    }
}
