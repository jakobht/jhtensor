use std::process::Command;

fn main() {
    // only run if the shader file changes
    println!("cargo:rerun-if-changed=src/shaders/add_func.metal");

    // 2. Compile .metal to binary .air file
    let compile_status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "src/shaders/add_func.metal",
            "-o",
            "add_func.air",
        ])
        .status()
        .expect("Failed to execute xcrun metal. Do you have Xcode command line tools installed?");

    assert!(compile_status.success(), "Failed to compile add_func.metal!");

    // 3. Link the .air file to a .metallib bundle
    let link_status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "add_func.air", "-o", "add_func.metallib"])
        .status()
        .expect("Failed to execute xcrun metallib.");

    assert!(link_status.success(), "Failed to link into add_func.metallib!");

    // 4. Clean up the intermediate .air file
    std::fs::remove_file("add_func.air").ok();
}
