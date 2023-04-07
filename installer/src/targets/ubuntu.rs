use dialoguer::{theme::ColorfulTheme, Select};
use std::process::Command;

pub fn install(branch: &str, wsl: bool) {
    println!("Selected WSL installation method");

    let items = vec!["NVIDIA", "AMD"];
    let gpu_type = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    // Update and upgrade the system
    Command::new("sudo")
        .arg("apt")
        .arg("update")
        .arg("&&")
        .arg("sudo")
        .arg("apt")
        .arg("upgrade")
        .arg("-y")
        .output()
        .expect("Failed to update the system");

    // Clone the repo
    Command::new("git")
        .arg("clone")
        .arg("https://github.com/VoltaML/voltaML-fast-stable-diffusion.git")
        .arg("--branch")
        .arg(branch)
        .output()
        .expect("Failed to clone the repo");

    // Install Python, ROCmInfo
    Command::new("sudo")
        .arg("apt")
        .arg("install")
        .arg("-y")
        .arg("python3.10")
        .arg("python3.10-venv")
        .arg("build-essential")
        .arg("python3-pip")
        .arg("rocminfo")
        .output()
        .expect("Failed to install Python dependencies");

    match gpu_type {
        0 => nvidia(wsl),
        1 => amd(),
        _ => println!("Error"),
    }
}

fn nvidia(_wsl: bool) {
    //
}

fn amd() {}
