use crate::utils::shell::run_command;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install(branch: &str, wsl: bool) {
    println!("Selected WSL installation method");

    let items = vec!["NVIDIA", "AMD"];
    let gpu_type = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    // Update and upgrade the system
    run_command(
        "sudo apt update && sudo apt upgrade -y",
        "Update and upgrade the system",
    );

    // Clone the repo
    run_command(
        format!(
            "git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git --branch {}",
            branch
        )
        .as_str(),
        "Clone the repo",
    );

    // Install Python, ROCmInfo
    run_command(
        "sudo apt install -y python3.10 python3.10-venv build-essential python3-pip rocminfo",
        "Install Python dependencies",
    );

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
