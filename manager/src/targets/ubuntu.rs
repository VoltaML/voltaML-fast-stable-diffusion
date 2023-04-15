use crate::utils::shell::run_command;
use console::style;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install(wsl: bool) {
    println!("Selected WSL installation method");

    let gpu_types = vec!["NVIDIA", "AMD"];
    let gpu_type = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&gpu_types)
        .interact()
        .unwrap();

    // Update and upgrade the system
    crate::apt::update();
    crate::apt::upgrade();

    // Clone the repo
    let branches = vec!["Main", "Experimental"];
    let branch = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&branches)
        .interact()
        .unwrap();
    crate::git::clone::clone_repo(
        "https://github.com/voltaML/voltaML-fast-stable-diffusion",
        ".",
        if branch == 0 { "main" } else { "experimental" },
    )
    .unwrap();

    // Install Python, ROCmInfo
    run_command(
        "sudo apt install -y python3.10 python3.10-venv build-essential python3-pip",
        "Install Python dependencies",
    )
    .unwrap();

    // Check Python
    crate::utils::python::is_python_installed();
    crate::utils::python::is_pip_installed();
    crate::utils::python::is_virtualenv_installed();

    // Install GPU Inference dependencies
    match gpu_type {
        0 => nvidia(wsl),
        1 => {
            println!("{} {}", style("[ERROR]").red(), "AMD not supported yet");
            return;
        }
        _ => println!("Error"),
    }

    // Check NVCC
    crate::utils::nvidia::is_nvcc_installed();

    // Insert the HUGGINGFACE_TOKEN
    crate::env::change_huggingface_token();

    // Create the virtual environment
    crate::utils::python::create_venv();

    // Install wheel
    crate::utils::python::pip_install("wheel");

    // Install AITemplate
    run_command(
        "git clone --recursive https://github.com/facebookincubator/AITemplate && cd AITemplate/python && ../../venv/bin/python setup.py bdist_wheel && ../../venv/bin/pip install dist/*.whl --force-reinstall",
        "Install AITemplate",
    )
    .unwrap();

    // Finish
    println!(
        "{} {}",
        style("[OK]").green(),
        "Installation complete, please run 'bash start.sh' to start the application'"
    );
}

fn nvidia(_wsl: bool) {
    // Check if nvidia repository is added to apt
    if !crate::utils::nvidia::is_nvidia_repo_added() {
        crate::utils::nvidia::add_nvidia_repo();
    }

    // Install CUDA if not installed
    if !crate::utils::nvidia::is_cuda_installed() {
        crate::apt::install("cuda");
    }
}
