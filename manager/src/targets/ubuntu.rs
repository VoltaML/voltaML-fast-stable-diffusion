use std::{error::Error, path::Path};

use crate::utils::shell::spawn_command;
use console::style;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install(wsl: bool, experimental: bool) {
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
    let res = crate::git::clone::clone_repo(
        "https://github.com/voltaML/voltaML-fast-stable-diffusion",
        "tmp",
        if experimental { "experimental" } else { "main" },
    );
    if res.is_err() {
        println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
        return;
    }

    // Move everything into current dir
    let res = crate::utils::shell::move_dir_recursive(&Path::new("tmp"), &Path::new("."));
    if res.is_err() {
        println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
        return;
    }

    // Install build-essential
    let res = crate::apt::install("build-essential");
    if res.is_err() {
        println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
        return;
    }

    // Check Python
    if !crate::utils::python::is_python_installed() {
        let res = crate::apt::install("python3.10");
        if res.is_err() {
            println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
            return;
        }
    }

    // Check pip
    if !crate::utils::python::is_pip_installed() {
        let res = crate::apt::install("python3-pip");
        if res.is_err() {
            println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
            return;
        }
    }

    // Check virtualenv
    let virtualenv_installed = crate::utils::python::is_virtualenv_installed();
    if !virtualenv_installed {
        println!(
            "{} virtualenv not installed, installing...",
            style("[INFO]").green()
        );
        let res = crate::apt::install("python3-venv");
        if res.is_err() {
            println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
            return;
        }
    } else {
        println!(
            "{} virtualenv installed, skipping...",
            style("[INFO]").green()
        );
    }

    // Install GPU Inference dependencies
    match gpu_type {
        0 => {
            let res = nvidia(wsl);
            if res.is_err() {
                println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
                return;
            }
        }
        1 => {
            println!("{} {}", style("[ERROR]").red(), "AMD not supported yet");
            return;
        }
        _ => println!("Error"),
    }

    // Check NVCC
    crate::utils::nvidia::is_nvcc_installed();

    // Insert the HUGGINGFACE_TOKEN
    crate::environ::change_huggingface_token();

    // Create the virtual environment
    let res = crate::utils::python::create_venv();
    if res.is_err() {
        println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
        return;
    }

    // Install wheel
    let res = crate::utils::python::pip_install_venv("wheel");
    if res.is_err() {
        println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
        return;
    }

    // Install AITemplate
    spawn_command(
        "bash -c \"git clone --recursive https://github.com/facebookincubator/AITemplate && cd AITemplate/python && ../../venv/bin/python setup.py bdist_wheel && ../../venv/bin/pip install dist/*.whl --force-reinstall\"",
        "Install AITemplate",
    )
    .unwrap();

    // Finish
    println!(
        "{} {}",
        style("[OK]").green(),
        "Installation complete, please select 'Start' to start the application'"
    );
}

fn nvidia(_wsl: bool) -> Result<(), Box<dyn Error>> {
    // Check if nvidia repository is added to apt
    if !crate::utils::nvidia::is_nvidia_repo_added() {
        crate::utils::nvidia::add_nvidia_repo()?;
    }

    // Install CUDA if not installed
    let cuda_installed = crate::utils::nvidia::is_cuda_installed();
    if cuda_installed.is_ok() {
        if !cuda_installed.unwrap() {
            crate::apt::install("cuda")?;
        }
    } else {
        println!(
            "{} {}",
            style("[ERROR]").red(),
            "Could not check if CUDA is installed, exiting"
        );
        return Err(cuda_installed.err().unwrap());
    }

    let export_check = crate::environ::check_cuda_exports();
    if export_check.is_ok() {
        if !export_check.unwrap() {
            println!(
                "{} {}",
                style("[OK]").green(),
                "CUDA exports are not present, adding them"
            );
            crate::environ::insert_cuda_exports()?;
            println!(
                "{} {}",
                style("[OK]").green(),
                "CUDA exports added to ~/.bashrc"
            );
        }
        println!(
            "{} {}",
            style("[OK]").green(),
            "CUDA exports are already present, continuing"
        );
        Ok(())
    } else {
        println!(
            "{} {}",
            style("[ERROR]").red(),
            "Could not check if CUDA exports are present, exiting"
        );
        Err(export_check.err().unwrap())
    }
}
