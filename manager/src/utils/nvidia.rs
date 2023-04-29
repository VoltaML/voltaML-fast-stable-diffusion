use std::error::Error;

use crate::apt::update;

use super::shell::run_command;

pub fn is_nvcc_installed() -> bool {
    run_command("nvcc --version", "Is NVCC available").is_ok()
}

pub fn nvcc_version() -> Result<String, Box<dyn Error>> {
    let output = run_command("nvcc --version", "Check NVCC version")?;
    let version = output
        .split("\n")
        .nth(3)
        .ok_or("Could not parse nvcc version output")?
        .split_whitespace()
        .nth(4)
        .ok_or("Could not parse nvcc version output")?
        .replace(",", "")
        .to_string();
    Ok(version)
}

pub fn is_cuda_installed() -> Result<bool, Box<dyn Error>> {
    let res = run_command("dpkg -l", "List dpgk packages")?;
    for line in res.lines() {
        let package_name = line.split_whitespace().nth(1);
        if package_name.unwrap_or("") == "cuda" {
            return Ok(true);
        }
    }
    return Ok(false);
}

pub fn is_nvidia_repo_added() -> bool {
    let res = run_command(
        "cat /etc/apt/preferences.d/cuda-repository-pin-600",
        "Check NVIDIA repo key",
    );
    if res.is_err() {
        return false;
    } else {
        return true;
    }
}

pub fn add_nvidia_repo() -> Result<(), Box<dyn Error>> {
    // Install Wget
    crate::apt::install("wget")?;

    // Get the NVIDIA repo key
    run_command(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin",
        "Get NVIDIA repo key",
    )?;

    // Move the NVIDIA repo key
    run_command(
        "sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600",
        "Move NVIDIA repo key",
    )?;

    // Add the NVIDIA repo key to apt
    run_command(
        "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub",
        "Insert NVIDIA repo key into apt",
    )?;

    // Add the NVIDIA repo to apt
    run_command(
        "sudo add-apt-repository -y \"deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /\"",
        "Add NVIDIA repo to apt",
    )?;

    // Update
    update();

    Ok(())
}
