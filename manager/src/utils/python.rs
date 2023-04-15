use std::{error::Error, path::Path};

use crate::targets::{detect_target, Target};

use super::shell::run_command;

pub fn python_executable() -> String {
    if detect_target() == Target::Windows {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

pub fn is_python_installed() -> bool {
    let executable = python_executable();
    run_command(&format!("{} --version", executable), "Is Python available").is_ok()
}

pub fn python_version() -> Result<String, Box<dyn Error>> {
    let executable = python_executable();
    let output = run_command(&format!("{} --version", executable), "Check Python version")?;
    let version = output
        .split(" ")
        .nth(1)
        .ok_or("Could not parse Python version output")?
        .replace("\n", "")
        .to_string();
    Ok(version)
}

pub fn is_pip_installed() -> bool {
    run_command("pip --version", "Is pip available").is_ok()
}

pub fn is_virtualenv_installed() -> bool {
    run_command("virtualenv --version", "Is virtualenv available").is_ok()
}

pub fn does_venv_exists() -> bool {
    let path = Path::new("venv");
    path.exists()
}

pub struct PythonPackage {
    pub name: String,
    pub version: String,
}

pub fn installed_packages() -> Result<Vec<PythonPackage>, Box<dyn Error>> {
    let output = run_command("venv/bin/pip list", "Installed packages")?;
    let mut packages = Vec::new();
    for line in output.lines().skip(2) {
        let package = line.split_whitespace().collect::<Vec<&str>>();
        let name = package[0].to_string();
        let version = package[1].to_string();
        packages.push(PythonPackage { name, version });
    }
    Ok(packages)
}

pub fn create_venv() {
    run_command("virtualenv venv", "Create virtualenv").unwrap();
}

pub fn pip_install(package: &str) {
    run_command(
        &format!("venv/bin/pip install {}", package),
        &format!("Install package {}", package),
    )
    .unwrap();
}
