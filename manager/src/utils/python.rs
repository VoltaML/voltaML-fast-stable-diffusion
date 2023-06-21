use std::{error::Error, path::Path};

use crate::targets::{detect_target, Target};

use super::shell::{run_command, spawn_command};

pub fn python_executable() -> String {
    if detect_target() == Target::Windows {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

pub fn get_venv_pip() -> String {
    if detect_target() == Target::Windows {
        "venv/Scripts/pip".to_string()
    } else {
        "venv/bin/pip".to_string()
    }
}

pub fn get_venv_python() -> String {
    if detect_target() == Target::Windows {
        "venv/Scripts/python".to_string()
    } else {
        "venv/bin/python".to_string()
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
    is_package_installed("virtualenv").unwrap_or(false)
}

pub fn does_venv_exists() -> bool {
    let path = Path::new("venv");
    path.exists()
}

pub struct PythonPackage {
    pub name: String,
    pub version: String,
}

pub fn installed_packages_venv() -> Result<Vec<PythonPackage>, Box<dyn Error>> {
    let output = run_command(&format!("{} list", get_venv_pip()), "Installed packages")?;
    let mut packages = Vec::new();
    for line in output.lines().skip(2) {
        let package = line.split_whitespace().collect::<Vec<&str>>();
        let name = package[0].to_string();
        let version = package[1].to_string();
        packages.push(PythonPackage { name, version });
    }
    Ok(packages)
}

pub fn installed_packages() -> Result<Vec<PythonPackage>, Box<dyn Error>> {
    let output = run_command(
        &format!("{} -m pip list", python_executable()),
        "Installed packages",
    )?;
    let mut packages = Vec::new();
    for line in output.lines().skip(2) {
        let package = line.split_whitespace().collect::<Vec<&str>>();
        let name = package[0].to_string();
        let version = package[1].to_string();
        packages.push(PythonPackage { name, version });
    }
    Ok(packages)
}

pub fn create_venv() -> Result<(), Box<dyn Error>> {
    run_command(
        &format!("{} -m virtualenv venv", python_executable()),
        "Create virtualenv",
    )?;
    Ok(())
}

pub fn pip_install_venv(package: &str) -> Result<(), Box<dyn Error>> {
    spawn_command(
        &format!("{} install {}", get_venv_pip(), package),
        &format!("Install package {}", package),
    )?;
    Ok(())
}

pub fn pip_install(package: &str) -> Result<(), Box<dyn Error>> {
    spawn_command(
        &format!("{} -m pip install {}", python_executable(), package),
        &format!("Install package {}", package),
    )?;
    Ok(())
}

pub fn is_package_installed_venv(package: &str) -> Result<bool, Box<dyn Error>> {
    let packages = installed_packages_venv()?;
    for p in packages {
        if p.name == package {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn is_package_installed(package: &str) -> Result<bool, Box<dyn Error>> {
    let packages = installed_packages()?;
    for p in packages {
        if p.name == package {
            return Ok(true);
        }
    }
    Ok(false)
}
