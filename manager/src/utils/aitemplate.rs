use std::{error::Error, fs, path::Path};

use console::style;
use shellexpand::tilde;

pub fn does_aitemplate_folder_exist() -> bool {
    let path = Path::new("AITemplate");
    path.exists()
}

pub fn is_aitemplate_installed() -> bool {
    let res = crate::utils::python::is_package_installed_venv("aitemplate");
    if res.is_ok() {
        return res.unwrap();
    } else {
        return false;
    }
}

pub fn reinstall_aitemplate_python_package() -> Result<(), Box<dyn Error>> {
    crate::utils::shell::spawn_command(
        "bash -c \"cd AITemplate/python && ../../venv/bin/python setup.py bdist_wheel && ../../venv/bin/pip install dist/*.whl --force-reinstall\"",
        "Install AITemplate",
    )?;
    Ok(())
}

pub fn wipe_aitemplate_cache_dir() -> Result<(), Box<dyn Error>> {
    let path = tilde("~/.aitemplate").to_string();
    let path = Path::new(&path);

    if !path.exists() {
        return Ok(());
    }
    println!(
        "{} Wiping AITemplate cache directory: {}",
        style("[INFO]").yellow(),
        path.display()
    );
    fs::remove_dir_all(path)?;
    Ok(())
}
