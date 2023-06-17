use std::{error::Error, path::Path};

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
