use std::path::Path;

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
