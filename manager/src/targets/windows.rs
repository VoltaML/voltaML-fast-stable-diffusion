use std::path::Path;

use console::style;

pub fn install(experimental: bool) {
    println!("Selected Windows installation method");

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

    // Check Python
    if !crate::utils::python::is_python_installed() {
        println!("{} Python not installed (or added to PATH), please install version 3.10 (tested) or later", style("[ERROR]").red());
    }

    // Check virtualenv
    let virtualenv_installed = crate::utils::python::is_virtualenv_installed();
    if !virtualenv_installed {
        println!(
            "{} virtualenv not installed, installing...",
            style("[INFO]").green()
        );
        let res = crate::utils::python::pip_install("virtualenv");
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

    // Finish
    println!(
        "{} {}",
        style("[OK]").green(),
        "Installation complete, please select 'Start' to start the application'"
    );
}
