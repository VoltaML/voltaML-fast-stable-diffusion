mod apt;
mod debug;
mod env;
pub mod git;
pub mod install;
mod targets;
mod utils;

use console::style;
use dialoguer::{theme::ColorfulTheme, Select};
use utils::shell::run_command;

fn main() {
    // Check the Git repo update status
    if git::git::is_git_repo_up_to_date() {
        println!("{} {}", style("[Ok]").green(), "Already up-to-date.");
    } else {
        println!("{} {}", style("[!]").yellow(), "Update available.");
    }

    // Loop the menu
    loop {
        let items = vec!["Update", "Install", "Configure", "Developer Menu", "Exit"];
        let response_id = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();
        let response = items[response_id];

        match response {
            "Exit" => std::process::exit(0),
            "Developer Menu" => {
                debug_menu();
            }
            "Install" => install::install(),
            "Update" => git::git::update_git_repo(),
            "Configure" => configure(),
            "Start" => {
                start_api();
            }
            _ => println!("Error"),
        }
    }
}

fn configure() {
    loop {
        let items = vec!["Exit", "Huggingface Token", "Logging Level"];
        let response = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();

        match response {
            0 => break,
            1 => env::change_huggingface_token(),
            2 => env::change_logging_level(),
            _ => println!("Error"),
        }
    }
}

fn debug_menu() {
    loop {
        let items = vec![
            "Exit",
            "Check NVCC",
            "Check Python",
            "Detect OS",
            "Clone repo",
            "Is Root",
            "Apt Update",
            "Apt Upgrade",
            "Does venv exist",
            "Installed Python packages",
            "Create venv",
            "Check CUDA",
            "Check NVIDIA CUDA repo key",
        ];
        let response = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();

        match response {
            0 => break,
            1 => {
                if utils::nvidia::is_nvcc_installed() {
                    let version = utils::nvidia::nvcc_version();
                    if version.is_ok() {
                        println!("{} {}", style("[OK]").green(), version.unwrap());
                    } else {
                        println!(
                            "{} {}",
                            style("[ERROR]").red(),
                            "Could not parse nvcc version output"
                        );
                    }
                } else {
                    println!("{} {}", style("[ERROR]").red(), "nvcc not available");
                }
            }
            2 => {
                if utils::python::is_python_installed() {
                    let version = utils::python::python_version();
                    if version.is_ok() {
                        println!("{} {}", style("[OK]").green(), version.unwrap());
                    } else {
                        println!(
                            "{} {}",
                            style("[ERROR]").red(),
                            "Could not parse Python version output"
                        );
                    }
                    utils::python::is_pip_installed();
                    utils::python::is_virtualenv_installed();
                } else {
                    println!("{} {}", style("[ERROR]").red(), "Python not available");
                }
            }
            3 => {
                let os = targets::detect_target();
                println!("Targer OS: {}", os.to_string());
            }
            4 => {
                println!("Cloning repo...");
                let res = git::clone::clone_repo(
                    "https://github.com/voltaML/voltaML-fast-stable-diffusion",
                    "tmp",
                    "experimental",
                );
                if res.is_ok() {
                    println!("{} {}", style("[OK]").green(), "Cloned repo");
                } else {
                    println!("{} {}", style("[ERROR]").red(), "Clone failed");
                }
            }
            5 => {
                if utils::is_root() {
                    println!("{} {}", style("[INFO]").green(), "Running as root");
                } else {
                    println!("{} {}", style("[INFO]").green(), "Running as user");
                }
            }
            6 => apt::update(),
            7 => apt::upgrade(),
            8 => {
                if utils::python::does_venv_exists() {
                    println!("{} {}", style("[INFO]").green(), "Venv exists");
                } else {
                    println!("{} {}", style("[ERROR]").red(), "No venv found");
                }
            }
            9 => {
                debug::python_packages().print_tty(false).unwrap();
            }
            10 => {
                utils::python::create_venv();
            }
            11 => {
                if utils::nvidia::is_cuda_installed() {
                    println!("{} {}", style("[OK]").green(), "CUDA installed");
                } else {
                    println!("{} {}", style("[ERROR]").red(), "CUDA not installed");
                }
            }
            12 => {
                if utils::nvidia::is_nvidia_repo_added() {
                    println!(
                        "{} {}",
                        style("[OK]").green(),
                        "NVIDIA CUDA repo key present"
                    );
                } else {
                    println!(
                        "{} {}",
                        style("[ERROR]").red(),
                        "NVIDIA CUDA repo key not present"
                    );
                }
            }
            _ => println!("Error"),
        }
    }
}

fn start_api() {
    run_command("venv/bin/python main.py", "Run the API").unwrap();
}
