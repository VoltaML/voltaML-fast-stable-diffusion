mod apt;
mod debug;
mod environ;
pub mod git;
pub mod install;
mod targets;
mod utils;

use console::style;
use dialoguer::{theme::ColorfulTheme, Select};
use std::{env, error::Error};
use utils::shell::spawn_command;

fn main() {
    // Check the Git repo update status
    let uptodate = git::git::is_git_repo_up_to_date();
    if uptodate.is_ok() {
        if uptodate.unwrap() {
            println!("{} {}", style("[Ok]").green(), "Repository up-to-date.");
        } else {
            println!(
                "{} {}",
                style("[!]").yellow(),
                "Repository ready to update."
            );
        }
    } else {
        println!(
            "{} {}",
            style("[!]").red(),
            "Could not check for repository updates."
        );
    }

    // Loop the menu
    loop {
        let items = vec![
            "Start",
            "Update",
            "Install",
            "Configure",
            "Developer Menu",
            "Exit",
        ];
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
            "Update" => {
                let res = git::git::update_git_repo();
                if res.is_ok() {
                    println!("{}", res.unwrap());
                } else {
                    println!("{} {}", style("[!]").red(), res.err().unwrap());
                }
            }
            "Configure" => configure(),
            "Start" => {
                let res = start_api();
                if res.is_err() {
                    println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
                }
            }
            _ => println!("Error"),
        }
    }
}

fn configure() {
    loop {
        let items = vec!["Back", "Huggingface Token", "Logging Level"];
        let response = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();

        match response {
            0 => break,
            1 => environ::change_huggingface_token(),
            2 => environ::change_logging_level(),
            _ => println!("Error"),
        }
    }
}

fn debug_menu() {
    loop {
        let items = vec![
            "Back",
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
            "Is virtualenv installed",
            "Install virtualenv",
            "Show $PATH",
            "Check CUDA exports in .bashrc",
            "Insert CUDA exports to .bashrc",
        ];
        let response_id = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();

        let response = items[response_id];

        match response {
            "Back" => break,
            "Check NVCC" => {
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
            "Check Python" => {
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
            "Detect OS" => {
                let os = targets::detect_target();
                println!("{} Targer OS: {}", style("[OK]").green(), os.to_string());
            }
            "Clone repo" => {
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
            "Is Root" => {
                if utils::is_root() {
                    println!("{} {}", style("[INFO]").green(), "Running as root");
                } else {
                    println!("{} {}", style("[INFO]").green(), "Running as user");
                }
            }
            "Apt Update" => apt::update(),
            "Apt Upgrade" => apt::upgrade(),
            "Does venv exist" => {
                if utils::python::does_venv_exists() {
                    println!("{} {}", style("[INFO]").green(), "Venv exists");
                } else {
                    println!("{} {}", style("[ERROR]").red(), "No venv found");
                }
            }
            "Installed Python packages" => {
                let res = debug::python_packages();
                if res.is_ok() {
                    res.unwrap().print_tty(false).unwrap();
                } else {
                    println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
                }
            }
            "Create venv" => {
                let res = utils::python::create_venv();
                if res.is_ok() {
                    println!("{} {}", style("[OK]").green(), "Venv created");
                } else {
                    println!("{} {}", style("[ERROR]").red(), res.err().unwrap());
                }
            }
            "Check CUDA" => {
                let res = utils::nvidia::is_cuda_installed();
                if res.is_ok() {
                    if res.unwrap() {
                        println!("{} {}", style("[OK]").green(), "CUDA installed");
                    } else {
                        println!("{} {}", style("[ERROR]").red(), "CUDA not installed");
                    }
                } else {
                    println!(
                        "{} {}: {}",
                        style("[ERROR]").red(),
                        "CUDA not installed",
                        res.err().unwrap()
                    );
                }
            }
            "Check NVIDIA CUDA repo key" => {
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
            "Is virtualenv installed" => {
                if utils::python::is_virtualenv_installed() {
                    println!("{} {}", style("[OK]").green(), "virtualenv installed");
                } else {
                    println!("{} {}", style("[ERROR]").red(), "virtualenv not installed");
                }
            }
            "Install virtualenv" => {
                let res = utils::python::install_virtualenv();
                if res.is_ok() {
                    println!("{} {}", style("[OK]").green(), "virtualenv installed");
                } else {
                    println!(
                        "{} {}: {}",
                        style("[ERROR]").red(),
                        "virtualenv failed to install",
                        res.err().unwrap()
                    );
                }
            }
            "Show $PATH" => {
                println!("Path: {}", env::var("PATH").unwrap_or("".to_string()));
            }
            "Check CUDA exports in .bashrc" => {
                let res = environ::check_cuda_exports();
                if res.is_ok() {
                    if res.unwrap() {
                        println!("{} {}", style("[OK]").green(), "CUDA exports present");
                    } else {
                        println!("{} {}", style("[ERROR]").red(), "CUDA exports not present");
                    }
                } else {
                    println!(
                        "{} {}: {}",
                        style("[ERROR]").red(),
                        "CUDA exports check failed",
                        res.err().unwrap()
                    );
                }
            }
            "Insert CUDA exports to .bashrc" => {
                let res = environ::insert_cuda_exports();
                if res.is_ok() {
                    println!("{} {}", style("[OK]").green(), "CUDA exports inserted");
                } else {
                    println!(
                        "{} {}: {}",
                        style("[ERROR]").red(),
                        "CUDA exports failed to insert",
                        res.err().unwrap()
                    );
                }
            }
            _ => println!("Error"),
        }
    }
}

fn start_api() -> Result<(), Box<dyn Error>> {
    spawn_command("venv/bin/python main.py", "Run the API")?;
    Ok(())
}
