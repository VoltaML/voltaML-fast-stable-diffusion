use crate::utils::shell::run_command;
use console::style;

pub fn update() {
    println!("{} {}", style("[!]").yellow(), "Updating packages...");
    let res = run_command("sudo apt update", "Update apt");
    if res.is_err() {
        println!("Error: {}", res.err().unwrap());
    }
}

pub fn upgrade() {
    println!("{} {}", style("[!]").yellow(), "Upgrading packages...");
    let res = run_command("sudo apt upgrade -y", "Upgrade apt");
    if res.is_err() {
        println!("Error: {}", res.err().unwrap());
    }
}

pub fn install(package: &str) {
    println!(
        "{} {}",
        style("[!]").yellow(),
        format!("Installing {}...", package)
    );
    let res = run_command(
        format!("sudo apt install -y {}", package).as_str(),
        "Install package",
    );
    if res.is_err() {
        println!("Error: {}", res.err().unwrap());
    }
}
