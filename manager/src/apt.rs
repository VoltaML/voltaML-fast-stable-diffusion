use std::error::Error;

use crate::utils::shell::spawn_command;
use console::style;

pub fn update() {
    println!("{} {}", style("[!]").yellow(), "Updating packages...");
    let res = spawn_command("sudo apt update", "Update apt");
    if res.is_err() {
        println!("Error: {}", res.err().unwrap());
    }
}

pub fn upgrade() {
    println!("{} {}", style("[!]").yellow(), "Upgrading packages...");
    let res = spawn_command("sudo apt upgrade -y", "Upgrade apt");
    if res.is_err() {
        println!("Error: {}", res.err().unwrap());
    }
}

pub fn install(package: &str) -> Result<(), Box<dyn Error>> {
    println!(
        "{} {}",
        style("[!]").yellow(),
        format!("Installing {}...", package)
    );
    spawn_command(
        format!("sudo apt install -y {}", package).as_str(),
        "Install package",
    )?;
    Ok(())
}
