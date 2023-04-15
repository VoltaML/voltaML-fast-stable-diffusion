use console::style;
use shlex::split;
use std::{fmt::Error, process::Command};

pub fn run_command(command: &str, command_name: &str) -> Result<String, Error> {
    let command_args: Vec<String> = split(command).unwrap();
    let output = Command::new(&command_args[0])
        .args(&command_args[1..])
        .output();
    if output.is_ok() {
        let output = output.unwrap();
        if output.status.success() {
            println!("{} {}", style("[OK]").green(), command_name);
            return Ok(String::from_utf8(output.stdout).unwrap());
        } else {
            println!("{} {}", style("[ERROR]").red(), command_name);
            return Err(Error);
        }
    } else {
        println!("{} {}", style("[ERROR]").red(), command_name);
        return Err(Error);
    }
}
