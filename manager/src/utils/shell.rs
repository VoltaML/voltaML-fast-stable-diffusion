use console::style;
use shlex::split;
use std::process::{exit, Command};

use super::{cleanup, write_finished_step};

pub fn run_command(command: &str, step_name: &str) -> String {
    let command_args: Vec<String> = split(command).unwrap();
    let output = Command::new(&command_args[0])
        .args(&command_args[1..])
        .output();
    if output.is_ok() {
        let output = output.unwrap();
        if output.status.success() {
            write_finished_step(step_name);
            println!("{} {}", style("[OK]").green(), step_name);
            return String::from_utf8(output.stdout).unwrap();
        } else {
            println!("{} {}", style("[ERROR]").red(), step_name);
            cleanup();
            exit(1);
        }
    } else {
        println!("{} {}", style("[ERROR]").red(), step_name);
        cleanup();
        exit(1);
    }
}
