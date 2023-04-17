use console::style;
use shlex::split;
use std::{error::Error, fs, path::Path, process::Command};

pub fn run_command(command: &str, command_name: &str) -> Result<String, Box<dyn Error>> {
    let command_args: Vec<String> = split(command).unwrap();
    let output = Command::new(&command_args[0])
        .args(&command_args[1..])
        .output();
    if output.is_ok() {
        let output = output?;
        if output.status.success() {
            if command_name != "" {
                println!("{} {}", style("[OK]").green(), command_name);
            }
            return Ok(String::from_utf8(output.stdout)?);
        } else {
            let err = String::from_utf8(output.stderr)?;
            if command_name != "" {
                println!(
                    "{} {}: {}",
                    style("[ERROR]").red(),
                    command_name,
                    err.to_string()
                );
            }
            return Err(err.into());
        }
    } else {
        if command_name != "" {
            println!("{} {}", style("[ERROR]").red(), command_name);
        }
        return Err(output.err().unwrap().into());
    }
}

pub fn spawn_command(command: &str, command_name: &str) -> Result<String, Box<dyn Error>> {
    let command_args: Vec<String> = split(command).unwrap();
    let child = Command::new(&command_args[0])
        .args(&command_args[1..])
        .spawn()?;
    let output = child.wait_with_output();

    if output.is_ok() {
        let output = output?;
        if output.status.success() {
            if command_name != "" {
                println!("{} {}", style("[OK]").green(), command_name);
            }
            return Ok(String::from_utf8(output.stdout)?);
        } else {
            let err = String::from_utf8(output.stderr)?;
            if command_name != "" {
                println!(
                    "{} {}: {}",
                    style("[ERROR]").red(),
                    command_name,
                    err.to_string()
                );
            }
            return Err(err.into());
        }
    } else {
        if command_name != "" {
            println!("{} {}", style("[ERROR]").red(), command_name);
        }
        return Err(output.err().unwrap().into());
    }
}

pub fn move_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;

        for entry in src.read_dir()? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            if src_path.is_dir() {
                move_dir_recursive(&src_path, &dst_path)?;
            } else {
                fs::rename(&src_path, &dst_path)?;
            }
        }
    } else {
        fs::rename(src, dst)?;
    }

    fs::remove_dir_all(src)?;
    Ok(())
}
