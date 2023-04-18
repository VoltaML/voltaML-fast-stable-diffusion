use crate::utils::shell::run_command;

pub mod ubuntu;
pub mod windows;

#[derive(PartialEq)]
pub enum Target {
    Windows,
    Linux,
    WSL,
    Unknown,
}

impl ToString for Target {
    fn to_string(&self) -> String {
        match *self {
            Target::Linux => "Linux".to_string(),
            Target::Windows => "Windows".to_string(),
            Target::WSL => "WSL".to_string(),
            Target::Unknown => "Unknown".to_string(),
        }
    }
}

pub fn detect_target() -> Target {
    if cfg!(target_os = "windows") {
        return Target::Windows;
    } else if cfg!(target_os = "linux") {
        let uname = run_command("uname -r", "");
        if uname.is_ok() {
            let uname = uname.unwrap();
            if uname.contains("microsoft") {
                Target::WSL
            } else {
                Target::Linux
            }
        } else {
            Target::Linux
        }
    } else {
        Target::Unknown
    }
}
