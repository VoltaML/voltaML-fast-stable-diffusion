use std::error::Error;

use crate::utils::shell::spawn_command;

pub fn clone_repo(url: &str, path: &str, branch_name: &str) -> Result<(), Box<dyn Error>> {
    spawn_command(
        &format!("git clone {} -b {} {}", url, branch_name, path),
        &format!("Clone repo {} into {}", url, path),
    )?;

    Ok(())
}
