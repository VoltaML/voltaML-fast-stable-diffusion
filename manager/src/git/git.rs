use std::error::Error;

use crate::utils::shell::spawn_command;

pub fn is_git_repo_up_to_date() -> Result<bool, Box<dyn Error>> {
    spawn_command("git fetch", "Fetch git changes")?;

    let current_branch = spawn_command(
        &format!("git rev-parse --abbrev-ref HEAD"),
        "Get current git branch",
    )?;

    println!("Current branch: {}", current_branch);

    let binding = spawn_command(
        &format!("git rev-parse origin/{}", current_branch),
        "Check if git repo is up-to-date",
    )?;
    let remote_commit = binding.trim();

    let binding = spawn_command("git rev-parse HEAD", "Check if git repo is up-to-date")?;
    let local_commit = binding.trim();

    Ok(remote_commit == local_commit)
}

pub fn update_git_repo() -> Result<String, Box<dyn Error>> {
    let output = spawn_command("git pull", "Update git repo")?;
    Ok(output)
}
