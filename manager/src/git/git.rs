use std::error::Error;

use crate::utils::shell::{run_command, spawn_command};

pub fn is_git_repo_up_to_date() -> Result<bool, Box<dyn Error>> {
    spawn_command("git fetch", "Fetch git changes")?;

    let current_branch_raw =
        run_command(&"git rev-parse --abbrev-ref HEAD", "Get current git branch")?;
    let current_branch = current_branch_raw.trim();

    let remote_command = format!("git rev-parse origin/{}", current_branch);
    let remote_commit = run_command(&remote_command, "Check if git repo is up-to-date")?;

    let local_commit_raw = spawn_command("git rev-parse HEAD", "Check if git repo is up-to-date")?;
    let local_commit = local_commit_raw.trim();

    Ok(remote_commit == local_commit)
}

pub fn update_git_repo() -> Result<String, Box<dyn Error>> {
    let output = spawn_command("git pull", "Update git repo")?;
    Ok(output)
}
