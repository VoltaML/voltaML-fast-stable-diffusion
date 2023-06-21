use std::error::Error;

use crate::utils::shell::spawn_command;

pub fn checkout_commit(repo_path_str: &str, commit_string: &str) -> Result<(), Box<dyn Error>> {
    spawn_command(
        &format!(
            "bash -c \"cd {} && git checkout {}\"",
            repo_path_str, commit_string
        ),
        &format!(
            "Checked out to commit {} in {}",
            commit_string, repo_path_str
        ),
    )?;

    Ok(())
}

pub fn checkout_branch(repo_path_str: &str, branch_name: &str) -> Result<(), Box<dyn Error>> {
    spawn_command(
        &format!(
            "git checkout --work-tree=\"{}\" {}",
            repo_path_str, branch_name
        ),
        &format!("Checkout branch {} in {}", branch_name, repo_path_str),
    )?;

    Ok(())
}
