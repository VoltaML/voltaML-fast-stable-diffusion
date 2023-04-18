use std::error::Error;

use console::style;
use git2::{BranchType, Repository};

pub fn is_git_repo_up_to_date() -> Result<bool, Box<dyn Error>> {
    let repo = Repository::discover(".");
    let repo = repo?;

    let head = repo.head()?;
    let branch = head.shorthand().unwrap();

    let branch_ref = repo.find_branch(branch, BranchType::Local)?;
    let upstream = branch_ref.upstream()?;
    let upstream_oid = upstream.get().peel_to_commit()?.id();

    let head_oid = head.peel_to_commit()?.id();
    let is_up_to_date = head_oid == upstream_oid;

    return Ok(is_up_to_date);
}

pub fn update_git_repo() -> Result<String, Box<dyn Error>> {
    let repo = Repository::discover(".")?;
    let binding = repo.head()?;
    let current_branch = binding.shorthand().unwrap();

    repo.find_remote("origin")?
        .fetch(&[current_branch], None, None)?;

    let fetch_head = repo.find_reference("FETCH_HEAD")?;
    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;
    let analysis = repo.merge_analysis(&[&fetch_commit])?;
    if analysis.0.is_up_to_date() {
        Ok(format!("{} {}", style("[Ok]").red(), "Already up-to-date."))
    } else if analysis.0.is_fast_forward() {
        let refname = format!("refs/heads/{}", current_branch);
        let mut reference = repo.find_reference(&refname)?;
        reference.set_target(fetch_commit.id(), "Fast-Forward")?;
        repo.set_head(&refname)?;
        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
        Ok(format!(
            "{} {}",
            style("[Ok]").green(),
            "Repository updated."
        ))
    } else {
        println!(
            "{} {}",
            style("[!]").red(),
            "Cannot fast-forward, please merge manually."
        );
        Err("Cannot fast-forward, please merge manually.".into())
    }
}
