use std::process::exit;

use console::style;
use git2::{BranchType, Repository};

pub fn is_git_repo_up_to_date() -> bool {
    let repo = Repository::discover(".");
    let repo = match repo {
        Ok(repo) => repo,
        Err(e) => {
            println!("{} {}", style("[!]").red(), e);
            exit(1)
        }
    };

    let head = repo.head().unwrap();
    let branch = head.shorthand().unwrap();

    let branch_ref = repo.find_branch(branch, BranchType::Local).unwrap();
    let upstream = branch_ref.upstream().unwrap();
    let upstream_oid = upstream.get().peel_to_commit().unwrap().id();

    let head_oid = head.peel_to_commit().unwrap().id();
    let is_up_to_date = head_oid == upstream_oid;

    return is_up_to_date;
}

pub fn update_git_repo() {
    let repo = Repository::discover(".").unwrap();
    let binding = repo.head().unwrap();
    let current_branch = binding.shorthand().unwrap();

    repo.find_remote("origin")
        .unwrap()
        .fetch(&[current_branch], None, None)
        .unwrap();

    let fetch_head = repo.find_reference("FETCH_HEAD").unwrap();
    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head).unwrap();
    let analysis = repo.merge_analysis(&[&fetch_commit]).unwrap();
    if analysis.0.is_up_to_date() {
        println!("{} {}", style("[Ok]").red(), "Already up-to-date.");
    } else if analysis.0.is_fast_forward() {
        let refname = format!("refs/heads/{}", current_branch);
        let mut reference = repo.find_reference(&refname).unwrap();
        reference
            .set_target(fetch_commit.id(), "Fast-Forward")
            .unwrap();
        repo.set_head(&refname).unwrap();
        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
            .unwrap();
    } else {
        println!(
            "{} {}",
            style("[!]").red(),
            "Cannot fast-forward, please merge manually."
        );
        exit(1)
    }
}
