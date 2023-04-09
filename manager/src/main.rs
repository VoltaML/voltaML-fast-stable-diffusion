pub mod git;
pub mod install;
mod targets;
mod utils;

use console::style;
use dialoguer::{theme::ColorfulTheme, Select};

fn main() {
    // Check the Git repo update status
    if git::git::is_git_repo_up_to_date() {
        println!("{} {}", style("[Ok]").green(), "Already up-to-date.");
    } else {
        println!("{} {}", style("[!]").yellow(), "Update available.");
    }

    // Loop the menu
    loop {
        let items = vec!["Install", "Update", "Exit"];
        let response = Select::with_theme(&ColorfulTheme::default())
            .default(0)
            .items(&items)
            .interact()
            .unwrap();

        match response {
            0 => install::install(),
            1 => git::git::update_git_repo(),
            2 => std::process::exit(0),
            _ => println!("Error"),
        }
    }
}
