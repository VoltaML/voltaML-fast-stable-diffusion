use crate::targets;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install() {
    let items = vec!["Windows", "WSL", "Ubuntu"];
    let response = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    let branches = vec!["main", "experimental"];
    let branch_id = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&branches)
        .interact()
        .unwrap();
    let branch = branches[branch_id].to_string();

    match response {
        0 => targets::windows::install(&branch),
        1 => targets::ubuntu::install(&branch, true),
        2 => targets::ubuntu::install(&branch, false),
        _ => println!("Error"),
    }
}
