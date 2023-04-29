use crate::targets;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install() {
    let items = vec!["Windows", "WSL", "Ubuntu"];
    let response = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    let branches = vec!["Main", "Experimental"];
    let branch = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&branches)
        .interact()
        .unwrap();

    match response {
        0 => targets::windows::install(),
        1 => targets::ubuntu::install(true, branch == 1),
        2 => targets::ubuntu::install(false, branch == 1),
        _ => println!("Error"),
    }
}
