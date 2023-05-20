use crate::targets;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Select};

pub fn install() {
    let response = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("{} Project will be installed in CURRENT directory. Are you sure you want to install here?", style("[WARNING]").yellow()))
        .interact()
        .unwrap_or(false);
    if !response {
        return;
    }

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
        0 => targets::windows::install(branch == 1),
        1 => targets::ubuntu::install(true, branch == 1),
        2 => targets::ubuntu::install(false, branch == 1),
        _ => println!("Error"),
    }
}
