use crate::targets;
use dialoguer::{theme::ColorfulTheme, Select};

pub fn install() {
    let items = vec!["Windows", "WSL", "Ubuntu"];
    let response = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    match response {
        0 => targets::windows::install(),
        1 => targets::ubuntu::install(true),
        2 => targets::ubuntu::install(false),
        _ => println!("Error"),
    }
}
