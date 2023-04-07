mod targets;

use dialoguer::{theme::ColorfulTheme, Select};

fn main() {
    let items = vec!["Windows", "WSL", "Ubuntu"];
    let response = Select::with_theme(&ColorfulTheme::default())
        .default(0)
        .items(&items)
        .interact()
        .unwrap();

    match response {
        0 => targets::windows::install(),
        1 => targets::wsl::install(),
        2 => targets::ubuntu::install(),
        _ => println!("Error"),
    }
}
