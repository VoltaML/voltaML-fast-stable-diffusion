use dialoguer::{theme::ColorfulTheme, Input, Select};
use regex::Regex;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::Path;

pub fn inject_variable(name: &str, value: &str) {
    let envfile = Path::new(".env");
    let file = OpenOptions::new()
        .read(true)
        .open(&envfile)
        .expect("Failed to open file");
    let mut contents = String::new();
    let mut reader = BufReader::new(file);
    reader
        .read_to_string(&mut contents)
        .expect("Failed to read file");

    let re = Regex::new(&format!("{}=.*", name)).expect("Invalid regular expression");
    println!("Found: {}", re.find(&contents).is_some());
    let new_contents = re.replace(&contents, &format!("{}={}", name, value));

    println!("Writing to file:\n{}", new_contents);

    let mut outfile = File::create(&envfile).expect("Failed to create file");
    outfile
        .write_all(new_contents.as_bytes())
        .expect("Failed to write to file");
}

pub fn change_huggingface_token() {
    let input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Huggingface token")
        .interact_text()
        .unwrap();

    inject_variable("HUGGINGFACE_TOKEN", &input);
}

pub fn change_logging_level() {
    let items = vec!["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
    let response = Select::with_theme(&ColorfulTheme::default())
        .default(1)
        .items(&items)
        .interact()
        .unwrap();

    let level = match response {
        0 => "DEBUG",
        1 => "INFO",
        2 => "WARNING",
        3 => "ERROR",
        4 => "CRITICAL",
        _ => "INFO",
    };

    inject_variable("LOG_LEVEL", level);
}
