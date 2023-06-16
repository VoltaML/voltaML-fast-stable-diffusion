use dialoguer::{theme::ColorfulTheme, Input, Select};
use regex::Regex;
use shellexpand::tilde;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::Path;

pub fn create_env_file() {
    if Path::new(".env").exists() {
        return;
    } else {
        // Copy the .env.example file to .env
        let mut file = File::create(".env").expect("Failed to create file");
        // Read the contents of the example file
        let mut contents = String::new();
        let mut reader = BufReader::new(
            File::open("example.env")
                .expect("Failed to open file .env.example. Please create it manually."),
        );
        reader
            .read_to_string(&mut contents)
            .expect("Failed to read file");
        // Write the contents to the new file
        file.write_all(contents.as_bytes())
            .expect("Failed to write to file");
    }
}

pub fn inject_variable(name: &str, value: &str) {
    create_env_file();
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
    let new_contents = re.replace(&contents, &format!("{}={}", name, value));

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

pub fn change_discord_bot_token() {
    let input: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Discord bot token")
        .interact_text()
        .unwrap();

    inject_variable("DISCORD_BOT_TOKEN", &input);
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

pub fn check_cuda_exports() -> Result<bool, Box<dyn Error>> {
    let mut file = OpenOptions::new()
        .read(true)
        .open(Path::new(&tilde("~/.bashrc").to_string()))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(
        contents.contains("export PATH=\"/usr/local/cuda/bin:$PATH\"")
            && contents
                .contains("export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:$LD_LIBRARY_PATH\""),
    )
}

pub fn insert_cuda_exports() -> Result<(), Box<dyn Error>> {
    // Insert export commands into .bashrc
    let mut file = OpenOptions::new()
        .append(true)
        .open(Path::new(&tilde("~/.bashrc").to_string()))?;

    file.write_all(b"\n")?;
    file.write_all(b"export PATH=\"/usr/local/cuda/bin:$PATH\"\n")?;
    file.write_all(b"export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:$LD_LIBRARY_PATH\"\n")?;

    // Insert into current shell
    // PATH
    let path = std::env::var("PATH").unwrap_or("".to_string());
    if path.is_empty() {
        std::env::set_var("PATH", "/usr/local/cuda/bin");
    } else {
        std::env::set_var("PATH", format!("/usr/local/cuda/bin:{}", path));
    }

    // LD_LIBRARY_PATH
    let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or("".to_string());
    if ld.is_empty() {
        std::env::set_var("LD_LIBRARY_PATH", "/usr/local/cuda/lib64");
    } else {
        std::env::set_var("LD_LIBRARY_PATH", format!("/usr/local/cuda/lib64:{}", ld));
    }

    Ok(())
}
