use std::{
    fs::File,
    io::{Read, Write},
};

pub mod shell;

#[allow(dead_code)]
pub fn get_finished_steps() -> Vec<String> {
    let mut finished_steps = Vec::new();
    let mut file = File::open("finished_steps.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    for line in contents.lines() {
        finished_steps.push(line.to_string());
    }
    return finished_steps;
}

pub fn write_finished_step(step: &str) {
    let mut file = File::create("finished_steps.txt").unwrap();
    file.write_all(step.as_bytes()).unwrap();
}

pub fn cleanup() {
    // TODO: Cleanup
}
