pub mod aitemplate;
pub mod nvidia;
pub mod python;
pub mod shell;

pub fn is_root() -> bool {
    let output = shell::run_command("whoami", "Check if root");
    if output.is_ok() {
        let output = output.unwrap();
        if output == "root" {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}
