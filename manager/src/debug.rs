use std::error::Error;

use crate::utils::python::installed_packages;
use prettytable::{row, Table};

pub fn python_packages() -> Result<Table, Box<dyn Error>> {
    let mut table = Table::new();
    table.add_row(row![FGb->"Name", FYb->"Version"]);
    let packages = installed_packages()?;
    for package in packages {
        table.add_row(row![FG->package.name, FY->package.version]);
    }
    return Ok(table);
}
