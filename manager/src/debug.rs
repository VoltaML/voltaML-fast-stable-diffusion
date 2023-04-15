use crate::utils::python::installed_packages;
use prettytable::{row, Table};

pub fn python_packages() -> Table {
    let mut table = Table::new();
    table.add_row(row![FGb->"Name", FYb->"Version"]);
    let packages = installed_packages().unwrap();
    for package in packages {
        table.add_row(row![FG->package.name, FY->package.version]);
    }
    return table;
}
