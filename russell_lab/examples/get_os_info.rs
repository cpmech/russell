//! Prints the operating-system and hardware report.

use russell_lab::get_os_info;

fn main() {
    println!("{}", get_os_info());
}
