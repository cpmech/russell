use russell_sparse::StrError;
use russell_sparse::get_system_info;

fn main() -> Result<(), StrError> {
    let info = get_system_info();
    println!("{}", info);
    Ok(())
}
