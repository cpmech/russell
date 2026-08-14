use russell_sparse::StrError;
use russell_sparse::get_system_info_linux;

fn main() -> Result<(), StrError> {
    let info = get_system_info_linux();
    println!("{}", info);
    Ok(())
}
