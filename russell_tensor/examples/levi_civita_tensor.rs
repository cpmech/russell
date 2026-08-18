use russell_tensor::{Rep, StrError, Tensor3};

fn main() -> Result<(), StrError> {
    let perm = Tensor3::constant_permutation(Rep::General, true)?;
    println!("{}", perm);
    Ok(())
}
