use russell_lab::*;

fn main() -> Result<(), StrError> {
    let u = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0]);
    assert_eq!(vec_norm(&u, Norm::One), 11.0);
    assert_eq!(vec_norm(&u, Norm::Euc), 5.0);
    assert_eq!(vec_norm(&u, Norm::Fro), 5.0); // same as Euc
    assert_eq!(vec_norm(&u, Norm::Inf), 3.0);
    assert_eq!(vec_norm(&u, Norm::Max), 3.0); // same as Inf
    Ok(())
}
