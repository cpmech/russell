use russell_lab::*;

fn main() -> Result<(), StrError> {
    // scale
    let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    vec_scale(&mut u, 0.5);
    let correct = "┌     ┐\n\
                   │ 0.5 │\n\
                   │   1 │\n\
                   │ 1.5 │\n\
                   └     ┘";
    assert_eq!(format!("{}", u), correct);

    // copy
    let mut v = Vector::from(&[-1.0, -2.0, -3.0]);
    vec_copy(&mut v, &u)?;
    assert_eq!(format!("{}", v), correct);
    Ok(())
}
