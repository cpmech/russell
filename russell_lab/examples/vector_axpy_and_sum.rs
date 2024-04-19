use russell_lab::*;

fn main() -> Result<(), StrError> {
    // axpy
    let u = Vector::from(&[10.0, 20.0, 30.0]);
    let mut v = Vector::from(&[10.0, 20.0, 30.0]);
    vec_update(&mut v, 0.1, &u)?;
    let correct = "┌    ┐\n\
                   │ 11 │\n\
                   │ 22 │\n\
                   │ 33 │\n\
                   └    ┘";
    assert_eq!(format!("{}", v), correct);

    // sum
    let w = Vector::filled(3, 1.0);
    let mut z = Vector::new(3);
    vec_add(&mut z, 1.0, &v, 100.0, &w)?;
    let correct = "┌     ┐\n\
                   │ 111 │\n\
                   │ 122 │\n\
                   │ 133 │\n\
                   └     ┘";
    assert_eq!(format!("{}", z), correct);
    Ok(())
}
