use russell_lab::*;

fn main() -> Result<(), StrError> {
    let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
    let v = Vector::from(&[4.0, 3.0, 2.0]);
    let m = u.dim();
    let n = v.dim();
    let mut a = Matrix::new(m, n);
    vec_outer(&mut a, 0.5, &u, &v)?;
    let correct = Matrix::from(&[
        [2.0, 1.5, 1.0], //
        [4.0, 3.0, 2.0], //
        [6.0, 4.5, 3.0], //
        [8.0, 6.0, 4.0], //
    ]);
    mat_approx_eq(&a, &correct, 1e-15);
    Ok(())
}
