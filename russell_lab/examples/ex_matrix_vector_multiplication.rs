use russell_lab::*;

fn main() -> Result<(), StrError> {
    // allocate matrix
    #[rustfmt::skip]
    let a = Matrix::from(&[
        [0.1, 0.2, 0.3],
        [1.0, 0.2, 0.3],
        [2.0, 0.2, 0.3],
        [3.0, 0.2, 0.3],
    ]);
    let (m, n) = a.dims();

    // allocate u
    let u = Vector::from(&[10.0, 20.0, 30.0]);

    // matrix-vector multiplication
    let mut v = Vector::new(m);
    mat_vec_mul(&mut v, 0.5, &a, &u)?;

    // check
    let mut half_a_times_u = Vector::new(m);
    for i in 0..m {
        for j in 0..n {
            half_a_times_u[i] += 0.5 * a.get(i, j) * u[j];
        }
    }
    vec_approx_eq(v.as_data(), half_a_times_u.as_data(), 1e-14);
    Ok(())
}
