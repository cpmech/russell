use russell_lab::*;

fn main() -> Result<(), StrError> {
    // allocate input matrices
    #[rustfmt::skip]
    let a = Matrix::from(&[
        [1.0, 2.0,  0.0, 1.0, -1.0],
        [2.0, 3.0, -1.0, 1.0,  1.0],
        [1.0, 2.0,  0.0, 4.0, -1.0],
        [4.0, 0.0,  3.0, 1.0,  1.0],
    ]);
    #[rustfmt::skip]
    let b = Matrix::from(&[
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0],
    ]);

    // allocate output matrix
    // (NOISE will be replaced by the results)
    const NOISE: f64 = 1234.0;
    #[rustfmt::skip]
    let mut c = Matrix::from(&[
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
    ]);

    // matrix multiplication
    // c = 0.5⋅a⋅b
    mat_mat_mul(&mut c, 0.5, &a, &b, 0.0)?;

    // check
    #[rustfmt::skip]
    let correct = Matrix::from(&[
       [ 1.0, -1.0,  3.5],
       [ 1.5,  1.0,  4.5],
       [ 2.5, -1.0,  5.0],
       [ 2.5,  1.0,  2.0],
    ]);
    mat_approx_eq(&c, &correct, 1e-15);
    Ok(())
}
