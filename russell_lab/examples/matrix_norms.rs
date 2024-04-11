use russell_lab::*;

fn main() -> Result<(), StrError> {
    #[rustfmt::skip]
    let a = Matrix::from(&[
        [-3.0, 5.0, 7.0],
        [ 2.0, 6.0, 4.0],
        [ 0.0, 2.0, 8.0],
    ]);
    let norm_one = mat_norm(&a, Norm::One);
    let norm_inf = mat_norm(&a, Norm::Inf);
    let norm_fro = mat_norm(&a, Norm::Fro);
    let norm_max = mat_norm(&a, Norm::Max);
    approx_eq(norm_one, 19.0, 1e-14);
    approx_eq(norm_inf, 15.0, 1e-14);
    approx_eq(norm_fro, f64::sqrt(207.0), 1e-14);
    approx_eq(norm_max, 8.0, 1e-14);

    #[rustfmt::skip]
    let a = Matrix::from(&[
        [-3.0, 5.0, 7.0],
        [ 2.0, 6.0, 4.0],
        [ 0.0, 2.0, 8.0],
        [ 2.0, 5.0, 9.0],
        [ 3.0, 3.0, 3.0],
    ]);
    let norm_one = mat_norm(&a, Norm::One);
    let norm_inf = mat_norm(&a, Norm::Inf);
    let norm_fro = mat_norm(&a, Norm::Fro);
    let norm_max = mat_norm(&a, Norm::Max);
    approx_eq(norm_one, 31.0, 1e-14);
    approx_eq(norm_inf, 16.0, 1e-14);
    approx_eq(norm_fro, f64::sqrt(344.0), 1e-14);
    approx_eq(norm_max, 9.0, 1e-14);
    Ok(())
}
