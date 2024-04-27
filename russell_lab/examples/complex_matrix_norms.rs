use russell_lab::*;

fn main() -> Result<(), StrError> {
    // --- first matrix --- (reals only) ----------------------------------------
    #[rustfmt::skip]
    let a = ComplexMatrix::from(&[
        [cpx!(-3.0,0.0), cpx!(5.0,0.0), cpx!(7.0,0.0)],
        [cpx!( 2.0,0.0), cpx!(6.0,0.0), cpx!(4.0,0.0)],
        [cpx!( 0.0,0.0), cpx!(2.0,0.0), cpx!(8.0,0.0)],
    ]);
    let norm_one = complex_mat_norm(&a, Norm::One);
    let norm_inf = complex_mat_norm(&a, Norm::Inf);
    let norm_fro = complex_mat_norm(&a, Norm::Fro);
    let norm_max = complex_mat_norm(&a, Norm::Max);
    approx_eq(norm_one, 19.0, 1e-14);
    approx_eq(norm_inf, 15.0, 1e-14);
    approx_eq(norm_fro, f64::sqrt(207.0), 1e-14);
    approx_eq(norm_max, 8.0, 1e-14);

    // --- second matrix --- (reals only) ---------------------------------------
    #[rustfmt::skip]
    let a = ComplexMatrix::from(&[
        [cpx!(-3.0,0.0), cpx!(5.0,0.0), cpx!(7.0,0.0)],
        [cpx!( 2.0,0.0), cpx!(6.0,0.0), cpx!(4.0,0.0)],
        [cpx!( 0.0,0.0), cpx!(2.0,0.0), cpx!(8.0,0.0)],
        [cpx!( 2.0,0.0), cpx!(5.0,0.0), cpx!(9.0,0.0)],
        [cpx!( 3.0,0.0), cpx!(3.0,0.0), cpx!(3.0,0.0)],
    ]);
    let norm_one = complex_mat_norm(&a, Norm::One);
    let norm_inf = complex_mat_norm(&a, Norm::Inf);
    let norm_fro = complex_mat_norm(&a, Norm::Fro);
    let norm_max = complex_mat_norm(&a, Norm::Max);
    approx_eq(norm_one, 31.0, 1e-14);
    approx_eq(norm_inf, 16.0, 1e-14);
    approx_eq(norm_fro, f64::sqrt(344.0), 1e-14);
    approx_eq(norm_max, 9.0, 1e-14);

    // --- third matrix --- (real and imag) -------------------------------------
    #[rustfmt::skip]
    let a = ComplexMatrix::from(&[
        [cpx!(-3.0,1.0), cpx!(5.0,3.0), cpx!(7.0,-1.0)],
        [cpx!( 2.0,2.0), cpx!(6.0,2.0), cpx!(4.0,-2.0)],
        [cpx!( 0.0,3.0), cpx!(2.0,1.0), cpx!(8.0,-3.0)],
    ]);
    let mut fro = 0.0;
    for v in a.as_data() {
        fro += v.abs() * v.abs();
    }
    fro = f64::sqrt(fro);
    let norm_one = complex_mat_norm(&a, Norm::One);
    let norm_inf = complex_mat_norm(&a, Norm::Inf);
    let norm_fro = complex_mat_norm(&a, Norm::Fro);
    let norm_max = complex_mat_norm(&a, Norm::Max);
    let last_col_sum_abs = a.get(0, 2).abs() + a.get(1, 2).abs() + a.get(2, 2).abs();
    let first_row_sum_abs = a.get(0, 0).abs() + a.get(0, 1).abs() + a.get(0, 2).abs();
    approx_eq(norm_one, last_col_sum_abs, 1e-14);
    approx_eq(norm_inf, first_row_sum_abs, 1e-14);
    approx_eq(norm_fro, fro, 1e-14);
    approx_eq(norm_max, a.get(2, 2).abs(), 1e-14);
    Ok(())
}
