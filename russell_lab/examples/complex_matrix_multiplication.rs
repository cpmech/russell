use russell_lab::*;

fn main() -> Result<(), StrError> {
    // allocate matrices
    #[rustfmt::skip]
    let a = ComplexMatrix::from(&[
        [cpx!(1.0, 0.0), cpx!(2.0, 0.0), cpx!( 0.0,  1.0), cpx!(1.0, 0.0), cpx!(-1.0, 0.0)],
        [cpx!(2.0, 0.0), cpx!(3.0, 0.0), cpx!(-1.0, -1.0), cpx!(1.0, 0.0), cpx!( 1.0, 0.0)],
        [cpx!(1.0, 0.0), cpx!(2.0, 0.0), cpx!( 0.0,  1.0), cpx!(4.0, 0.0), cpx!(-1.0, 0.0)],
        [cpx!(4.0, 0.0), cpx!(0.0, 0.0), cpx!( 3.0, -1.0), cpx!(1.0, 0.0), cpx!( 1.0, 0.0)],
    ]);
    #[rustfmt::skip]
    let b = ComplexMatrix::from(&[
        [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0,  1.0)],
        [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(3.0, -1.0)],
        [cpx!(0.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0,  1.0)],
        [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(1.0, -1.0)],
        [cpx!(0.0, 0.0), cpx!(2.0, 0.0), cpx!(0.0,  1.0)],
    ]);
    const NOISE: Complex64 = cpx!(123.0, 456.0);
    #[rustfmt::skip]
    let mut c = ComplexMatrix::from(&[
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
    ]);

    // multiply matrices
    // c = (0.5 - 2i) ⋅ a ⋅ b
    let alpha = cpx!(0.5, -2.0);
    let beta = cpx!(0.0, 0.0);
    complex_mat_mat_mul(&mut c, alpha, &a, &b, beta)?;

    // check
    #[rustfmt::skip]
    let correct = &[
        [cpx!(1.0, -4.0), cpx!(-1.0,  4.0), cpx!(-1.0, -13.0)],
        [cpx!(1.5, -6.0), cpx!( 1.0, -4.0), cpx!(-1.0, -21.5)],
        [cpx!(2.5,-10.0), cpx!(-1.0,  4.0), cpx!(-5.5, -20.5)],
        [cpx!(2.5,-10.0), cpx!( 1.0, -4.0), cpx!(14.5,  -7.0)],
    ];
    complex_mat_approx_eq(&c, correct, 1e-15);
    Ok(())
}
