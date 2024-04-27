use russell_lab::*;

fn main() -> Result<(), StrError> {
    // Example from Intel MKL

    // matrix
    #[rustfmt::skip]
    let mut a = ComplexMatrix::from(&[
	   [cpx!( 1.23, -5.50), cpx!(-2.14, -1.12), cpx!(-4.30, -7.10), cpx!( 1.27,  7.29)],
	   [cpx!( 7.91, -5.38), cpx!(-9.92, -0.79), cpx!(-6.47,  2.52), cpx!( 8.90,  6.92)],
	   [cpx!(-9.80, -4.86), cpx!(-9.18, -1.12), cpx!(-6.51, -2.67), cpx!(-8.82,  1.25)],
	   [cpx!(-7.32,  7.57), cpx!( 1.37,  0.43), cpx!(-5.86,  7.38), cpx!( 5.41,  5.37)],
	]);

    // right-hand-side
    #[rustfmt::skip]
    let mut b = ComplexVector::from(&[
	   cpx!( 8.33, -7.32),
       cpx!(-6.18, -4.80),
       cpx!(-5.71, -2.80),
       cpx!(-1.60,  3.08),
	]);

    // save copies
    let a_copy = a.clone();
    let b_copy = b.clone();

    // solve b := x := A⁻¹ b
    complex_solve_lin_sys(&mut b, &mut a).unwrap();

    // alias for convenience
    let x = &b;

    // compute a times x
    let m = a.nrow();
    let mut ax = ComplexVector::new(m);
    let one = cpx!(1.0, 0.0);
    complex_mat_vec_mul(&mut ax, one, &a_copy, x)?;

    // print results
    println!("a =\n{:.3}", a_copy);
    println!("x =\n{:.3}", x);
    println!("b =\n{:.3}", b_copy);
    println!("a ⋅ x = b = \n{:.3}", ax);

    // check
    complex_vec_approx_eq(&ax, &b_copy, 1e-13);
    Ok(())
}
