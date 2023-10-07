use num_complex::Complex64;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // Example from Intel MKL

    // matrix
    #[rustfmt::skip]
    let mut a = ComplexMatrix::from(&[
	   [Complex64::new( 1.23, -5.50), Complex64::new(-2.14, -1.12), Complex64::new(-4.30, -7.10), Complex64::new( 1.27,  7.29)],
	   [Complex64::new( 7.91, -5.38), Complex64::new(-9.92, -0.79), Complex64::new(-6.47,  2.52), Complex64::new( 8.90,  6.92)],
	   [Complex64::new(-9.80, -4.86), Complex64::new(-9.18, -1.12), Complex64::new(-6.51, -2.67), Complex64::new(-8.82,  1.25)],
	   [Complex64::new(-7.32,  7.57), Complex64::new( 1.37,  0.43), Complex64::new(-5.86,  7.38), Complex64::new( 5.41,  5.37)],
	]);

    // right-hand-side
    #[rustfmt::skip]
    let mut b = ComplexVector::from(&[
	   Complex64::new( 8.33, -7.32),
       Complex64::new(-6.18, -4.80),
       Complex64::new(-5.71, -2.80),
       Complex64::new(-1.60,  3.08),
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
    let one = Complex64::new(1.0, 0.0);
    complex_mat_vec_mul(&mut ax, one, &a_copy, x)?;

    // print results
    println!("a =\n{:.3}", a_copy);
    println!("x =\n{:.3}", x);
    println!("b =\n{:.3}", b_copy);
    println!("a ⋅ x = b = \n{:.3}", ax);

    // check
    complex_vec_approx_eq(ax.as_data(), b_copy.as_data(), 1e-13);
    Ok(())
}
