use num_complex::Complex64;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // Example from:
    // https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/zgesv_example.html

    #[rustfmt::skip]
    let mut a = ComplexMatrix::from(&[
        [Complex64::new(-1.34, 2.55), Complex64::new( 0.28, 3.17), Complex64::new(-6.39,-2.20), Complex64::new( 0.72,-0.92)],
        [Complex64::new(-0.17,-1.41), Complex64::new( 3.31,-0.15), Complex64::new(-0.15, 1.34), Complex64::new( 1.29, 1.38)],
        [Complex64::new(-3.29,-2.39), Complex64::new(-1.91, 4.42), Complex64::new(-0.14,-1.35), Complex64::new( 1.72, 1.35)],
        [Complex64::new( 2.41, 0.39), Complex64::new(-0.56, 1.47), Complex64::new(-0.83,-0.69), Complex64::new(-1.96, 0.67)],
    ]);

    let mut b = ComplexVector::from(&[
        Complex64::new(26.26, 51.78),
        Complex64::new(6.43, -8.68),
        Complex64::new(-5.75, 25.31),
        Complex64::new(1.16, 2.57),
    ]);

    // solve b := x := A⁻¹ b
    complex_solve_lin_sys(&mut b, &mut a).unwrap();

    // print results
    println!("a (after) =\n{:.3}", a);
    println!("b (after) =\n{:.3}", b);

    // expected results
    let correct = ComplexVector::from(&[
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -3.0),
        Complex64::new(-4.0, -5.0),
        Complex64::new(0.0, 6.0),
    ]);
    println!("expected =\n{:.3}", correct);
    complex_vec_approx_eq(b.as_data(), correct.as_data(), 1e-14);
    Ok(())
}
