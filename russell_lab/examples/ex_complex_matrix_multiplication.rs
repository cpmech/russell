use num_complex::Complex64;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // allocate matrices
    #[rustfmt::skip]
    let a = ComplexMatrix::from(&[
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new( 0.0,  1.0), Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
        [Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0), Complex64::new(-1.0, -1.0), Complex64::new(1.0, 0.0), Complex64::new( 1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new( 0.0,  1.0), Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0)],
        [Complex64::new(4.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new( 3.0, -1.0), Complex64::new(1.0, 0.0), Complex64::new( 1.0, 0.0)],
    ]);
    #[rustfmt::skip]
    let b = ComplexMatrix::from(&[
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0,  1.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(3.0, -1.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0,  1.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, -1.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(0.0,  1.0)],
    ]);
    const NOISE: Complex64 = Complex64::new(123.0, 456.0);
    #[rustfmt::skip]
    let mut c = ComplexMatrix::from(&[
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
        [NOISE, NOISE, NOISE],
    ]);

    // multiply matrices
    // c = (0.5 - 2i) ⋅ a ⋅ b
    let alpha = Complex64::new(0.5, -2.0);
    complex_mat_mat_mul(&mut c, alpha, &a, &b)?;

    // check
    #[rustfmt::skip]
    let correct = &[
        [Complex64::new(1.0, -4.0), Complex64::new(-1.0,  4.0), Complex64::new(-1.0, -13.0)],
        [Complex64::new(1.5, -6.0), Complex64::new( 1.0, -4.0), Complex64::new(-1.0, -21.5)],
        [Complex64::new(2.5,-10.0), Complex64::new(-1.0,  4.0), Complex64::new(-5.5, -20.5)],
        [Complex64::new(2.5,-10.0), Complex64::new( 1.0, -4.0), Complex64::new(14.5,  -7.0)],
    ];
    complex_mat_approx_eq(&c, correct, 1e-15);
    Ok(())
}
