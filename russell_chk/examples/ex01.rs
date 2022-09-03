use num_complex::Complex64;
use russell_chk::{approx_eq, assert_deriv_approx_eq, assert_vec_approx_eq, complex_approx_eq};

fn main() {
    // check float point number
    approx_eq(0.0000123, 0.000012, 1e-6);

    // check vector of float point numbers
    assert_vec_approx_eq!(&[0.01, 0.012], &[0.012, 0.01], 1e-2);

    // check derivative using central differences
    struct Arguments {}
    let f = |x: f64, _: &mut Arguments| -x;
    let args = &mut Arguments {};
    let at_x = 8.0;
    let dfdx = -1.01;
    assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);

    // check complex numbers
    complex_approx_eq(Complex64::new(1.0, 8.0), Complex64::new(1.001, 8.0), 1e-2);
}
