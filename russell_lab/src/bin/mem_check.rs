use num_complex::Complex64;
use russell_lab::{cpx, ComplexVector, FourierTransform1d};

fn main() {
    // check FFTW interface
    let mut fft = match FourierTransform1d::new() {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new): {}", e);
            return;
        }
    };

    let u = ComplexVector::from(&[cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(4.0, 0.0), cpx!(9.0, 0.0)]);
    let mut uu = ComplexVector::new(u.dim());

    match fft.execute(&mut uu, &u, false, false) {
        Ok(_) => (),
        Err(e) => {
            println!("FAIL(execute): {}", e);
            return;
        }
    }
}
