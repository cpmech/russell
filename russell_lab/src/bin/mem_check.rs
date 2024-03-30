use num_complex::Complex64;
use russell_lab::{cpx, ComplexVector, FFTw};

fn main() {
    // check FFTW interface
    let mut fft = match FFTw::new() {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new): {}", e);
            return;
        }
    };

    let u = ComplexVector::from(&[cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(4.0, 0.0), cpx!(9.0, 0.0)]);
    let mut uu = ComplexVector::new(u.dim());

    match fft.dft_1d(&mut uu, &u, false, false) {
        Ok(_) => (),
        Err(e) => {
            println!("FAIL(execute): {}", e);
            return;
        }
    }
}
