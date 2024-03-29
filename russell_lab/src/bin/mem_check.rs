use russell_lab::FourierTransform1d;

fn main() {
    // check FFTW interface
    let _fft_1d = match FourierTransform1d::new() {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new FourierTransform1d): {}", e);
            return;
        }
    };
}
