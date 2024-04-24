use num_complex::Complex64;
use russell_lab::{complex_vec_approx_eq, complex_vec_copy, ComplexVector, FFTw};
use rustfft::FftPlanner;

fn fft_rfft(u: &mut ComplexVector) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(u.dim());
    fft.process(u.as_mut_data());
}

fn fft_fftw(u: &mut ComplexVector) {
    let mut v = ComplexVector::new(u.dim());
    let mut fft = FFTw::new();
    fft.dft_1d(&mut v, u, false).unwrap();
    complex_vec_copy(u, &v).unwrap();
}

fn main() {
    const SIZE: usize = 750;

    let mut test_rfft = ComplexVector::new(SIZE);
    let mut test_fftw = ComplexVector::new(SIZE);

    // Zero vectors
    fft_rfft(&mut test_rfft);
    fft_fftw(&mut test_fftw);
    complex_vec_approx_eq(&test_rfft, &test_fftw, 1e-50);

    for i in 0..SIZE {
        let ui = Complex64 {
            re: (-((i as f64 - SIZE as f64 / 2.0) / (SIZE / 10) as f64).powi(2)).exp(),
            im: 0.0,
        };
        test_rfft[i] = ui;
        test_fftw[i] = ui;
    }

    // Gaussian vectors
    fft_rfft(&mut test_rfft);
    fft_fftw(&mut test_fftw);
    complex_vec_approx_eq(&test_rfft, &test_fftw, 1e-13);
}
