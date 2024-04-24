use plotpy::{Curve, Plot};
use russell_lab::{complex_vec_approx_eq, complex_vec_copy, ComplexVector, FFTw, StrError, Vector};
use rustfft::FftPlanner;

fn gen_data(size: usize, a: f64) -> (Vector, Vector) {
    let t = Vector::linspace(-1.0, 1.0, size).unwrap();
    let y = t.get_mapped(|t| f64::exp(-a * t * t));
    (t, y)
}

fn run_rfft(u: &mut ComplexVector) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(u.dim());
    fft.process(u.as_mut_data());
}

fn run_fftw(u: &mut ComplexVector) {
    let mut v = ComplexVector::new(u.dim());
    let mut fft = FFTw::new();
    fft.dft_1d(&mut v, u, false).unwrap();
    complex_vec_copy(u, &v).unwrap();
}

fn main() -> Result<(), StrError> {
    const L: usize = 501;
    const A: f64 = 10.0;

    // zero vectors
    let mut u_rfft = ComplexVector::new(L);
    let mut u_fftw = ComplexVector::new(L);
    run_rfft(&mut u_rfft);
    run_fftw(&mut u_fftw);
    complex_vec_approx_eq(&u_rfft, &u_fftw, 1e-50);

    // gaussian vectors
    let (t, y) = gen_data(L, A);
    let mut u_rfft = ComplexVector::from(y.as_data());
    let mut u_fftw = ComplexVector::from(y.as_data());
    run_rfft(&mut u_rfft);
    run_fftw(&mut u_fftw);
    complex_vec_approx_eq(&u_rfft, &u_fftw, 1e-13);

    // process the results
    const M: usize = L / 2 + 1;
    let ff = Vector::linspace(0.0, 1.0, M)?; // frequency domain
    let mut pp = Vector::new(M); // single-sided spectrum
    let den = L as f64;
    for i in 0..M {
        pp[i] = 2.0 * u_rfft[i].norm() / den;
    }

    // plot
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    curve1.draw(t.as_data(), y.as_data());
    curve2.draw(ff.as_data(), pp.as_data());
    let mut plot = Plot::new();
    plot.set_subplot(1, 2, 1)
        .add(&curve1)
        .grid_and_labels("time: $t$", "$y(t)$")
        .set_subplot(1, 2, 2)
        .add(&curve2)
        .grid_and_labels("frequency: $f$", "spectrum")
        .set_figure_size_points(700.0, 350.0)
        .save("/tmp/russell_lab/fftw_versus_rfft.svg")?;
    Ok(())
}
