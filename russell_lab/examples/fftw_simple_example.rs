use num_complex::Complex64;
use plotpy::{Curve, Plot, RayEndpoint, SuperTitleParams, Text};
use russell_lab::math::PI;
use russell_lab::{cpx, ComplexVector, FFTw, StrError};

const OUT_DIR: &str = "/tmp/russell_lab/";

fn main() -> Result<(), StrError> {
    // constants
    const F: f64 = 1000.0; // sampling frequency
    const T: f64 = 1.0 / F; // sampling period
    const L: usize = 1500; // length of signal

    // generate a signal with a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz sinusoid of amplitude 1.0
    let mut t = vec![0.0; L]; // time vector
    let mut u = vec![0.0; L]; // original signal
    let mut z = ComplexVector::new(L); // complex array
    for i in 0..L {
        t[i] = (i as f64) * T;
        u[i] = 0.7 * f64::sin(2.0 * PI * 50.0 * t[i]) + f64::sin(2.0 * PI * 120.0 * t[i]);
        z[i] = cpx!(u[i], 0.0);
    }

    // perform the DFT on the original signal
    let mut fft = FFTw::new();
    let mut zz = ComplexVector::new(L);
    fft.dft_1d(&mut zz, &z, false)?;

    // process the results
    const M: usize = L / 2 + 1;
    let mut pp = vec![0.0; M]; // single-sided spectrum of the original signal
    let mut ff = vec![0.0; M]; // frequency domain f
    let den = L as f64;
    for i in 0..M {
        pp[i] = 2.0 * zz[i].norm() / den;
        ff[i] = F * (i as f64) / den;
    }

    // plot
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut text1 = Text::new();
    curve1.draw(&&t[0..50], &&u[0..50]);
    curve2.set_line_style("--").set_line_color("green");
    curve2.draw_ray(0.0, 0.7, RayEndpoint::Horizontal);
    curve2.draw_ray(0.0, 1.0, RayEndpoint::Horizontal);
    curve3.draw(&ff, &pp);
    text1.draw(52.0, 0.02, "50 Hz");
    text1.draw(122.0, 0.02, "120 Hz");
    let mut params = SuperTitleParams::new();
    params.set_align_vertical("bottom").set_y(0.9);
    let mut plot = Plot::new();
    let path = format!("{}/fftw_simple_example.svg", OUT_DIR);
    plot.set_subplot(2, 1, 1)
        .add(&curve1)
        .grid_labels_legend("$t\\,[\\mu s]$", "$u$")
        .set_subplot(2, 1, 2)
        .add(&curve2)
        .add(&curve3)
        .add(&text1)
        .grid_labels_legend("$f\\,[Hz]$", "spectrum")
        .set_figure_size_points(600.0, 600.0)
        .set_super_title(
            "Signal with a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz sinusoid of amplitude 1.0",
            Some(params),
        )
        .save(path.as_str())?;
    Ok(())
}
