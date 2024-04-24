use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use num_complex::Complex64;
use russell_lab::{complex_vec_copy, cpx, ComplexVector, FFTw, Vector};
use rustfft::FftPlanner;

const A: f64 = 10.0;

fn gen_data(size: usize) -> ComplexVector {
    let t = Vector::linspace(-1.0, 1.0, size).unwrap();
    let mut u = ComplexVector::new(size);
    for i in 0..size {
        u[i] = cpx!(f64::exp(-A * t[i] * t[i]), 0.0);
    }
    u
}

fn bench_fft(c: &mut Criterion) {
    let sizes: Vec<usize> = (1..1101).collect();
    // let sizes = [1];
    let mut group = c.benchmark_group("fft");
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("RustFFT", size), size, |b, &size| {
            let u = gen_data(size);
            let mut v = ComplexVector::new(size);
            let mut planner = FftPlanner::<f64>::new();
            let fft = planner.plan_fft_forward(size);
            b.iter(|| {
                complex_vec_copy(&mut v, &u).unwrap();
                fft.process(v.as_mut_data());
            });
        });
        group.bench_with_input(BenchmarkId::new("FFTW", size), size, |b, &size| {
            let u = gen_data(size);
            let mut v = ComplexVector::new(size);
            let mut fft = FFTw::new();
            b.iter(|| {
                fft.dft_1d(&mut v, &u, false).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);
