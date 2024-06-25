use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_lab::*;

fn bench_chebyshev_eval(c: &mut Criterion) {
    let f = |x: f64, _: &mut NoArgs| Ok(f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x));
    let (xa, xb) = (-1.0, 1.0);
    let args = &mut 0;
    let nns = [1, 5, 10, 50, 100, 150, 200, 500, 1000, 1500, 2000];
    let mut group = c.benchmark_group("chebyshev_eval");
    for nn in &nns {
        group.throughput(Throughput::Elements(*nn as u64));
        group.bench_with_input(BenchmarkId::new("clenshaw", nn), nn, |b, &nn| {
            let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
            b.iter(|| interp.eval((xa + xb) / 2.0).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("trigonometric", nn), nn, |b, &nn| {
            let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
            b.iter(|| interp.eval_using_trig((xa + xb) / 2.0).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chebyshev_eval);
criterion_main!(benches);
