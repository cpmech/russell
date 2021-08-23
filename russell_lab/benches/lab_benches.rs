use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_lab::*;

fn bench_add_vectors(c: &mut Criterion) {
    let sizes = &[1, 4, 16, 32, 64, 128];
    let mut group = c.benchmark_group("lab_add_vectors");
    for size in sizes {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let u = Vector::new(size);
            let v = Vector::new(size);
            let mut w = Vector::new(size);
            b.iter(|| add_vectors(&mut w, 1.0, &u, 1.0, &v));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_add_vectors);
criterion_main!(benches);
