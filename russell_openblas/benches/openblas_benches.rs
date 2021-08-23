use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_openblas::*;

fn benchmark_add_vectors(c: &mut Criterion) {
    let sizes = &[1, 4, 16, 32, 64, 128];
    let mut group = c.benchmark_group("openblas_add_vectors");
    for size in sizes {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("Native", size), size, |b, &size| {
            let u = vec![0.0; size];
            let v = vec![0.0; size];
            let mut w = vec![0.0; size];
            b.iter(|| add_vectors_native(&mut w, 1.0, &u, 1.0, &v));
        });
        group.bench_with_input(BenchmarkId::new("Oblas", size), size, |b, &size| {
            let u = vec![0.0; size];
            let v = vec![0.0; size];
            let mut w = vec![0.0; size];
            b.iter(|| add_vectors_oblas(&mut w, 1.0, &u, 1.0, &v));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_add_vectors);
criterion_main!(benches);
