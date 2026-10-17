//! Square matrix-matrix multiplication throughput.
//!
//! The native path's `dgemm` is a naive triple loop (see `native/blas.rs`), and
//! every decomposition in the crate eventually leans on BLAS-3 through it. This
//! measures the baseline so that a blocked kernel's effect is a number, not an
//! opinion, and so that the question "is a blocked LAPACK port worth it?" has
//! evidence behind it.
//!
//! Throughput is reported as elements per iteration; multiply by 2 to get FLOPs
//! (one multiply and one add per inner-loop step), so `Elements` here is really
//! "inner-loop steps".

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_lab::{Matrix, mat_mat_mul};

fn bench_mat_mat_mul(c: &mut Criterion) {
    let sizes = [64usize, 128, 256, 384, 512];
    let mut group = c.benchmark_group("mat_mat_mul");
    // The largest case is seconds-scale with a naive kernel; keep the sample
    // count low so a full run stays practical.
    group.sample_size(10);
    for size in sizes {
        // n^3 inner-loop steps.
        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut a = Matrix::new(size, size);
            let mut bb = Matrix::new(size, size);
            let mut c = Matrix::new(size, size);
            // Deterministic, non-trivial data.
            for i in 0..size * size {
                a.as_mut_data()[i] = ((i % 17) as f64) - 8.0;
                bb.as_mut_data()[i] = ((i % 13) as f64) - 6.0;
            }
            b.iter(|| mat_mat_mul(&mut c, 1.0, &a, &bb, 0.0).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mat_mat_mul);
criterion_main!(benches);
