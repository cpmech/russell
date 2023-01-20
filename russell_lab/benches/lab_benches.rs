use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};
use russell_lab::{mat_eigen_sym, mat_eigen_sym_jacobi, vec_add, Matrix, Vector};

fn _bench_vec_add(c: &mut Criterion) {
    let sizes = &[1, 4, 16, 32, 64, 128];
    let mut group = c.benchmark_group("lab_vec_add");
    for size in sizes {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let u = Vector::new(size);
            let v = Vector::new(size);
            let mut w = Vector::new(size);
            b.iter(|| vec_add(&mut w, 1.0, &u, 1.0, &v));
        });
    }
    group.finish();
}

fn bench_mat_eigen_sym(c: &mut Criterion) {
    let sizes: Vec<usize> = (1..33).collect();
    let mut group = c.benchmark_group("lab_mat_eigen_sym");
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("JacobiRotation", size), size, |b, &size| {
            let mut a = Matrix::filled(size, size, 2.0);
            let mut v = Matrix::new(size, size);
            let mut l = Vector::new(size);
            b.iter(|| mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("OpenBLAS", size), size, |b, &size| {
            let mut a = Matrix::filled(size, size, 2.0);
            let mut l = Vector::new(size);
            b.iter(|| mat_eigen_sym(&mut l, &mut a).unwrap());
        });
    }
    group.finish();
}

// criterion_group!(benches, bench_vec_add, bench_mat_eigen_sym);
criterion_group!(benches, bench_mat_eigen_sym);
criterion_main!(benches);
