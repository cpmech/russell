//! Benchmarks comparing the speed of the four eigenvalue methods available in
//! `EigenValuesT2::calculate_mx`:
//!
//! * `EigenValMethod::AnalyticalHZ` — stable closed-form (Habera & Zilian 2025)
//! * `EigenValMethod::AnalyticalHA22` — Box-1 discriminant (Harari & Albocher 2022)
//! * `EigenValMethod::AnalyticalHA23` — seven-square discriminant (Harari & Albocher 2023)
//! * `EigenValMethod::Iterative` — iterative Jacobi rotations
//!
//! Two symmetric input tensors are used:
//!
//! 1. `distinct` — well-separated eigenvalues
//! 2. `coalescent` — two nearly equal eigenvalues (the tough case for the
//!    discriminant-based methods)
//!
//! Only the eigenvalues are computed (via `calculate_mx`); the eigenprojectors
//! are not, since that is the common factor across the four methods.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::{EigenValMethod, EigenValuesT2, Tensor2};

/// Symmetric tensor with well-separated eigenvalues
const DISTINCT: [[f64; 3]; 3] = [
    [1.0, 0.1, 0.2], // 1
    [0.1, 2.0, 0.3], // 2
    [0.2, 0.3, 3.0], // 3
];

/// Symmetric tensor with two nearly equal eigenvalues
const COALESCENT: [[f64; 3]; 3] = [
    [1.0, 0.1, 0.2],        // 1
    [0.1, 1.0 + 1e-9, 0.3], // 2
    [0.2, 0.3, 3.0],        // 3
];

/// Eigenvalue methods, in benchmark order
const METHODS: [(&str, EigenValMethod); 4] = [
    ("analytical_hz", EigenValMethod::AnalyticalHZ),
    ("analytical_ha22", EigenValMethod::AnalyticalHA22),
    ("analytical_ha23", EigenValMethod::AnalyticalHA23),
    ("iterative", EigenValMethod::Iterative),
];

/// Benchmarks the four eigenvalue methods for a given input tensor
fn bench_eigenvalues(crit: &mut Criterion, name: &str, matrix: &[[f64; 3]; 3]) {
    let mut group = crit.benchmark_group(format!("eigenvalues_{}", name));

    for (label, method) in METHODS {
        group.bench_with_input(BenchmarkId::new(label, ""), &(), |b, _| {
            let aa = Tensor2::<6>::from_std_matrix(matrix).unwrap();
            let mut calc = EigenValuesT2::new();
            let mut ll = [0.0; 3];
            b.iter(|| {
                calc.calculate_mx(&mut ll, &aa, method).unwrap();
                std::hint::black_box(&ll);
            });
        });
    }

    group.finish();
}

fn bench_distinct(crit: &mut Criterion) {
    bench_eigenvalues(crit, "distinct", &DISTINCT);
}

fn bench_coalescent(crit: &mut Criterion) {
    bench_eigenvalues(crit, "coalescent", &COALESCENT);
}

criterion_group!(benches, bench_distinct, bench_coalescent);
criterion_main!(benches);
