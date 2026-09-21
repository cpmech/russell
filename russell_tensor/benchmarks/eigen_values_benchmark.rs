//! Benchmarks for the eigen-evaluation stages of `russell_tensor`:
//!
//! 1. `eigenvalues_{case}` — `EigenValuesT2::calculate_mx` (values only)
//! 2. `eigen_projectors_{case}` — `EigenProjsT2::calculate_mx` (values + Sylvester projectors)
//! 3. `eigen_proj_derivs_distinct` — the `EigenProjDerivsT2` derivative algorithms
//!
//! The eigenvalue methods are:
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
//! The eigenprojector derivatives are only defined for distinct eigenvalues, so they
//! are benchmarked for the `distinct` input only, comparing the two algorithms of
//! `EigenProjDerivsT2` (both using the Habera-Zilian eigenvalues):
//!
//! * `char_poly` — `calc_with_char_poly` (Panteghini's characteristic polynomial)
//! * `with_inv` — `calc_with_inv` (Miehe's inverse-based approach)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::{EigenProjDerivsT2, EigenProjsT2, EigenValMethod, EigenValuesT2, Tensor2, Tensor4};

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

/// Benchmarks `EigenValuesT2::calculate_mx` for the four methods
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

/// Benchmarks `EigenProjsT2::calculate_mx` (eigenvalues + Sylvester projectors)
fn bench_projectors(crit: &mut Criterion, name: &str, matrix: &[[f64; 3]; 3]) {
    let mut group = crit.benchmark_group(format!("eigen_projectors_{}", name));

    for (label, method) in METHODS {
        group.bench_with_input(BenchmarkId::new(label, ""), &(), |b, _| {
            let aa = Tensor2::<6>::from_std_matrix(matrix).unwrap();
            let mut calc = EigenProjsT2::new();
            let mut ll = [0.0; 3];
            let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
            b.iter(|| {
                calc.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();
                std::hint::black_box((&ll, &projs));
            });
        });
    }

    group.finish();
}

/// Benchmarks the two `EigenProjDerivsT2` derivative algorithms (distinct input)
fn bench_proj_derivs(crit: &mut Criterion, matrix: &[[f64; 3]; 3]) {
    let mut group = crit.benchmark_group("eigen_proj_derivs_distinct");

    // characteristic polynomial (Panteghini)
    group.bench_with_input(BenchmarkId::new("char_poly", ""), &(), |b, _| {
        let aa = Tensor2::<6>::from_std_matrix(matrix).unwrap();
        let mut calc = EigenProjDerivsT2::new();
        let mut ll = [0.0; 3];
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let mut dpp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        b.iter(|| {
            calc.calc_with_char_poly(&mut ll, &mut projs, &mut dpp, &aa, EigenValMethod::AnalyticalHZ)
                .unwrap();
            std::hint::black_box((&ll, &projs, &dpp));
        });
    });

    // inverse-based (Miehe)
    group.bench_with_input(BenchmarkId::new("with_inv", ""), &(), |b, _| {
        let aa = Tensor2::<6>::from_std_matrix(matrix).unwrap();
        let mut calc = EigenProjDerivsT2::new();
        let mut ll = [0.0; 3];
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let mut dpp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        b.iter(|| {
            calc.calc_with_inv(&mut ll, &mut projs, &mut dpp, &aa, EigenValMethod::AnalyticalHZ)
                .unwrap();
            std::hint::black_box((&ll, &projs, &dpp));
        });
    });

    group.finish();
}

fn bench_distinct(crit: &mut Criterion) {
    bench_eigenvalues(crit, "distinct", &DISTINCT);
}

fn bench_coalescent(crit: &mut Criterion) {
    bench_eigenvalues(crit, "coalescent", &COALESCENT);
}

fn bench_projectors_distinct(crit: &mut Criterion) {
    bench_projectors(crit, "distinct", &DISTINCT);
}

fn bench_projectors_coalescent(crit: &mut Criterion) {
    bench_projectors(crit, "coalescent", &COALESCENT);
}

fn bench_proj_derivs_distinct(crit: &mut Criterion) {
    bench_proj_derivs(crit, &DISTINCT);
}

criterion_group!(
    benches,
    bench_distinct,
    bench_coalescent,
    bench_projectors_distinct,
    bench_projectors_coalescent,
    bench_proj_derivs_distinct
);
criterion_main!(benches);
