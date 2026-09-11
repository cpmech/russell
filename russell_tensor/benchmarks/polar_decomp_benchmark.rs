//! Benchmarks comparing the speed of the polar-decomposition algorithms,
//! all invoked through the unified `polar_decomp_mx` dispatcher:
//!
//! * `PolarAlgo::Iterative` — Brannon's iterative fixed-point (3×3)
//! * `PolarAlgo::Quaternion` — Higham & Noferini quaternion-based, direct (3×3)
//! * `PolarAlgo::Eigen` — classic: eigenvalues of C = Fᵀ F (3×3)
//! * `PolarAlgo::SVD` — classic: singular value decomposition (3×3)
//!
//! Two benchmark groups:
//!
//! 1. `polar_rotation_general_{case}` — all algorithms for mildly-, well-,
//!    moderately-, and ill-conditioned 3×3 matrices.
//! 2. `polar_rotation_in_plane` — all algorithms for an in-plane matrix.
//!
//! Notes:
//!
//! * Every algorithm is benchmarked through `polar_decomp_mx`, which computes
//!   the rotation `R` and the right stretch `U` together.
//! * `PolarAlgo::Eigen` squares the condition number (via `C = Fᵀ F`), so it
//!   fails for very ill-conditioned `F` (`cond(F) ≳ 1e8`); it is not benchmarked
//!   for the ill-conditioned case.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::{PolarAlgo, Tensor2, polar_decomp_mx};

/// Well-conditioned matrix (example 03, McGinty; κ ≈ 4)
const WELL_CONDITIONED: [[f64; 3]; 3] = [
    [1.0, 0.495, 0.5],     // 1
    [-0.333, 1.0, -0.247], // 2
    [0.959, 0.0, 1.5],     // 3
];

/// In-plane matrix (example 01, Brannon; 60° rotation about E3, κ ≈ 6)
const IN_PLANE: [[f64; 3]; 3] = [
    [0.61784609690826542, -0.70889727457341833, 0.0], // 1
    [0.59014083110323967, 0.13215390309173483, 0.0],  // 2
    [0.0, 0.0, 3.0],                                  // 3
];

/// Higham & Noferini test (5.2) for a given scale factor y; κ ≈ 1/y.
fn case52(y: f64) -> [[f64; 3]; 3] {
    [
        [
            (720.0 * y - 25.0) / 1275.0,
            (-650.0 * y + 300.0) / 1275.0,
            (710.0 * y + 300.0) / 1275.0,
        ],
        [
            (396.0 * y + 70.0) / 1275.0,
            (-145.0 * y - 840.0) / 1275.0,
            (178.0 * y - 840.0) / 1275.0,
        ],
        [
            (972.0 * y - 10.0) / 1275.0,
            (610.0 * y + 120.0) / 1275.0,
            (-529.0 * y + 120.0) / 1275.0,
        ],
    ]
}

/// Mildly distorted (solid-mechanics-like) matrix: a 0.5 rad rotation times a stretch
/// with principal stretches `{1.0, 1.05, 1.1}` (distortion `x = 1.1`)
fn mild() -> [[f64; 3]; 3] {
    let (s, c) = 0.5f64.sin_cos();
    [
        [c, -s * 1.05, 0.0], // 1
        [s, c * 1.05, 0.0],  // 2
        [0.0, 0.0, 1.1],     // 3
    ]
}

/// Benchmarks all algorithms for a given input matrix
///
/// The `with_eigen` flag controls whether the Eigen algorithm is benchmarked;
/// it fails for very ill-conditioned matrices (`cond(F) ≳ 1e8`).
fn bench_general(crit: &mut Criterion, name: &str, aa: &[[f64; 3]; 3], with_eigen: bool) {
    let mut group = crit.benchmark_group(format!("polar_rotation_general_{}", name));

    // Iterative: Brannon
    group.bench_with_input(BenchmarkId::new("iterative", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(aa).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Iterative, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    // Quaternion: Higham & Noferini
    group.bench_with_input(BenchmarkId::new("quaternion", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(aa).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Quaternion, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    // Eigen (classic: eigenvalues of C = Fᵀ F)
    if with_eigen {
        group.bench_with_input(BenchmarkId::new("eigen", ""), &(), |b, _| {
            let ff = Tensor2::<9>::from_std_matrix(aa).unwrap();
            let mut rr = Tensor2::<9>::new();
            let mut uu = Tensor2::<6>::new();
            b.iter(|| {
                polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Eigen, &ff).unwrap();
                std::hint::black_box((&rr, &uu));
            });
        });
    }

    // SVD (classic: singular value decomposition)
    group.bench_with_input(BenchmarkId::new("svd", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(aa).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::SVD, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    group.finish();
}

/// Benchmarks all algorithms for an in-plane matrix
fn bench_in_plane(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("polar_rotation_in_plane");

    // Iterative: Brannon (3×3)
    group.bench_with_input(BenchmarkId::new("iterative", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(&IN_PLANE).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Iterative, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    // Quaternion: Higham & Noferini
    group.bench_with_input(BenchmarkId::new("quaternion", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(&IN_PLANE).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Quaternion, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    // Eigen (classic)
    group.bench_with_input(BenchmarkId::new("eigen", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(&IN_PLANE).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::Eigen, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    // SVD (classic)
    group.bench_with_input(BenchmarkId::new("svd", ""), &(), |b, _| {
        let ff = Tensor2::<9>::from_std_matrix(&IN_PLANE).unwrap();
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        b.iter(|| {
            polar_decomp_mx(&mut rr, &mut uu, None, PolarAlgo::SVD, &ff).unwrap();
            std::hint::black_box((&rr, &uu));
        });
    });

    group.finish();
}

fn bench_mild(crit: &mut Criterion) {
    bench_general(crit, "mild", &mild(), true);
}

fn bench_well_conditioned(crit: &mut Criterion) {
    bench_general(crit, "well_conditioned", &WELL_CONDITIONED, true);
}

fn bench_moderate_conditioned(crit: &mut Criterion) {
    bench_general(crit, "moderate_conditioned", &case52(1e-3), true);
}

fn bench_ill_conditioned(crit: &mut Criterion) {
    bench_general(crit, "ill_conditioned", &case52(1e-8), false);
}

criterion_group!(
    benches,
    bench_mild,
    bench_well_conditioned,
    bench_moderate_conditioned,
    bench_ill_conditioned,
    bench_in_plane
);
criterion_main!(benches);
