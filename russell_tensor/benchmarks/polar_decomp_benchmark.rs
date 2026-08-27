//! Benchmarks comparing the speed of the polar-rotation algorithms:
//!
//! * `polar_rotation_brannon` — iterative fixed-point (3×3)
//! * `polar_quaternion_higham` — quaternion-based, direct (3×3)
//! * `polar_rotation_brannon2d` — closed-form, in-plane only (2×2)
//!
//! Two benchmark groups:
//!
//! 1. `polar_rotation_general_{case}` — Brannon vs Higham for well-,
//!    moderately-, and ill-conditioned 3×3 matrices.
//! 2. `polar_rotation_in_plane` — all three algorithms for an in-plane matrix.
//!
//! Note: `polar_quaternion_higham` computes the stretch `H` together with the
//! rotation `Q` (the quaternion algorithm does not separate them), whereas
//! `polar_rotation_brannon` and `polar_rotation_brannon2d` compute only `R`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use russell_tensor::{Rep, Tensor2, polar_quaternion_higham, polar_rotation_brannon, polar_rotation_brannon2d};

/// Well-conditioned matrix (example 03, McGinty; κ ≈ 4)
const WELL_CONDITIONED: [[f64; 3]; 3] = [[1.0, 0.495, 0.5], [-0.333, 1.0, -0.247], [0.959, 0.0, 1.5]];

/// In-plane matrix (example 01, Brannon; 60° rotation about E3, κ ≈ 6)
const IN_PLANE: [[f64; 3]; 3] = [
    [0.61784609690826542, -0.70889727457341833, 0.0],
    [0.59014083110323967, 0.13215390309173483, 0.0],
    [0.0, 0.0, 3.0],
];

/// Higham & Noferini test (5.2) for a given scale factor y; κ ≈ 1/(√3 y).
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

/// Benchmarks Brannon vs Higham for a given input matrix
fn bench_general(crit: &mut Criterion, name: &str, aa: &[[f64; 3]; 3]) {
    let mut group = crit.benchmark_group(format!("polar_rotation_general_{}", name));

    // Brannon (iterative fixed-point; rotation only)
    group.bench_with_input(BenchmarkId::new("brannon", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(aa, Rep::General).unwrap();
        let mut rr = Tensor2::new(Rep::General);
        b.iter(|| {
            polar_rotation_brannon(&mut rr, &ff).unwrap();
            std::hint::black_box(rr.get(0));
        });
    });

    // Higham & Noferini (quaternion, direct; also computes the stretch H)
    group.bench_with_input(BenchmarkId::new("higham", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(aa, Rep::General).unwrap();
        let mut qq = Tensor2::new(Rep::General);
        let mut hh = Tensor2::new(Rep::Symmetric);
        b.iter(|| {
            polar_quaternion_higham(&mut qq, &mut hh, &ff).unwrap();
            std::hint::black_box(qq.get(0));
        });
    });

    group.finish();
}

/// Benchmarks all three algorithms for an in-plane matrix
fn bench_in_plane(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("polar_rotation_in_plane");

    // Brannon (iterative, 3×3)
    group.bench_with_input(BenchmarkId::new("brannon", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(&IN_PLANE, Rep::General).unwrap();
        let mut rr = Tensor2::new(Rep::General);
        b.iter(|| {
            polar_rotation_brannon(&mut rr, &ff).unwrap();
            std::hint::black_box(rr.get(0));
        });
    });

    // Brannon (closed-form, 2×2)
    group.bench_with_input(BenchmarkId::new("brannon2d", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(&IN_PLANE, Rep::General).unwrap();
        let mut rr = Tensor2::new(Rep::General);
        b.iter(|| {
            polar_rotation_brannon2d(&mut rr, &ff).unwrap();
            std::hint::black_box(rr.get(0));
        });
    });

    // Higham & Noferini (quaternion, direct)
    group.bench_with_input(BenchmarkId::new("higham", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(&IN_PLANE, Rep::General).unwrap();
        let mut qq = Tensor2::new(Rep::General);
        let mut hh = Tensor2::new(Rep::Symmetric);
        b.iter(|| {
            polar_quaternion_higham(&mut qq, &mut hh, &ff).unwrap();
            std::hint::black_box(qq.get(0));
        });
    });

    group.finish();
}

fn bench_well_conditioned(crit: &mut Criterion) {
    bench_general(crit, "well_conditioned", &WELL_CONDITIONED);
}

fn bench_moderate_conditioned(crit: &mut Criterion) {
    bench_general(crit, "moderate_conditioned", &case52(1e-3));
}

fn bench_ill_conditioned(crit: &mut Criterion) {
    bench_general(crit, "ill_conditioned", &case52(1e-8));
}

criterion_group!(
    benches,
    bench_well_conditioned,
    bench_moderate_conditioned,
    bench_ill_conditioned,
    bench_in_plane
);
criterion_main!(benches);
