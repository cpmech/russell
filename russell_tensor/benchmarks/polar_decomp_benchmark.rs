//! Benchmarks comparing the speed of the two polar-decomposition algorithms:
//!
//! * **Brannon** (iterative fixed-point) — `polar_rotation_brannon`
//! * **Higham & Noferini** (quaternion, direct) — `polar_quaternion_higham`
//!
//! Three cases are benchmarked, spanning the condition-number range:
//!
//! * `well_conditioned`   — κ ≈ 4 (example 03, McGinty)
//! * `moderate_conditioned` — κ ≈ 6·10² (Higham test 5.2 with y = 1e-3)
//! * `ill_conditioned`    — κ ≈ 6·10⁷ (Higham test 5.2 with y = 1e-8)
//!
//! Brannon's algorithm iterates more as the condition number grows, whereas
//! Higham's is direct (fixed work), so their relative speed is expected to
//! depend on the conditioning.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use russell_tensor::{polar_decomp, PolarAlgo, Rep, Tensor2};

/// Well-conditioned matrix (example 03, McGinty; κ ≈ 4)
const WELL_CONDITIONED: [[f64; 3]; 3] = [[1.0, 0.495, 0.5], [-0.333, 1.0, -0.247], [0.959, 0.0, 1.5]];

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

/// Benchmarks both algorithms for a given input matrix
fn bench_case(crit: &mut Criterion, name: &str, aa: &[[f64; 3]; 3]) {
    let mut group = crit.benchmark_group(format!("polar_decomp_{}", name));

    // Brannon (iterative fixed-point)
    group.bench_with_input(BenchmarkId::new("brannon", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(aa, Rep::General).unwrap();
        let mut rr = Tensor2::new(Rep::General);
        let mut uu = Tensor2::new(Rep::Symmetric);
        b.iter(|| {
            polar_decomp(&mut rr, &mut uu, None, PolarAlgo::Brannon, &ff).unwrap();
            std::hint::black_box(rr.get(0));
        });
    });

    // Higham & Noferini (quaternion, direct)
    group.bench_with_input(BenchmarkId::new("higham", ""), &(), |b, _| {
        let ff = Tensor2::from_std_matrix(aa, Rep::General).unwrap();
        let mut qq = Tensor2::new(Rep::General);
        let mut hh = Tensor2::new(Rep::Symmetric);
        b.iter(|| {
            polar_decomp(&mut qq, &mut hh, None, PolarAlgo::Higham, &ff).unwrap();
            std::hint::black_box(qq.get(0));
        });
    });

    group.finish();
}

fn bench_well_conditioned(crit: &mut Criterion) {
    bench_case(crit, "well_conditioned", &WELL_CONDITIONED);
}

fn bench_moderate_conditioned(crit: &mut Criterion) {
    bench_case(crit, "moderate_conditioned", &case52(1e-3));
}

fn bench_ill_conditioned(crit: &mut Criterion) {
    bench_case(crit, "ill_conditioned", &case52(1e-8));
}

criterion_group!(
    benches,
    bench_well_conditioned,
    bench_moderate_conditioned,
    bench_ill_conditioned
);
criterion_main!(benches);
