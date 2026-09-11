//! Compares the accuracy of the deviatoric invariants `J2` and `J3` computed with
//! the Habera-Zilian and Harari-Albocher methods.
//!
//! The benchmark follows the `eig3x3` library's invariants benchmark: symmetric
//! matrices are built as `A = U ⋅ diag(d) ⋅ Uᵀ` with the orthogonal transformation
//! `U_symm` and the diagonal cases `d(δ)` from the Habera-Zilian test suite.
//!
//! Four variants are compared:
//!
//! * `HZ_api` — [Tensor2::invariant_jj2_hz] / [Tensor2::invariant_jj3_hz]
//! * `HZ_raw` — the Habera-Zilian formulas applied directly to the 3×3 matrix
//! * `HA` — [Tensor2::invariant_jj2] / [Tensor2::invariant_jj3]
//! * `naive` — the monomial deviatoric formulas
//!
//! The reference values are computed with double-double (f64×2) arithmetic and are
//! therefore accurate to about 30 digits (validated against exact rational
//! arithmetic). For each method the maximum absolute and relative errors over the
//! cases are reported as a function of `δ`.
//!
//! Notes:
//!
//! * The `HZ_api` path converts the tensor to standard components and back, which
//!   costs a few extra ulp; the `HZ_raw` column isolates the formula itself.
//! * As `δ → 0` the reference `J3` approaches rounding noise, so its relative error
//!   is limited by the conditioning of the problem (as in the `eig3x3` benchmark);
//!   the absolute error is the more meaningful metric there.

use russell_tensor::Tensor2;

/// Number of variants compared
const NV: usize = 4;

/// Variant labels
const LABELS: [&str; NV] = ["HZ_api", "HZ_raw", "HA", "naive"];

fn main() {
    let u = u_symm();
    let deltas = [
        1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 500.0,
    ];

    println!("Symmetric tensors, A = U_symm ⋅ diag(d) ⋅ U_symmᵀ");
    println!("(reference: double-double, ~30 digits)");

    // Accumulate the errors: rows = deltas, columns = variants
    let mut j2_abs: Vec<[f64; NV]> = vec![[0.0; NV]; deltas.len()];
    let mut j2_rel: Vec<[f64; NV]> = vec![[0.0; NV]; deltas.len()];
    let mut j3_abs: Vec<[f64; NV]> = vec![[0.0; NV]; deltas.len()];
    let mut j3_rel: Vec<[f64; NV]> = vec![[0.0; NV]; deltas.len()];

    for (k, &delta) in deltas.iter().enumerate() {
        for name in CASES {
            let d = case_diagonal(name, delta);
            let a = build_a(&u, &d);
            let tt = Tensor2::<6>::from_std_matrix(&a).unwrap();
            let (r2, r3) = ref_invariants(&a);
            let variants = [hz_api(&tt), hz_raw(&a), ha(&tt), naive(&a)];
            for (m, (w2, w3)) in variants.iter().enumerate() {
                let e2 = f64::abs(w2 - r2);
                let e3 = f64::abs(w3 - r3);
                j2_abs[k][m] = j2_abs[k][m].max(e2);
                j2_rel[k][m] = j2_rel[k][m].max(e2 / f64::abs(r2).max(1e-300));
                j3_abs[k][m] = j3_abs[k][m].max(e3);
                j3_rel[k][m] = j3_rel[k][m].max(e3 / f64::abs(r3).max(1e-300));
            }
        }
    }

    print_j2(&deltas, &j2_abs, &j2_rel);
    print_j3(&deltas, &j3_abs, &j3_rel);
}

fn print_j2(deltas: &[f64], abs: &[[f64; NV]], rel: &[[f64; NV]]) {
    println!("\n=== J2: max error over the cases ===");
    print_header();
    for (k, &delta) in deltas.iter().enumerate() {
        print_row(delta, &abs[k], &rel[k]);
    }
}

fn print_j3(deltas: &[f64], abs: &[[f64; NV]], rel: &[[f64; NV]]) {
    println!("\n=== J3: max error over the cases ===");
    print_header();
    for (k, &delta) in deltas.iter().enumerate() {
        print_row(delta, &abs[k], &rel[k]);
    }
}

fn print_header() {
    let mut line = format!("{:>8}", "δ");
    for label in LABELS {
        line.push_str(&format!(" {:>10}", label));
    }
    line.push_str("  |");
    for label in LABELS {
        line.push_str(&format!(" {:>10}", label));
    }
    println!("{}", line);
}

fn print_row(delta: f64, abs: &[f64; NV], rel: &[f64; NV]) {
    let mut line = format!("{:>8.0e}", delta);
    for a in abs {
        line.push_str(&format!(" {:>10.2e}", a));
    }
    line.push_str("  |");
    for r in rel {
        line.push_str(&format!(" {:>10.2e}", r));
    }
    println!("{}", line);
}

/// Diagonal cases `d(δ)` from the Habera-Zilian test suite
const CASES: [&str; 11] = [
    "single",
    "single_lim_J3",
    "single_lim_disc_t",
    "single_lim_disc_n",
    "single_lim_J3J2",
    "single_J3",
    "single_J3_lim_J2",
    "double",
    "double_lim_J3J2",
    "triple_J3",
    "d3",
];

fn case_diagonal(name: &str, delta: f64) -> [f64; 3] {
    let a = 1.0;
    match name {
        "single" => [-a / 4.0, a / 4.0, (2.0 + 2.0 * delta) * a / 4.0],
        "single_lim_J3" => [(-1.0 - delta) * a / 4.0, 0.0, (1.0 + 2.0 * delta) * a / 4.0],
        "single_lim_disc_t" => [-a, a, (1.0 + delta) * a],
        "single_lim_disc_n" => [0.0, (2.0 - delta) * a / 2.0, (2.0 + delta) * a / 2.0],
        "single_lim_J3J2" => [(1.0 - delta) * a, a, (1.0 + 2.0 * delta) * a],
        "single_J3" => [(-1.0 - delta) * a / 2.0, 0.0, (1.0 + delta) * a / 2.0],
        "single_J3_lim_J2" => [(1.0 - delta) * a, a, (1.0 + delta) * a],
        "double" => [(-1.0 - delta) * a, a, a],
        "double_lim_J3J2" => [a, a, (1.0 + delta) * a],
        "triple_J3" => [-delta, 0.0, delta],
        "d3" => [0.0, a, (2.0 + delta) * a],
        _ => panic!("unknown case: {}", name),
    }
}

/// Orthogonal transformation matrix from the Habera-Zilian test suite
fn u_symm() -> [[f64; 3]; 3] {
    let r2 = f64::sqrt(2.0);
    [[1.0 / r2, -0.5, 0.5], [1.0 / r2, 0.5, -0.5], [0.0, 1.0 / r2, 1.0 / r2]]
}

/// Builds the symmetric matrix `A = U ⋅ diag(d) ⋅ Uᵀ` (and symmetrizes it)
fn build_a(u: &[[f64; 3]; 3], d: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                a[i][j] += u[i][k] * d[k] * u[j][k];
            }
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let m = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = m;
            a[j][i] = m;
        }
    }
    a
}

/// Habera-Zilian symmetric invariants using the `Tensor2` wrappers (round-trips
/// through standard components)
fn hz_api(aa: &Tensor2<6>) -> (f64, f64) {
    (aa.invariant_jj2_hz(), aa.invariant_jj3_hz())
}

/// Habera-Zilian symmetric invariants applied directly to the 3×3 matrix
fn hz_raw(a: &[[f64; 3]; 3]) -> (f64, f64) {
    let (a00, a01, a02) = (a[0][0], a[0][1], a[0][2]);
    let (a11, a12, a22) = (a[1][1], a[1][2], a[2][2]);
    let d0 = a00 - a11;
    let d1 = a00 - a22;
    let d2 = a11 - a22;
    let off_diag = a01 * a01 + a02 * a02 + a12 * a12;
    let diag = (d0 * d0 + d1 * d1 + d2 * d2) / 6.0;
    let j2 = off_diag + diag;
    let t1 = d1 + d2;
    let t2 = d0 - d2;
    let t3 = -d0 - d1;
    let off = 2.0 * a01 * a12 * a02;
    let mixed = (a01 * a01 * t1 + a02 * a02 * t2 + a12 * a12 * t3) / 3.0;
    let dg = (t1 * t2 * t3) / 27.0;
    (j2, off + mixed - dg)
}

/// Harari-Albocher symmetric invariants
fn ha(aa: &Tensor2<6>) -> (f64, f64) {
    (aa.invariant_jj2(), aa.invariant_jj3())
}

/// Naive invariants `(J2, J3)` based on the monomial formulas
fn naive(a: &[[f64; 3]; 3]) -> (f64, f64) {
    let m = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let s = [
        [a[0][0] - m, a[0][1], a[0][2]],
        [a[1][0], a[1][1] - m, a[1][2]],
        [a[2][0], a[2][1], a[2][2] - m],
    ];
    let mut j2 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            j2 += 0.5 * s[i][j] * s[j][i];
        }
    }
    let j3 = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1]) - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
        + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);
    (j2, j3)
}

// --- double-double reference -------------------------------------------------------------------

/// Minimal double-double number (f64×2) giving about 30 accurate digits
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    #[inline]
    fn from(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    #[inline]
    fn add(self, o: Dd) -> Dd {
        // Knuth's TwoSum
        let s = self.hi + o.hi;
        let v = s - self.hi;
        let e = (self.hi - (s - v)) + (o.hi - v);
        let e = e + self.lo + o.lo;
        let s2 = s + e;
        let e2 = e - (s2 - s);
        Dd { hi: s2, lo: e2 }
    }

    #[inline]
    fn sub(self, o: Dd) -> Dd {
        self.add(Dd { hi: -o.hi, lo: -o.lo })
    }

    #[inline]
    fn mul(self, o: Dd) -> Dd {
        // Dekker's TwoProduct (using fma)
        let p = self.hi * o.hi;
        let e = self.hi.mul_add(o.hi, -p);
        let e = e + self.hi * o.lo + self.lo * o.hi;
        let s = p + e;
        let e2 = e - (s - p);
        Dd { hi: s, lo: e2 }
    }

    #[inline]
    fn div(self, o: Dd) -> Dd {
        let q1 = self.hi / o.hi;
        let r = self.sub(o.mul(Dd::from(q1)));
        let q2 = r.hi / o.hi;
        let r = r.sub(o.mul(Dd::from(q2)));
        let q3 = r.hi / o.hi;
        Dd::from(q1).add(Dd::from(q2)).add(Dd::from(q3))
    }
}

/// High-precision reference invariants `(J2, J3)` of a symmetric 3×3 matrix
fn ref_invariants(a: &[[f64; 3]; 3]) -> (f64, f64) {
    let m = [
        [Dd::from(a[0][0]), Dd::from(a[0][1]), Dd::from(a[0][2])],
        [Dd::from(a[1][0]), Dd::from(a[1][1]), Dd::from(a[1][2])],
        [Dd::from(a[2][0]), Dd::from(a[2][1]), Dd::from(a[2][2])],
    ];
    let tr = m[0][0].add(m[1][1]).add(m[2][2]);
    let iso = tr.div(Dd::from(3.0));
    let s = [
        [m[0][0].sub(iso), m[0][1], m[0][2]],
        [m[1][0], m[1][1].sub(iso), m[1][2]],
        [m[2][0], m[2][1], m[2][2].sub(iso)],
    ];
    let mut j2 = Dd::ZERO;
    for i in 0..3 {
        for j in 0..3 {
            j2 = j2.add(s[i][j].mul(s[j][i]));
        }
    }
    let j2 = j2.div(Dd::from(2.0));
    let j3 = s[0][0]
        .mul(s[1][1].mul(s[2][2]).sub(s[1][2].mul(s[2][1])))
        .sub(s[0][1].mul(s[1][0].mul(s[2][2]).sub(s[1][2].mul(s[2][0]))))
        .add(s[0][2].mul(s[1][0].mul(s[2][1]).sub(s[1][1].mul(s[2][0]))));
    (j2.hi, j3.hi)
}
