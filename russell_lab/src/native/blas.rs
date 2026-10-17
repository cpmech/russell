//! Native BLAS: the level-1, -2 and -3 operations the LAPACK routines need.
//!
//! Only what `russell_lab` actually calls is implemented, which is a small
//! subset of BLAS. Each function names its BLAS equivalent.
//!
//! # Column-major indexing
//!
//! Every matrix here is column-major with `nrow` rows, so element `(i, j)` is at
//! `data[i + j * nrow]`. Getting that detail wrong is the classic source of a
//! plausible-looking but wrong kernel, so each routine states its indexing in a
//! comment and the tests exercise non-square shapes where a transposed index
//! cannot coincidentally be right.

/// `y := alpha * x + y` (daxpy), where `y` and `x` are the same length.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "daxpy: vectors must have the same length");
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

/// `x := alpha * x` (dscal).
pub fn dscal(alpha: f64, x: &mut [f64]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

/// `y := x` (dcopy).
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn dcopy(x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "dcopy: vectors must have the same length");
    y.copy_from_slice(x);
}

/// `x . y` (ddot).
///
/// BLAS's `ddot` returns 0 for mismatched lengths by reading the shorter one;
/// this does the same rather than panicking, since that is the documented
/// behaviour callers may rely on.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Euclidean norm `sqrt(x . x)` (dnrm2).
///
/// Scaled to avoid overflow and underflow for very large or very small vectors,
/// as the reference implementation is.
pub fn dnrm2(x: &[f64]) -> f64 {
    // Blue's scaling: find the largest magnitude first, then accumulate in units
    // of it, so a vector of 1e200s does not overflow in the squares.
    let mut scale = 0.0f64;
    for &v in x {
        let a = v.abs();
        if a > scale {
            scale = a;
        }
    }
    if scale == 0.0 || !scale.is_finite() {
        // All zeros, or a non-finite entry (which propagates as it should).
        return x.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
    let inv = 1.0 / scale;
    let ss: f64 = x.iter().map(|v| (v * inv) * (v * inv)).sum();
    scale * ss.sqrt()
}

/// Index of the first entry with the largest absolute value (idamax).
///
/// 0-based, unlike BLAS's `idamax`, which is 1-based. The *first* maximum wins,
/// which matters when equal magnitudes appear and is what BLAS specifies. An
/// empty slice yields 0.
pub fn idamax(x: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_abs = f64::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        let a = v.abs();
        if a > best_abs {
            best_abs = a;
            best = i;
        }
    }
    best
}

/// Sum of absolute values `|x|_1` (dasum).
pub fn dasum(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

/// `y := alpha * A * x + beta * y` (dgemv, trans = 'N').
///
/// `a` is `m x n` column-major with `m` rows.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemv(alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dgemv: a must be m*n");
    assert_eq!(x.len(), n, "dgemv: x must have n entries");
    assert_eq!(y.len(), m, "dgemv: y must have m entries");

    // Scale y first so it can be read and written in place.
    if beta == 0.0 {
        for v in y.iter_mut() {
            *v = 0.0;
        }
    } else if beta != 1.0 {
        for v in y.iter_mut() {
            *v *= beta;
        }
    }

    for j in 0..n {
        let xj = x[j];
        if xj == 0.0 {
            continue;
        }
        let c = alpha * xj;
        // Column j occupies a[j*m .. (j+1)*m].
        let col = &a[j * m..(j + 1) * m];
        for i in 0..m {
            y[i] += c * col[i];
        }
    }
}

/// `y := alpha * A^T * x + beta * y` (dgemv, trans = 'T').
///
/// `a` is `m x n` column-major with `m` rows, so `A^T` is `n x m`: `x` has `m`
/// entries and `y` has `n`.
///
/// Implemented directly rather than through [`dgemv`], because transposing into a
/// temporary would cost a full copy of the matrix for what is a change of loop
/// order.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemv_trans(alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dgemv_trans: a must be m*n");
    assert_eq!(x.len(), m, "dgemv_trans: x must have m entries");
    assert_eq!(y.len(), n, "dgemv_trans: y must have n entries");

    for j in 0..n {
        // Row j of A^T is column j of A, so this is a dot product per column.
        let col = &a[j * m..(j + 1) * m];
        let mut sum = 0.0;
        for i in 0..m {
            sum += col[i] * x[i];
        }
        y[j] = if beta == 0.0 {
            alpha * sum
        } else {
            alpha * sum + beta * y[j]
        };
    }
}

/// `A := alpha * x * y^T + A` (dger), with `A` `m x n` column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dger(alpha: f64, x: &[f64], y: &[f64], a: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dger: a must be m*n");
    assert_eq!(x.len(), m, "dger: x must have m entries");
    assert_eq!(y.len(), n, "dger: y must have n entries");

    for j in 0..n {
        let c = alpha * y[j];
        if c == 0.0 {
            continue;
        }
        let col = &mut a[j * m..(j + 1) * m];
        for i in 0..m {
            col[i] += c * x[i];
        }
    }
}

/// `C := alpha * A * B + beta * C` (dgemm, both trans = 'N').
///
/// `a` is `m x k`, `b` is `k x n`, `c` is `m x n`, all column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemm(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "dgemm: a must be m*k");
    assert_eq!(b.len(), k * n, "dgemm: b must be k*n");
    assert_eq!(c.len(), m * n, "dgemm: c must be m*n");

    if m == 0 || n == 0 {
        return;
    }

    // Scale C once, up front. The micro-kernel then only accumulates, so `beta`
    // does not have to reach inside the hot loop.
    match beta {
        0.0 => c.fill(0.0),
        1.0 => {}
        _ => {
            for x in c.iter_mut() {
                *x *= beta;
            }
        }
    }
    if k == 0 || alpha == 0.0 {
        return;
    }

    // Block sizes. The packed panels are the only large allocation; everything
    // inside them is reused across the whole inner loop, which is the point --
    // the naive kernel re-reads `a` and `b` from memory for every element of
    // `c`, so it is bound by cache misses rather than by arithmetic.
    const MC: usize = 128;
    const KC: usize = 256;
    const NC: usize = 256;

    let mut ap = vec![0.0f64; MC * KC];
    let mut bp = vec![0.0f64; KC * NC];

    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    let use_avx = has_avx2_fma();
    #[cfg(not(all(target_arch = "x86_64", not(target_arch = "wasm32"))))]
    let use_avx = false;

    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);

            // Pack B's panel column-major: bp[p + j*kc] = b[pc+p][jc+j].
            // Contiguous in p, which is the inner-loop direction.
            for j in 0..nc {
                let b_col = &b[(pc) + (jc + j) * k..(pc + kc) + (jc + j) * k];
                let dst = &mut bp[j * kc..j * kc + kc];
                dst.copy_from_slice(b_col);
            }

            for ic in (0..m).step_by(MC) {
                let mc = MC.min(m - ic);
                // Pack A's panel: ap[i + p*MC] = a[ic+i][pc+p], contiguous in i
                // so the micro-kernel's vector load is a straight 32-byte read.
                // `alpha` is folded in here rather than in the kernels: the
                // micro-kernel keeps C in registers and accumulates, so applying
                // a scale per element there would defeat the point. Scaling the
                // packed panel costs one multiply per element of A per block and
                // leaves both kernels as pure accumulators.
                for p in 0..kc {
                    let a_col = &a[(ic) + (pc + p) * m..(ic + mc) + (pc + p) * m];
                    let dst = &mut ap[p * MC..p * MC + mc];
                    if alpha == 1.0 {
                        dst.copy_from_slice(a_col);
                    } else {
                        for (d, s) in dst.iter_mut().zip(a_col) {
                            *d = alpha * s;
                        }
                    }
                }
                gemm_panel(&ap, &bp, &mut c[ic + jc * m..], m, mc, nc, kc, MC, kc, use_avx);
            }
        }
    }
}

/// Whether this CPU has AVX2 and FMA, checked once.
///
/// The micro-kernel is compiled with `#[target_feature(enable = "avx2,fma")]`,
/// so calling it on a CPU without them is undefined behaviour. A build with
/// `-C target-feature=+avx2` would make this redundant, but the crate cannot
/// assume that -- it is built for whatever target the consumer chooses, and
/// x86-64 baseline does not include AVX2.
#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
fn has_avx2_fma() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
}

/// Accumulates one packed panel product into `c`, whose element `(i, j)` is at
/// `c[i + j * ldc]` with the panel starting at row `ic` and column `jc`.
///
/// Split out so the safe wrapper above stays readable and the `unsafe` block is
/// confined to one place.
#[allow(clippy::too_many_arguments)]
fn gemm_panel(
    ap: &[f64],
    bp: &[f64],
    c: &mut [f64],
    ldc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    lda: usize,
    ldb: usize,
    use_avx: bool,
) {
    // The panel's leading dimensions are the packed strides, so the micro-kernel
    // reads `ap[i + p*lda]` and `bp[p + j*ldb]`.
    let mut j = 0;
    while j < nc {
        let mut i = 0;
        if j + 4 <= nc && use_avx {
            #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
            {
                while i + 4 <= mc {
                    // Safety: interior indices are in bounds by construction
                    // (i + 4 <= mc <= lda rows, j + 4 <= nc). The loads are
                    // unaligned: `ap` starts at i, which is a multiple of 4 but
                    // not necessarily 32-byte aligned.
                    unsafe {
                        gemm_micro_4x4(
                            ap.as_ptr().add(i),
                            lda,
                            bp.as_ptr().add(j * ldb),
                            ldb,
                            c.as_mut_ptr().add(i + j * ldc),
                            ldc,
                            kc,
                        );
                    }
                    i += 4;
                }
            }
            // Remainder rows of this 4-wide column strip.
            while i < mc {
                for jj in j..j + 4 {
                    let mut sum = 0.0;
                    for p in 0..kc {
                        sum += ap[i + p * lda] * bp[p + jj * ldb];
                    }
                    c[i + jj * ldc] += sum;
                }
                i += 1;
            }
            j += 4;
        } else {
            // Fewer than four columns left: no 4x4 block fits.
            for jj in j..nc {
                for ii in 0..mc {
                    let mut sum = 0.0;
                    for p in 0..kc {
                        sum += ap[ii + p * lda] * bp[p + jj * ldb];
                    }
                    c[ii + jj * ldc] += sum;
                }
            }
            j = nc;
        }
    }
}

/// The 4x4 AVX2+FMA micro-kernel: four rows by four columns of the packed
/// panels, accumulated in four vector registers.
///
/// Register pressure is the reason for 4x4. Four `c` accumulators plus one `a`
/// vector is five of the sixteen YMM registers, so the loop does not spill; the
/// `b` scalars are folded in as memory operands. Each iteration of `p` performs
/// one 32-byte load of `a` and four broadcasts, against 32 flops -- so this is
/// arithmetic-bound, unlike the naive kernel.
///
/// # Safety
///
/// Requires AVX2 and FMA. `ap` must have `4 + (kc-1)*lda` elements in reach,
/// `bp` must have `4 + (kc-1)*ldb`, and `c` must have `4 + 3*ldc` from the
/// pointer, with `kc >= 1`.
#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_micro_4x4(ap: *const f64, lda: usize, bp: *const f64, ldb: usize, c: *mut f64, ldc: usize, kc: usize) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_broadcast_sd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    // The whole body is unsafe: raw-pointer reads on the packed panels and
    // the AVX2/FMA intrinsics. Written as one block so the pre-2024 `unsafe
    // fn` allowance does not hide which operations are actually unsafe.
    unsafe {
        let mut c0 = _mm256_setzero_pd();
        let mut c1 = _mm256_setzero_pd();
        let mut c2 = _mm256_setzero_pd();
        let mut c3 = _mm256_setzero_pd();

        // The four columns of B this block advances. `ldb` is the packed column
        // stride, so column `j` occupies `bp[j*ldb .. j*ldb + kc]` and its p-th
        // element is `bp[j*ldb + p]`. Getting this the wrong way round -- reading
        // `bp[p*ldb + j]`, i.e. B's row p -- silently computes a different product,
        // which is the same wrong-axis mistake this port has produced before.
        let b0 = bp;
        let b1 = bp.add(ldb);
        let b2 = bp.add(2 * ldb);
        let b3 = bp.add(3 * ldb);
        for p in 0..kc {
            let av = _mm256_loadu_pd(ap.add(p * lda));
            c0 = _mm256_fmadd_pd(av, _mm256_broadcast_sd(&*b0.add(p)), c0);
            c1 = _mm256_fmadd_pd(av, _mm256_broadcast_sd(&*b1.add(p)), c1);
            c2 = _mm256_fmadd_pd(av, _mm256_broadcast_sd(&*b2.add(p)), c2);
            c3 = _mm256_fmadd_pd(av, _mm256_broadcast_sd(&*b3.add(p)), c3);
        }

        // Fold the four accumulators into C. Read-modify-write, so that this kernel
        // composes over splits of `k` without needing beta applied per panel.
        let mut t = [0.0f64; 4];
        _mm256_storeu_pd(t.as_mut_ptr(), _mm256_add_pd(_mm256_loadu_pd(c), c0));
        core::ptr::copy_nonoverlapping(t.as_ptr(), c, 4);
        _mm256_storeu_pd(t.as_mut_ptr(), _mm256_add_pd(_mm256_loadu_pd(c.add(ldc)), c1));
        core::ptr::copy_nonoverlapping(t.as_ptr(), c.add(ldc), 4);
        _mm256_storeu_pd(t.as_mut_ptr(), _mm256_add_pd(_mm256_loadu_pd(c.add(2 * ldc)), c2));
        core::ptr::copy_nonoverlapping(t.as_ptr(), c.add(2 * ldc), 4);
        _mm256_storeu_pd(t.as_mut_ptr(), _mm256_add_pd(_mm256_loadu_pd(c.add(3 * ldc)), c3));
        core::ptr::copy_nonoverlapping(t.as_ptr(), c.add(3 * ldc), 4);
    }
}

/// `C := alpha * A * A^T + beta * C` (dsyrk, uplo = 'L' or 'U').
///
/// `a` is `n x k` column-major and `c` is `n x n`. Only the requested triangle of
/// `c` is written; the other is left alone, as BLAS specifies, so a caller that
/// only reads one triangle is not surprised by the rest containing stale data.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dsyrk(alpha: f64, a: &[f64], beta: f64, c: &mut [f64], n: usize, k: usize, upper: bool) {
    assert_eq!(a.len(), n * k, "dsyrk: a must be n*k");
    assert_eq!(c.len(), n * n, "dsyrk: c must be n*n");

    for j in 0..n {
        // Only the requested triangle of column j. Both branches must be the
        // same type, so the inclusive end is expressed as `j + 1`.
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i + l * n] * a[j + l * n];
            }
            let idx = i + j * n;
            c[idx] = if beta == 0.0 {
                alpha * sum
            } else {
                alpha * sum + beta * c[idx]
            };
        }
    }
}

/// `nrm := |A|` for the one-norm, infinity-norm or Frobenius norm (dlange).
///
/// `a` is `m x n` column-major. `norm` is the BLAS character: `'1'`, `'I'` (or
/// `'i'`) for infinity, `'F'` (or `'E'`) for Frobenius, and `'M'` for the maximum
/// absolute entry.
///
/// # Panics
///
/// Panics on an unrecognised norm character, which in Fortran would be an
/// argument error and here should be a compile-time-visible mistake.
pub fn dlange(norm: char, a: &[f64], m: usize, n: usize) -> f64 {
    assert_eq!(a.len(), m * n, "dlange: a must be m*n");
    match norm {
        'M' | 'm' => a.iter().fold(0.0f64, |acc, v| acc.max(v.abs())),
        // 'O' is LAPACK's other spelling of the one-norm, and is what the C shim
        // sends for `Norm::One`.
        '1' | 'O' | 'o' => {
            // Maximum column sum.
            (0..n)
                .map(|j| a[j * m..(j + 1) * m].iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0f64, f64::max)
        }
        'I' | 'i' => {
            // Maximum row sum.
            (0..m)
                .map(|i| (0..n).map(|j| a[i + j * m].abs()).sum::<f64>())
                .fold(0.0f64, f64::max)
        }
        'F' | 'f' | 'E' | 'e' => dnrm2(a),
        other => panic!("dlange: unrecognised norm '{other}'; expected one of 1, O, I, F, E, M"),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-square matrix, so a transposed index cannot accidentally be right.
    /// Column-major storage of
    ///   [ 1  2  3 ]
    ///   [ 4  5  6 ]
    fn a23() -> Vec<f64> {
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    }

    #[test]
    fn level1_operations() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        daxpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0]);

        dscal(0.5, &mut y);
        assert_eq!(y, vec![6.0, 12.0, 18.0]);

        let mut z = vec![0.0; 3];
        dcopy(&y, &mut z);
        assert_eq!(z, y);

        assert_eq!(ddot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        // Mismatched lengths read the shorter one, as BLAS does.
        assert_eq!(ddot(&[1.0, 2.0, 3.0], &[4.0, 5.0]), 14.0);

        assert_eq!(dnrm2(&[3.0, 4.0]), 5.0);
        assert_eq!(dnrm2(&[]), 0.0);
        assert_eq!(dasum(&[-1.0, 2.0, -3.0]), 6.0);
    }

    #[test]
    fn dnrm2_does_not_overflow_or_underflow() {
        // A naive sum of squares overflows to infinity here.
        let big = vec![1e200, 1e200];
        let n = dnrm2(&big);
        assert!(n.is_finite(), "dnrm2 overflowed: {n}");
        assert!((n / (1e200 * 2.0f64.sqrt()) - 1.0).abs() < 1e-12);

        // ...and underflows to zero here.
        let tiny = vec![1e-200, 1e-200];
        let n = dnrm2(&tiny);
        assert!(n > 0.0, "dnrm2 underflowed to zero");
    }

    #[test]
    fn dgemv_matches_hand_computation() {
        // y := A * [1, 1, 1] over the 2x3 matrix, so y is the row sums: [6, 15].
        let a = a23();
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        dgemv(1.0, &a, &x, 0.0, &mut y, 2, 3);
        assert_eq!(y, vec![6.0, 15.0]);

        // beta must scale the existing y.
        let mut y2 = vec![1.0, 1.0];
        dgemv(2.0, &a, &x, 3.0, &mut y2, 2, 3);
        assert_eq!(y2, vec![2.0 * 6.0 + 3.0, 2.0 * 15.0 + 3.0]);
    }

    /// The naive triple loop, used as the reference for the blocked kernel.
    fn naive_dgemm(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i + l * m] * b[l + j * k];
                }
                let idx = i + j * m;
                c[idx] = if beta == 0.0 {
                    alpha * sum
                } else {
                    alpha * sum + beta * c[idx]
                };
            }
        }
    }

    /// The blocked kernel must agree with the naive one on every shape.
    ///
    /// The sizes are chosen to land on each branch the kernel has, because the
    /// small hand-written cases above never reach any of them: the 4x4
    /// micro-kernel needs `mc, nc >= 4`; the scalar row and column remainders
    /// need `mc % 4 != 0` and `nc % 4 != 0`; and the packing blocks are MC=128,
    /// KC=256, NC=256, so `m > 128`, `k > 256` or `n > 256` are needed before
    /// the block loop iterates more than once.
    #[test]
    fn dgemm_blocked_matches_naive_on_every_branch() {
        let shapes = [
            // tiny / degenerate
            (1, 1, 1),
            (1, 5, 1),
            (5, 1, 1),
            (1, 1, 5),
            (2, 3, 4),
            // micro-kernel boundaries and remainders
            (3, 3, 3),
            (4, 4, 4),
            (5, 5, 5),
            (7, 9, 11),
            (8, 8, 8),
            (13, 6, 17),
            // block boundaries: MC=128, KC=256, NC=256
            (128, 8, 8),
            (129, 8, 8),
            (8, 128, 8),
            (8, 129, 8),
            (8, 8, 256),
            (8, 8, 257),
            (200, 300, 260),
            (260, 130, 300),
        ];
        for (m, n, k) in shapes {
            let a: Vec<f64> = (0..m * k).map(|i| ((i * 7 % 19) as f64) - 9.0).collect();
            let b: Vec<f64> = (0..k * n).map(|i| ((i * 11 % 23) as f64) - 11.0).collect();
            let c0: Vec<f64> = (0..m * n).map(|i| ((i % 5) as f64) * 0.25).collect();

            for (alpha, beta) in [(1.0, 1.0), (1.0, 0.0), (2.5, -1.5), (0.0, 3.0)] {
                let mut want = c0.clone();
                naive_dgemm(alpha, &a, &b, beta, &mut want, m, n, k);
                let mut got = c0.clone();
                dgemm(alpha, &a, &b, beta, &mut got, m, n, k);
                for idx in 0..m * n {
                    let scale = want[idx].abs().max(1.0);
                    assert!(
                        (got[idx] - want[idx]).abs() / scale < 1e-11,
                        "m={m} n={n} k={k} alpha={alpha} beta={beta}: element {idx} is {} not {}",
                        got[idx],
                        want[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn dgemm_matches_hand_computation() {
        // A (2x3) * B (3x2) = C (2x2), all from a23-like data.
        //   A = [[1,2,3],[4,5,6]]
        //   B = [[7,8],[9,10],[11,12]]  stored column-major as 3x2
        let a = a23();
        let b = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // column-major 3x2
        let mut c = vec![0.0; 4];
        dgemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 3);
        // C[0][0] = 1*7+2*9+3*11 = 58 ; C[1][0] = 4*7+5*9+6*11 = 139
        // C[0][1] = 1*8+2*10+3*12 = 64 ; C[1][1] = 4*8+5*10+6*12 = 154
        assert_eq!(c, vec![58.0, 139.0, 64.0, 154.0]);
    }

    #[test]
    fn dgemm_beta_scales_c() {
        let a = a23();
        let b = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];
        dgemm(1.0, &a, &b, 2.0, &mut c, 2, 2, 3);
        assert_eq!(c, vec![58.0 + 2.0, 139.0 + 2.0, 64.0 + 2.0, 154.0 + 2.0]);
    }

    #[test]
    fn dsyrk_computes_both_triangles_and_leaves_the_other_alone() {
        // A = [[1,2],[3,4]] as 2x2 column-major: A A^T = [[5,11],[11,25]].
        let a = vec![1.0, 3.0, 2.0, 4.0];
        let n = 2;
        let k = 2;

        let mut lower = vec![0.0; 4];
        dsyrk(1.0, &a, 0.0, &mut lower, n, k, false);
        // Lower triangle: (0,0)=5, (1,0)=11, (1,1)=25; (0,1) untouched.
        assert_eq!(lower[0], 5.0);
        assert_eq!(lower[1], 11.0);
        assert_eq!(lower[3], 25.0);
        assert_eq!(lower[2], 0.0, "the upper triangle must be left alone");

        let mut upper = vec![0.0; 4];
        dsyrk(1.0, &a, 0.0, &mut upper, n, k, true);
        assert_eq!(upper[0], 5.0);
        assert_eq!(upper[2], 11.0);
        assert_eq!(upper[3], 25.0);
        assert_eq!(upper[1], 0.0, "the lower triangle must be left alone");
    }

    #[test]
    fn dlange_norms() {
        let a = a23(); // [[1,2,3],[4,5,6]]
        assert_eq!(dlange('M', &a, 2, 3), 6.0);
        // Column sums: 5, 7, 9 -> max 9.
        assert_eq!(dlange('1', &a, 2, 3), 9.0);
        // Row sums: 6, 15 -> max 15.
        assert_eq!(dlange('I', &a, 2, 3), 15.0);
        // Frobenius: sqrt(1+4+9+16+25+36) = sqrt(91).
        assert!((dlange('F', &a, 2, 3) - 91.0f64.sqrt()).abs() < 1e-12);
        // 'E' is an alias of 'F'.
        assert_eq!(dlange('E', &a, 2, 3), dlange('F', &a, 2, 3));
    }

    #[test]
    #[should_panic(expected = "unrecognised norm")]
    fn dlange_rejects_an_unknown_norm() {
        dlange('X', &[1.0], 1, 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Complex BLAS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// `Complex64` is `#[repr(C)] { re: f64, im: f64 }`, so a slice of it is a
// sequence of interleaved real/imaginary pairs and can be driven by the real
// kernels applied to the two components separately. That is exact rather than an
// approximation: BLAS's complex operations are defined componentwise in the
// same way.
//
// The alternative -- viewing the buffer as `&[f64]` of twice the length and
// treating the complex layout as a stride-2 real matrix -- would work too, but it
// makes the *matrix* stride depend on the complex stride and is much easier to get
// subtly wrong. Adapting the routines below, which already index a matrix, is
// clearer and exercises code that is already tested.

use num_complex::Complex64;

/// `x := alpha * x` for a complex vector (zscal).
///
/// A complex `alpha` mixes the components, so this cannot decompose into two real
/// scalings; it is a genuine complex multiply.
pub fn zscal(alpha: Complex64, x: &mut [Complex64]) {
    for v in x.iter_mut() {
        *v = alpha * *v;
    }
}

/// `y := alpha * x + y` for complex vectors (zaxpy).
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn zaxpy(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    assert_eq!(x.len(), y.len(), "zaxpy: vectors must have the same length");
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

/// `y := x` for complex vectors (zcopy).
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn zcopy(x: &[Complex64], y: &mut [Complex64]) {
    assert_eq!(x.len(), y.len(), "zcopy: vectors must have the same length");
    y.copy_from_slice(x);
}

/// Euclidean norm of a complex vector (dznrm2), which is real.
pub fn dznrm2(x: &[Complex64]) -> f64 {
    // Sum of squared moduli, scaled to avoid overflow as in dnrm2.
    let mut scale = 0.0f64;
    for v in x {
        let a = v.norm();
        if a > scale {
            scale = a;
        }
    }
    if scale == 0.0 || !scale.is_finite() {
        return x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    }
    let inv = 1.0 / scale;
    let ss: f64 = x.iter().map(|v| (v.norm() * inv) * (v.norm() * inv)).sum();
    scale * ss.sqrt()
}

/// `y := alpha * A * x + beta * y` for complex data (zgemv, trans = 'N').
///
/// `a` is `m x n` column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemv(
    alpha: Complex64,
    a: &[Complex64],
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
    m: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * n, "zgemv: a must be m*n");
    assert_eq!(x.len(), n, "zgemv: x must have n entries");
    assert_eq!(y.len(), m, "zgemv: y must have m entries");

    for v in y.iter_mut() {
        *v = beta * *v;
    }
    for j in 0..n {
        let c = alpha * x[j];
        if c == Complex64::new(0.0, 0.0) {
            continue;
        }
        let col = &a[j * m..(j + 1) * m];
        for i in 0..m {
            y[i] += c * col[i];
        }
    }
}

/// `C := alpha * A * B + beta * C` for complex data (zgemm, both 'N').
///
/// `a` is `m x k`, `b` is `k x n`, `c` is `m x n`, all column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemm(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k, "zgemm: a must be m*k");
    assert_eq!(b.len(), k * n, "zgemm: b must be k*n");
    assert_eq!(c.len(), m * n, "zgemm: c must be m*n");

    for j in 0..n {
        for i in 0..m {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum += a[i + l * m] * b[l + j * k];
            }
            let idx = i + j * m;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
}

/// `C := alpha * A * A^H + beta * C` for complex data (zherk).
///
/// `a` is `n x k` column-major, `c` is `n x n`. `c` is Hermitian, so only the
/// requested triangle is written and the diagonal is real by construction; the
/// imaginary part of a diagonal entry is set to zero rather than left to rounding.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zherk(alpha: f64, a: &[Complex64], beta: f64, c: &mut [Complex64], n: usize, k: usize, upper: bool) {
    assert_eq!(a.len(), n * k, "zherk: a must be n*k");
    assert_eq!(c.len(), n * n, "zherk: c must be n*n");

    for j in 0..n {
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                // a[i][l] * conj(a[j][l])
                sum += a[i + l * n] * a[j + l * n].conj();
            }
            let idx = i + j * n;
            let mut v =
                Complex64::new(alpha * sum.re, alpha * sum.im) + Complex64::new(beta * c[idx].re, beta * c[idx].im);
            if i == j {
                // Hermitian diagonal must be exactly real.
                v.im = 0.0;
            }
            c[idx] = v;
        }
    }
}

/// `C := alpha * A * A^T + beta * C` for complex data (zsyrk).
///
/// Like [`zherk`] but without conjugation, as BLAS defines `zsyrk` for complex
/// symmetric matrices.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zsyrk(alpha: Complex64, a: &[Complex64], beta: Complex64, c: &mut [Complex64], n: usize, k: usize, upper: bool) {
    assert_eq!(a.len(), n * k, "zsyrk: a must be n*k");
    assert_eq!(c.len(), n * n, "zsyrk: c must be n*n");

    for j in 0..n {
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum += a[i + l * n] * a[j + l * n];
            }
            let idx = i + j * n;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
}

/// `|A|` for a complex matrix (zlange), returning a real norm.
///
/// `norm` is `'1'`/`'O'`, `'I'`, `'F'`/`'E'` or `'M'`, as in [`dlange`]. The one- and
/// infinity-norms use the modulus of each entry, as LAPACK specifies.
///
/// # Panics
///
/// Panics on an unrecognised norm character.
pub fn zlange(norm: char, a: &[Complex64], m: usize, n: usize) -> f64 {
    assert_eq!(a.len(), m * n, "zlange: a must be m*n");
    match norm {
        'M' | 'm' => a.iter().fold(0.0f64, |acc, v| acc.max(v.norm())),
        '1' | 'O' | 'o' => {
            // Maximum column sum of moduli. 'O' is LAPACK's other one-norm
            // spelling and is what the C shim sends.
            (0..n)
                .map(|j| a[j * m..(j + 1) * m].iter().map(|v| v.norm()).sum::<f64>())
                .fold(0.0f64, f64::max)
        }
        'I' | 'i' => {
            // Maximum row sum of moduli.
            (0..m)
                .map(|i| (0..n).map(|j| a[i + j * m].norm()).sum::<f64>())
                .fold(0.0f64, f64::max)
        }
        'F' | 'f' | 'E' | 'e' => {
            // Frobenius: sqrt of the sum of squared moduli.
            let sum: f64 = a.iter().map(|v| v.norm_sqr()).sum();
            sum.sqrt()
        }
        other => panic!("zlange: unrecognised norm '{other}'; expected one of 1, O, I, F, E, M"),
    }
}

/// The symmetric rank-k update BLAS calls `dsyrk`, in the orientation the caller
/// actually has.
///
/// `dsyrk` is specified as `C := alpha A A^T + beta C` with `A` of `(n, k)`, or as
/// `C := alpha A^T A + beta C` with `A` of `(k, n)`. Both are the same
/// computation with the two operands exchanged, so rather than duplicate the
/// kernel this adapts the operands and defers to [`dsyrk`].
///
/// Only the `upper` or lower triangle of `c` is written.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn dsyrk_rank(alpha: f64, a: &[f64], beta: f64, c: &mut [f64], n: usize, k: usize, upper: bool, transposed: bool) {
    if !transposed {
        // C := alpha A A^T with A of (n, k): the contraction is over A's columns,
        // which is exactly what `dsyrk` does.
        dsyrk(alpha, a, beta, c, n, k, upper);
        return;
    }

    // C := alpha A^T A with A of (k, n). The *output* is still (n, n), but the
    // contraction runs over A's k rows -- so `k` here is the row count, and
    // delegating to `dsyrk` with the same arguments would contract over the wrong
    // length. This was wrong once: it produced 39 where 15 was expected.
    assert_eq!(a.len(), k * n, "dsyrk_rank: a must be k*n");
    assert_eq!(c.len(), n * n, "dsyrk_rank: c must be n*n");

    for j in 0..n {
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = 0.0;
            for l in 0..k {
                // a[l][i] * a[l][j], with a of (k, n) column-major.
                sum += a[l + i * k] * a[l + j * k];
            }
            let idx = i + j * n;
            c[idx] = if beta == 0.0 {
                alpha * sum
            } else {
                alpha * sum + beta * c[idx]
            };
        }
    }
}

/// `y := alpha * A^T * x + beta * y` for complex data (zgemv, trans = 'T').
///
/// `a` is `m x n` column-major, so `A^T` is `n x m`: `x` has `m` entries and `y`
/// has `n`. Not conjugated; BLAS spells that `CblasConjTrans` and it is a
/// different operation.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemv_trans(
    alpha: Complex64,
    a: &[Complex64],
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
    m: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * n, "zgemv_trans: a must be m*n");
    assert_eq!(x.len(), m, "zgemv_trans: x must have m entries");
    assert_eq!(y.len(), n, "zgemv_trans: y must have n entries");

    for j in 0..n {
        let col = &a[j * m..(j + 1) * m];
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..m {
            sum += col[i] * x[i];
        }
        y[j] = alpha * sum + beta * y[j];
    }
}

/// `C := alpha * A^T * A + beta * C` for complex data (zsyrk, trans = 'T').
///
/// `a` is `k x n` column-major and `c` is `n x n`, so the contraction runs over
/// `a`'s `k` rows. Not the same call as [`zsyrk`], which contracts over `a`'s
/// columns; passing the same shapes to both is silently wrong.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zsyrk_rank(
    alpha: Complex64,
    a: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    n: usize,
    k: usize,
    upper: bool,
) {
    assert_eq!(a.len(), k * n, "zsyrk_rank: a must be k*n");
    assert_eq!(c.len(), n * n, "zsyrk_rank: c must be n*n");

    for j in 0..n {
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum += a[l + i * k] * a[l + j * k];
            }
            let idx = i + j * n;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
}

/// `C := alpha * A^H * A + beta * C` for complex data (zherk, trans = 'C').
///
/// `a` is `k x n` column-major, `c` is `n x n` and Hermitian. The diagonal is
/// forced real, as it must be for a Hermitian matrix.
///
/// **Which operand is conjugated matters.** The entry is
/// `sum_l conj(a[l][i]) * a[l][j]`: the conjugation is on the **first** index,
/// which is the one that is contracted with `A^H`'s row. Conjugating the second
/// instead gives the complex conjugate of every off-diagonal entry -- a
/// different Hermitian matrix, and the one this function used to compute. It is
/// a convincing near-miss: the shape and the real parts are right and only the
/// imaginary parts differ, so it takes a genuinely complex test matrix to see it.
/// Verified against `zgemm('C','N')` and `zherk('L','C')` on the test data, where
/// `C(2,1)` must be `15-1j` and the second-index form gives `15+5j`.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zherk_rank(alpha: f64, a: &[Complex64], beta: f64, c: &mut [Complex64], n: usize, k: usize, upper: bool) {
    assert_eq!(a.len(), k * n, "zherk_rank: a must be k*n");
    assert_eq!(c.len(), n * n, "zherk_rank: c must be n*n");

    for j in 0..n {
        let i_range = if upper { 0..j + 1 } else { j..n };
        for i in i_range {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                // conj(a[l][i]) * a[l][j]: the conjugation is on `i`, not `j`.
                sum += a[l + i * k].conj() * a[l + j * k];
            }
            let idx = i + j * n;
            let mut v =
                Complex64::new(alpha * sum.re, alpha * sum.im) + Complex64::new(beta * c[idx].re, beta * c[idx].im);
            if i == j {
                // A Hermitian diagonal is real by definition.
                v.im = 0.0;
            }
            c[idx] = v;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod complex_tests {
    use super::*;

    /// A `2 x 4` genuinely complex matrix: with real or Hermitian data the
    /// conjugation order in `zherk_rank` makes no difference, so this has to be
    /// complex to test anything at all.
    fn complex_2x4() -> Vec<Complex64> {
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 1.0), // column 0
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0), // column 1
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0), // column 2
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 2.0), // column 3
        ]
    }

    fn assert_close(got: Complex64, want: Complex64, what: &str) {
        assert!(
            (got - want).norm() < 1e-13,
            "{what}: got {got} but the reference says {want}"
        );
    }

    /// `C := A^H A` lower triangle, compared entry by entry with reference BLAS.
    ///
    /// The expected values are the lower triangle of `zgemm('C','N')` on the same
    /// data, cross-checked against `zherk('L','C')` with `alpha=1, beta=0`.
    /// Column-major lower triangle, column by column: (0,0), (1,0), (2,0), (3,0),
    /// (1,1), (2,1), (3,1), (2,2), (3,2), (3,3).
    #[test]
    fn zherk_rank_conjugates_the_first_index() {
        let a = complex_2x4();
        let (k, n) = (2usize, 4usize);
        let mut c = vec![Complex64::new(0.0, 0.0); n * n];
        zherk_rank(1.0, &a, 0.0, &mut c, n, k, false);

        #[rustfmt::skip]
        let lower = [
            (0, 0, 12.0,  0.0),
            (1, 0,  5.0, -1.0),
            (2, 0, 10.0,  2.0),
            (3, 0,  6.0, -6.0),
            (1, 1,  5.0,  0.0),
            (2, 1,  5.0,  0.0),
            (3, 1,  3.0, -2.0),
            (2, 2, 10.0,  0.0),
            (3, 2,  4.0, -6.0),
            (3, 3,  6.0,  0.0),
        ];
        for (i, j, re, im) in lower {
            assert_close(c[i + j * n], Complex64::new(re, im), &format!("({i},{j})"));
        }

        // The upper triangle must be untouched.
        for j in 1..n {
            for i in 0..j {
                assert_eq!(c[i + j * n], Complex64::new(0.0, 0.0), "({i},{j}) must be left alone");
            }
        }

        // Conjugating the second index instead would give the complex conjugate
        // of every off-diagonal entry; this is the assertion that fails if the
        // conjugation order is ever "fixed" back.
        // (row 1, column 0) is `1 + 0 * n`; written out plainly because the
        // `0 * n` term is what `erasing_op` (rightly) objects to.
        assert_eq!(c[1].im, -1.0, "the off-diagonal imaginary part must not be negated");
    }

    /// The same product in the upper triangle, which writes the mirror image.
    #[test]
    fn zherk_rank_upper_triangle() {
        let a = complex_2x4();
        let (k, n) = (2usize, 4usize);
        let mut c = vec![Complex64::new(0.0, 0.0); n * n];
        zherk_rank(1.0, &a, 0.0, &mut c, n, k, true);
        assert_close(c[0 + 1 * n], Complex64::new(5.0, 1.0), "(0,1)");
        assert_close(c[1 + 2 * n], Complex64::new(5.0, 0.0), "(1,2)");
        // The lower triangle must be untouched -- (row 1, column 0) is `c[1]`.
        assert_eq!(c[1], Complex64::new(0.0, 0.0));
    }

    /// `beta` must scale the existing `c`, and the diagonal must come out exactly
    /// real even when `beta * c` has an imaginary part there.
    #[test]
    fn zherk_rank_applies_beta_and_clears_the_diagonal() {
        let a = complex_2x4();
        let (k, n) = (2usize, 4usize);
        let mut c = vec![Complex64::new(0.0, 0.0); n * n];
        for v in c.iter_mut() {
            *v = Complex64::new(1.0, 1.0);
        }
        zherk_rank(2.0, &a, 3.0, &mut c, n, k, false);
        // C(0,0) = 2*12 + 3*(1+1j) -> real 27, imaginary forced to zero.
        assert_close(c[0], Complex64::new(27.0, 0.0), "(0,0)");
        // C(1,0) = 2*(5-1j) + 3*(1+1j) = 13 + 1j.
        assert_close(c[1], Complex64::new(13.0, 1.0), "(1,0)");
    }
}
