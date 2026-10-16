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
        y[j] = if beta == 0.0 { alpha * sum } else { alpha * sum + beta * y[j] };
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

    for j in 0..n {
        let c_col = &mut c[j * m..(j + 1) * m];
        // b[l + j*k] is b[l][j]; a[i + l*m] is a[i][l].
        for i in 0..m {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i + l * m] * b[l + j * k];
            }
            c_col[i] = if beta == 0.0 { alpha * sum } else { alpha * sum + beta * c_col[i] };
        }
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
            c[idx] = if beta == 0.0 { alpha * sum } else { alpha * sum + beta * c[idx] };
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
        '1' => {
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
        other => panic!("dlange: unrecognised norm '{other}'; expected one of 1, I, F, E, M"),
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
pub fn zgemv(alpha: Complex64, a: &[Complex64], x: &[Complex64], beta: Complex64, y: &mut [Complex64], m: usize, n: usize) {
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
pub fn zgemm(alpha: Complex64, a: &[Complex64], b: &[Complex64], beta: Complex64, c: &mut [Complex64], m: usize, n: usize, k: usize) {
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
            let mut v = Complex64::new(alpha * sum.re, alpha * sum.im) + Complex64::new(beta * c[idx].re, beta * c[idx].im);
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
/// `norm` is `'1'`, `'I'`, `'F'`/`'E'` or `'M'`, as in [`dlange`]. The one- and
/// infinity-norms use the modulus of each entry, as LAPACK specifies.
///
/// # Panics
///
/// Panics on an unrecognised norm character.
pub fn zlange(norm: char, a: &[Complex64], m: usize, n: usize) -> f64 {
    assert_eq!(a.len(), m * n, "zlange: a must be m*n");
    match norm {
        'M' | 'm' => a.iter().fold(0.0f64, |acc, v| acc.max(v.norm())),
        '1' => {
            // Maximum column sum of moduli.
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
        other => panic!("zlange: unrecognised norm '{other}'; expected one of 1, I, F, E, M"),
    }
}
