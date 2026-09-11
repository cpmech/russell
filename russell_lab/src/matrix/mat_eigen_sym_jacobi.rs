use super::Matrix;
use crate::{StrError, Vector};

/// Defines the tolerance to accept a diagonal matrix
const TOLERANCE: f64 = 1e-15;

/// Defines the max number of iterations
const N_MAX_ITERATIONS: usize = 20;

/// Performs the Jacobi transformation of a symmetric matrix to find its eigenvectors and eigenvalues
///
/// The Jacobi method consists of a sequence of orthogonal similarity transformations. Each
/// transformation (a Jacobi rotation) is just a plane rotation designed to annihilate one of the
/// off-diagonal matrix elements. Successive transformations undo previously set zeros, but the
/// off-diagonal elements nevertheless get smaller and smaller. Accumulating the product of the
/// transformations as you go gives the matrix of eigenvectors (Q), while the elements of the final
/// diagonal matrix (A) are the eigenvalues.
///
/// The Jacobi method is absolutely foolproof for all real symmetric matrices.
///
/// ```text
/// A = V ⋅ L ⋅ Vᵀ
/// ```
///
/// # Input
///
/// * `l` -- vector with dimension N; overwritten with the eigenvalues.
/// * `v` -- matrix with dimensions N×N; overwritten with the eigenvectors (one per column).
/// * `a` -- the symmetric matrix to be decomposed. Must be square with dimension N ≥ 1. It is
///   overwritten on output. **Important:** Symmetry is not checked here.
///
/// # Output
///
/// * `l` -- the eigenvalues (unsorted)
/// * `v` -- the eigenvectors, where the i-th column corresponds to the eigenvalue `l[i]` (unsorted)
/// * `a` -- overwritten with the diagonalized matrix (eigenvalues on the diagonal)
/// * Returns the number of Jacobi iterations performed
///
/// The decomposition satisfies `A = V ⋅ L ⋅ Vᵀ`, where `L` is the diagonal matrix with the
/// eigenvalues on its diagonal and the columns of `V` are the corresponding (unit) eigenvectors.
/// Both the eigenvalues and the eigenvectors are returned in an arbitrary (unsorted) order.
///
/// # Notes
///
/// 1. The tolerance is fixed at `1e-15`
///    (for the sum of the absolute value of the upper off-diagonal elements)
/// 2. The maximum number of iterations is fixed at `20`
/// 3. For matrices of order greater than about 10, say, the algorithm is slower,
///    by a significant constant factor, than the QR method.
/// 4. This function is recommended for small matrices only, e.g., dim ≤ 32
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_eigen_sym_jacobi, Matrix, Vector};
/// use russell_lab::StrError;
///
/// fn main() -> Result<(), StrError> {
///     // set matrix
///     let data = [
///         [2.0, 0.0, 0.0],
///         [0.0, 3.0, 4.0],
///         [0.0, 4.0, 9.0],
///     ];
///     let mut a = Matrix::from(&data);
///
///     // perform the eigen-decomposition
///     let mut l = Vector::new(3);
///     let mut v = Matrix::new(3, 3);
///     mat_eigen_sym_jacobi(&mut l, &mut v, &mut a)?;
///     println!("eigenvalues =\n{}", l);
///     println!("eigenvectors =\n{:.5}", v);
///
///     // check results
///     assert_eq!(
///         format!("{}", l),
///         "┌    ┐\n\
///          │  2 │\n\
///          │  1 │\n\
///          │ 11 │\n\
///          └    ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn mat_eigen_sym_jacobi(l: &mut Vector, v: &mut Matrix, a: &mut Matrix) -> Result<usize, StrError> {
    // check
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    if m == 0 {
        return Err("matrix dimension must be ≥ 1");
    }
    let (mm, nn) = v.dims();
    if mm != m || nn != n {
        return Err("v and a matrices must have the same dimensions");
    }
    if l.dim() != n {
        return Err("l vector has incompatible dimension");
    }

    // auxiliary arrays
    let mut b = vec![0.0; n];
    let mut z = vec![0.0; n];

    // initialize b and l to the diagonal of A
    for p in 0..n {
        b[p] = a.get(p, p);
        l[p] = b[p];
    }

    // initialize v to the identity matrix
    for p in 0..n {
        for q in 0..n {
            v.set(p, q, 0.0);
        }
        v.set(p, p, 1.0);
    }

    // auxiliary variables
    let mut sm: f64;
    let mut h: f64;
    let mut t: f64;
    let mut theta: f64;
    let mut c: f64;
    let mut s: f64;
    let mut tau: f64;
    let mut g: f64;

    // perform iterations
    for iteration in 0..N_MAX_ITERATIONS {
        // sum magnitude of upper off-diagonal elements
        sm = 0.0;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                sm += f64::abs(a.get(p, q));
            }
        }

        // exit point
        if sm < TOLERANCE {
            return Ok(iteration + 1);
        }

        // rotations
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                h = l[q] - l[p];
                if f64::abs(h) <= TOLERANCE {
                    t = 1.0;
                } else {
                    theta = 0.5 * h / (a.get(p, q));
                    t = 1.0 / (f64::abs(theta) + f64::sqrt(1.0 + theta * theta));
                    if theta < 0.0 {
                        t = -t;
                    }
                }
                c = 1.0 / f64::sqrt(1.0 + t * t);
                s = t * c;
                tau = s / (1.0 + c);
                h = t * a.get(p, q);
                z[p] -= h;
                z[q] += h;
                l[p] -= h;
                l[q] += h;
                a.set(p, q, 0.0);
                // case of rotations 0 ≤ j < p
                for j in 0..p {
                    g = a.get(j, p);
                    h = a.get(j, q);
                    a.set(j, p, g - s * (h + g * tau));
                    a.set(j, q, h + s * (g - h * tau));
                }
                // case of rotations p < j < q
                for j in (p + 1)..q {
                    g = a.get(p, j);
                    h = a.get(j, q);
                    a.set(p, j, g - s * (h + g * tau));
                    a.set(j, q, h + s * (g - h * tau));
                }
                // case of rotations q < j < n
                for j in (q + 1)..n {
                    g = a.get(p, j);
                    h = a.get(q, j);
                    a.set(p, j, g - s * (h + g * tau));
                    a.set(q, j, h + s * (g - h * tau));
                }
                // Q matrix
                for j in 0..n {
                    g = v.get(j, p);
                    h = v.get(j, q);
                    v.set(j, p, g - s * (h + g * tau));
                    v.set(j, q, h + s * (g - h * tau));
                }
            }
        }
        for p in 0..n {
            b[p] += z[p];
            l[p] = b[p];
            z[p] = 0.0;
        }
    }

    Err("Jacobi rotation did not converge")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Matrix, mat_eigen_sym_jacobi};
    use crate::math::SQRT_2;
    use crate::matrix::testing::check_eigen_sym;
    use crate::{AsArray2D, Vector, mat_approx_eq, vec_approx_eq};

    fn calc_eigen<'a, T>(data: &'a T) -> (usize, Vector, Matrix)
    where
        T: AsArray2D<'a, f64>,
    {
        let mut a = Matrix::from(data);
        let (m, n) = a.dims();
        let mut v = Matrix::new(m, n);
        let mut l = Vector::new(n);
        let nit = mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        (nit, l, v)
    }

    /// Sorts the eigenvalues in ascending order, reordering the eigenvector columns to match
    fn sort_eigen(l: &Vector, v: &Matrix) -> (Vector, Matrix) {
        let n = l.dim();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| l[i].total_cmp(&l[j]));
        let mut ls = Vector::new(n);
        let mut vs = Matrix::new(n, n);
        for (k, &i) in idx.iter().enumerate() {
            ls.set(k, l[i]);
            for r in 0..n {
                vs.set(r, k, v.get(r, i));
            }
        }
        (ls, vs)
    }

    /// Checks each eigenvector column against the reference (up to sign) via |v·v_ref| ≈ 1
    fn check_eigenvec_sign(v: &Matrix, v_ref: &[[f64; 3]; 3], tol: f64) {
        for k in 0..3 {
            let mut dot = 0.0;
            for r in 0..3 {
                dot += v.get(r, k) * v_ref[r][k];
            }
            let err = (dot.abs() - 1.0).abs();
            assert!(
                err < tol,
                "eigenvector {} not aligned with reference: |v·v_ref|-1 = {:e}",
                k,
                err
            );
        }
    }

    #[test]
    fn mat_eigen_sym_jacobi_handles_errors() {
        let mut a = Matrix::new(0, 1);
        let mut v = Matrix::new(1, 1);
        let mut l = Vector::new(0);
        assert_eq!(
            mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).err(),
            Some("matrix must be square")
        );
        let mut a = Matrix::new(0, 0);
        assert_eq!(
            mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).err(),
            Some("matrix dimension must be ≥ 1")
        );
        let mut a = Matrix::new(2, 2);
        assert_eq!(
            mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).err(),
            Some("v and a matrices must have the same dimensions")
        );
        let mut a = Matrix::new(1, 1);
        assert_eq!(
            mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).err(),
            Some("l vector has incompatible dimension")
        );
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_0() {
        // 1x1 matrix
        let data = &[[2.0]];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 1);
        mat_approx_eq(&v, &[[1.0]], 1e-15);
        vec_approx_eq(&l, &[2.0], 1e-15);

        // 2x2 matrix
        let data = &[[2.0, 1.0], [1.0, 2.0]];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 2);
        mat_approx_eq(
            &v,
            &[[1.0 / SQRT_2, 1.0 / SQRT_2], [-1.0 / SQRT_2, 1.0 / SQRT_2]],
            1e-15,
        );
        vec_approx_eq(&l, &[1.0, 3.0], 1e-15);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_1() {
        #[rustfmt::skip]
        let correct = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        // all zero
        #[rustfmt::skip]
        let data = &[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 1);
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(&l, &[0.0, 0.0, 0.0], 1e-15);

        // 2-repeated, with one zero diagonal entry
        #[rustfmt::skip]
        let data = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 1);
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(&l, &[2.0, 2.0, 0.0], 1e-15);
        check_eigen_sym(data, &v, &l, 1e-15);

        // 3-repeated / diagonal
        #[rustfmt::skip]
        let data = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 1);
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(&l, &[2.0, 2.0, 2.0], 1e-15);
        check_eigen_sym(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_2() {
        #[rustfmt::skip]
        let data = &[
		    [2.0, 0.0, 0.0],
		    [0.0, 3.0, 4.0],
		    [0.0, 4.0, 9.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 2);
        let d = 1.0 / f64::sqrt(5.0);
        #[rustfmt::skip]
        let correct = &[
            [1.0,  0.0,   0.0  ],
            [0.0,  2.0*d, 1.0*d],
            [0.0, -1.0*d, 2.0*d],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(&l, &[2.0, 1.0, 11.0], 1e-15);
        check_eigen_sym(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_3() {
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 2.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 5);
        #[rustfmt::skip]
        let correct = &[
            [ 7.81993314738381295e-01, 5.26633230856907386e-01,  3.33382506832158143e-01],
            [-7.14394870018381645e-02, 6.07084171793832561e-01, -7.91419742017035133e-01],
            [-6.19179178753124115e-01, 5.95068272145819699e-01,  5.12358171676802088e-01],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(
            &l,
            &[
                -1.55809924785903786e+00,
                6.69537390404459476e+00,
                8.62725343814443657e-01,
            ],
            1e-15,
        );
        check_eigen_sym(data, &v, &l, 1e-14);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_4() {
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 0.0, 2.0, 4.0],
            [3.0, 0.0, 2.0, 1.0, 3.0],
            [4.0, 2.0, 1.0, 1.0, 2.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let (nit, l, v) = calc_eigen(data);
        assert_eq!(nit, 6);
        #[rustfmt::skip]
        let correct = &[
            [ 4.265261184874604e-01, 5.285232769688938e-01,  1.854383137677959e-01,  2.570216184506737e-01, -6.620355997875309e-01],
            [-3.636641874245161e-01, 4.182907021187977e-01, -7.200691218899387e-01, -3.444995789572199e-01, -2.358002271092630e-01],
            [-5.222548807800880e-01, 3.413546312786976e-01,  6.672573809673910e-01, -4.053509412317634e-01, -3.442465966457679e-02],
            [-4.133525029362699e-01, 3.807798553184266e-01, -3.950209555261502e-02,  7.608554466087614e-01,  3.220015278111787e-01],
            [ 4.921517823299884e-01, 5.330851261396132e-01, -1.789590676939640e-02, -2.684204380363021e-01,  6.334327718104180e-01],
        ];
        mat_approx_eq(&v, correct, 1e-13);
        vec_approx_eq(
            &l,
            &[
                -2.485704750172629e+00,
                1.244545682971212e+01,
                2.694072690168129e+00,
                2.073336609414627e-01,
                -4.861158430649138e+00,
            ],
            1e-12,
        );
        check_eigen_sym(data, &v, &l, 1e-14);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_5() {
        let samples = &[
            (
                // 0
                2,
                [[1.0, 2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 0.0, -2.0]],
                1e-15,
            ),
            (
                // 1
                2,
                [[-100.0, 33.0, 0.0], [33.0, -200.0, 0.0], [0.0, 0.0, 150.0]],
                1e-13,
            ),
            (
                // 2
                4,
                [[1.0, 2.0, 4.0], [2.0, -2.0, 3.0], [4.0, 3.0, -2.0]],
                1e-14,
            ),
            (
                // 3
                4,
                [[-100.0, -10.0, 20.0], [-10.0, -200.0, 15.0], [20.0, 15.0, -300.0]],
                1e-13,
            ),
            (
                // 4
                2,
                [[-100.0, 0.0, -10.0], [0.0, -200.0, 0.0], [-10.0, 0.0, 100.0]],
                1e-13,
            ),
            (
                // 5
                2,
                [[0.13, 1.2, 0.0], [1.2, -20.0, 0.0], [0.0, 0.0, -28.0]],
                1e-14,
            ),
            (
                // 6
                2,
                [[-10.0, 3.3, 0.0], [3.3, -2.0, 0.0], [0.0, 0.0, 1.5]],
                1e-15,
            ),
            (
                // 7
                4,
                [[0.1, 0.2, 0.8], [0.2, -1.3, 0.3], [0.8, 0.3, -0.2]],
                1e-15,
            ),
            (
                // 8
                4,
                [[-10.0, -1.0, 2.0], [-1.0, -20.0, 1.0], [2.0, 1.0, -30.0]],
                1e-14,
            ),
            (
                // 9
                2,
                [[-10.0, 0.0, -1.0], [0.0, -20.0, 0.0], [-1.0, 0.0, 10.0]],
                1e-15,
            ),
        ];
        let mut test_id = 0;
        for (nit_correct, data, tol) in samples {
            println!("test = {}", test_id);
            let (nit, l, v) = calc_eigen(data);
            assert_eq!(nit, *nit_correct);
            check_eigen_sym(data, &v, &l, *tol);
            test_id += 1;
        }
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_6() {
        let size = 8;

        let mut a = Matrix::filled(size, size, 2.0);
        let a_copy = a.clone();
        let mut v = Matrix::new(size, size);
        let mut l = Vector::new(size);
        let nit = mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        assert_eq!(nit, 4);
        // println!("a =\n{}", a);
        // println!("nit = {}", nit);
        // println!("l =\n{}", l);
        // println!("v =\n{}", v);
        check_eigen_sym(&a_copy, &v, &l, 1e-14);

        let mut a = Matrix::filled(size, size, (size + 1) as f64);
        for i in 0..(size - 1) {
            for j in (i + 1)..size {
                a.set(i, j, (i + j) as f64);
                a.set(j, i, (i + j) as f64);
            }
        }
        let a_copy = a.clone();
        let mut v = Matrix::new(size, size);
        let mut l = Vector::new(size);
        let nit = mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        assert_eq!(nit, 7);
        // println!("a =\n{}", a);
        // println!("nit = {}", nit);
        // println!("l =\n{}", l);
        // println!("v =\n{}", v);
        check_eigen_sym(&a_copy, &v, &l, 1e-12);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_7() {
        // Reference values generated with Python/NumPy:
        //
        //     import numpy as np
        //
        //     def gen(a):
        //         a = np.array(a, dtype=float)
        //         w, v = np.linalg.eigh(a)   # ascending eigenvalues; orthonormal eigenvectors in columns
        //         for j in range(3):          # normalize each eigenvector so its largest component is positive
        //             i = np.argmax(np.abs(v[:, j]))
        //             if v[i, j] < 0:
        //                 v[:, j] *= -1.0
        //         print(w)
        //         print(v)                    # v[component, eigenvector]
        //
        //     # singular (det = 0)
        //     gen([[1, 2, 3], [2, 5, 7], [3, 7, 10]])
        //     gen([[2, 3, 0], [3, 6, 0], [0, 0, 0]])
        //     gen([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        //     # ill-posed
        //     gen([[1, 1e-6, 2e-6], [1e-6, 1, 3e-6], [2e-6, 3e-6, 1]])
        //     gen([[1e6, 200, 0], [200, 1, 50], [0, 50, 1e-6]])
        //     # well-posed
        //     gen([[4, 1, 2], [1, 5, 3], [2, 3, 6]])
        //     gen([[5, 2, 1], [2, 4, 0], [1, 0, 3]])
        //
        // The computed eigenvalues are sorted in ascending order and compared directly. The
        // eigenvectors are unique only up to sign, so they are compared via |v·v_ref| ≈ 1. For
        // repeated or nearly-repeated eigenvalues the eigenvectors are also free to rotate within
        // the invariant subspace, so in those cases only the eigenvalues and the residual (a·v = v·λ)
        // are checked.

        // ----------------------------- singular matrices (det = 0) -----------------------------

        // rank 2, distinct nonzero eigenvalues
        let data = &[[1.0, 2.0, 3.0], [2.0, 5.0, 7.0], [3.0, 7.0, 10.0]];
        let l_ref = &[0.0, 0.18975032409334291, 15.810249675906647];
        let v_ref = &[
            [0.5773502691896242, 0.7815027659502837, 0.23647429771272274],
            [0.5773502691896271, -0.5955441321364421, 0.5585640995843871],
            [-0.5773502691896262, 0.18595863381383737, 0.7950383972971101],
        ];
        let (_, l, v) = calc_eigen(data);
        let (ls, vs) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-13);
        check_eigen_sym(data, &v, &l, 1e-13);
        check_eigenvec_sign(&vs, v_ref, 1e-12);

        // rank 2, block-diagonal
        let data = &[[2.0, 3.0, 0.0], [3.0, 6.0, 0.0], [0.0, 0.0, 0.0]];
        let l_ref = &[0.0, 0.39444872453601065, 7.60555127546399];
        let v_ref = &[
            [0.0, 0.8816745987679436, 0.47185792553202427],
            [0.0, -0.47185792553202427, 0.8816745987679436],
            [1.0, 0.0, 0.0],
        ];
        let (_, l, v) = calc_eigen(data);
        let (ls, vs) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-13);
        check_eigen_sym(data, &v, &l, 1e-13);
        check_eigenvec_sign(&vs, v_ref, 1e-12);

        // rank 1 (two repeated zero eigenvalues; eigenvectors are not unique -> residual only)
        let data = &[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        let l_ref = &[0.0, 0.0, 14.0];
        let (_, l, v) = calc_eigen(data);
        let (ls, _) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-13);
        check_eigen_sym(data, &v, &l, 1e-13);

        // ----------------------------- ill-posed matrices -----------------------------

        // nearly-degenerate eigenvalues (eigenvectors are ill-determined -> residual only)
        let data = &[[1.0, 1e-6, 2e-6], [1e-6, 1.0, 3e-6], [2e-6, 3e-6, 1.0]];
        let l_ref = &[0.9999967980882232, 0.9999990888211926, 1.0000041130905848];
        let (_, l, v) = calc_eigen(data);
        let (ls, _) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-12);
        check_eigen_sym(data, &v, &l, 1e-12);

        // large condition number (widely spread eigenvalues)
        let data = &[[1e6, 200.0, 0.0], [200.0, 1.0, 50.0], [0.0, 50.0, 1e-6]];
        let l_ref = &[-49.52230246122888, 50.48230342272884, 1000000.0400000386];
        let v_ref = &[
            [0.0001407339565551668, -0.00014210569648128322, 0.999999979999962],
            [-0.7037046301236495, 0.7104926132919764, 0.0002000001885001661],
            [0.7104926275032636, 0.703704644196348, 1.000000902501756e-8],
        ];
        let (_, l, v) = calc_eigen(data);
        let (ls, vs) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-7);
        check_eigen_sym(data, &v, &l, 1e-7);
        check_eigenvec_sign(&vs, v_ref, 1e-12);

        // ----------------------------- well-posed matrices -----------------------------

        let data = &[[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]];
        let l_ref = &[2.1943971674224088, 3.386770156607549, 9.418832675970037];
        let v_ref = &[
            [-0.4412246968734644, 0.8155834192895004, 0.37435872241603624],
            [-0.5773502691896257, -0.5773502691896258, 0.5773502691896254],
            [0.6870134158337713, 0.03860508835262272, 0.7256185041863941],
        ];
        let (_, l, v) = calc_eigen(data);
        let (ls, vs) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-13);
        check_eigen_sym(data, &v, &l, 1e-13);
        check_eigenvec_sign(&vs, v_ref, 1e-12);

        let data = &[[5.0, 2.0, 1.0], [2.0, 4.0, 0.0], [1.0, 0.0, 3.0]];
        let l_ref = &[1.9999999999999993, 3.267949192431123, 6.732050807568877];
        let v_ref = &[
            [-0.5773502691896257, 0.21132486540518677, 0.7886751345948129],
            [0.577350269189626, -0.5773502691896256, 0.5773502691896256],
            [0.5773502691896254, 0.7886751345948131, 0.21132486540518713],
        ];
        let (_, l, v) = calc_eigen(data);
        let (ls, vs) = sort_eigen(&l, &v);
        vec_approx_eq(&ls, l_ref, 1e-13);
        check_eigen_sym(data, &v, &l, 1e-13);
        check_eigenvec_sign(&vs, v_ref, 1e-12);
    }
}
