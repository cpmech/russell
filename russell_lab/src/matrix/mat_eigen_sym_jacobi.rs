use super::Matrix;
use crate::{StrError, Vector};

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
/// * `a` -- matrix to compute eigenvalues (SYMMETRIC and SQUARE)
///
/// # Output
///
/// * `l` -- the eigenvalues (unsorted)
/// * `v` -- matrix which columns are the eigenvectors (unsorted)
/// * `a` -- will be modified
///
/// # Notes
///
/// 1. The tolerance is fixed at `1e-15`
/// 2. The maximum number of iterations is fixed at `20`
/// 3. For matrices of order greater than about 10, say, the algorithm is slower,
///    by a significant constant factor, than the QR method.
///
/// # Reference
///
/// This code is based on Section 11.1 Jacobi Transformations (page 463) of Numerical Recipes in C.
///
/// * Press WH, Teukolsky SA, Vetterling WT and Flannery BP (1992),
///   Numerical Recipes in C: The Art of Scientific Computing, 2nd Edition
pub fn mat_eigen_sym_jacobi(l: &mut Vector, v: &mut Matrix, a: &mut Matrix) -> Result<(), StrError> {
    // constants
    const TOLERANCE: f64 = 1e-15;
    const N_MAX_ITERATIONS: usize = 20;

    // check
    let (m, n) = a.dims();
    if m != n {
        return Err("matrix must be square");
    }
    let (mm, nn) = v.dims();
    if mm != m || nn != n {
        return Err("Q matrix mut have the same dimensions as A matrix");
    }
    if l.dim() != n {
        return Err("v vector must have the same dimension as the number of rows of A matrix");
    }

    // auxiliary arrays
    let mut b = vec![0.0; n];
    let mut z = vec![0.0; n];

    // initialize b and v to the diagonal of A
    for p in 0..n {
        b[p] = a.get(p, p);
        l[p] = b[p];
    }

    // initialize Q to the identity matrix
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
    for _ in 0..N_MAX_ITERATIONS {
        // sum off-diagonal elements
        sm = 0.0;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                sm += f64::abs(a.get(p, q));
            }
        }

        // exit point
        if sm < TOLERANCE {
            return Ok(());
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
            z[p] = 0.0;
            l[p] = b[p];
        }
    }

    Err("Jacobi rotation did not converge")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_eigen_sym_jacobi, Matrix};
    use crate::testing::check_eigen_real;
    use crate::{mat_approx_eq, Vector};
    use russell_chk::vec_approx_eq;

    #[test]
    fn mat_eigen_sym_jacobi_works_1() {
        #[rustfmt::skip]
        let data = &[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let mut a = Matrix::from(data);
        let mut v = Matrix::new(3, 3);
        let mut l = Vector::new(3);
        mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[2.0, 2.0, 2.0], 1e-15);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_2() {
        #[rustfmt::skip]
        let data = &[
		    [2.0, 0.0, 0.0],
		    [0.0, 3.0, 4.0],
		    [0.0, 4.0, 9.0],
        ];
        let mut a = Matrix::from(data);
        let mut v = Matrix::new(3, 3);
        let mut l = Vector::new(3);
        mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        let d = 1.0 / f64::sqrt(5.0);
        #[rustfmt::skip]
        let correct = &[
            [1.0,  0.0,   0.0  ],
            [0.0,  2.0*d, 1.0*d],
            [0.0, -1.0*d, 2.0*d],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(l.as_data(), &[2.0, 1.0, 11.0], 1e-15);
        check_eigen_real(data, &v, &l, 1e-15);
    }

    #[test]
    fn mat_eigen_sym_jacobi_works_3() {
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 2.0],
        ];
        let mut a = Matrix::from(data);
        let mut v = Matrix::new(3, 3);
        let mut l = Vector::new(3);
        mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [ 7.81993314738381295e-01, 5.26633230856907386e-01,  3.33382506832158143e-01],
            [-7.14394870018381645e-02, 6.07084171793832561e-01, -7.91419742017035133e-01],
            [-6.19179178753124115e-01, 5.95068272145819699e-01,  5.12358171676802088e-01],
        ];
        mat_approx_eq(&v, correct, 1e-15);
        vec_approx_eq(
            l.as_data(),
            &[
                -1.55809924785903786e+00,
                6.69537390404459476e+00,
                8.62725343814443657e-01,
            ],
            1e-15,
        );
        check_eigen_real(data, &v, &l, 1e-14);
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
        let mut a = Matrix::from(data);
        let mut v = Matrix::new(5, 5);
        let mut l = Vector::new(5);
        mat_eigen_sym_jacobi(&mut l, &mut v, &mut a).unwrap();
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
            l.as_data(),
            &[
                -2.485704750172629e+00,
                1.244545682971212e+01,
                2.694072690168129e+00,
                2.073336609414627e-01,
                -4.861158430649138e+00,
            ],
            1e-12,
        );
        check_eigen_real(data, &v, &l, 1e-14);
    }
}
