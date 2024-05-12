use super::{Tensor2, Tensor4};
use russell_lab::{mat_vec_mul, mat_vec_mul_update, vec_mat_mul, vec_outer, vec_outer_update};

#[allow(unused)]
use crate::Mandel; // for documentation

/// Performs the dyadic product between two Tensor2 resulting a Tensor4
///
/// ```text
/// D = α a ⊗ b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = α aᵢⱼ bₖₗ
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Dₘₙ = α aₘ bₙ
/// ```
///
/// # Output
///
/// * `dd` -- the tensor `D`; with the same [Mandel] as `a` and `b`
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b` and `dd`
/// * `b` -- second tensor; with the same [Mandel] as `a` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t2_dyad_t2, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [ 1.0, 10.0, 0.0],
///         [-2.0, -1.0, 0.0],
///         [ 0.0,  0.0, 2.0],
///     ], Mandel::General)?;
///
///     let b = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut dd = Tensor4::new(Mandel::General);
///     t2_dyad_t2(&mut dd, 1.0, &a, &b);
///
///     assert_eq!(
///         format!("{:.1}", dd.as_matrix()),
///         "┌                                                       ┐\n\
///          │   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0 │\n\
///          │  -1.0  -2.0  -3.0  -4.0  -5.0  -6.0  -7.0  -8.0  -9.0 │\n\
///          │   2.0   4.0   6.0   8.0  10.0  12.0  14.0  16.0  18.0 │\n\
///          │  10.0  20.0  30.0  40.0  50.0  60.0  70.0  80.0  90.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │  -2.0  -4.0  -6.0  -8.0 -10.0 -12.0 -14.0 -16.0 -18.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          └                                                       ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_dyad_t2(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_outer(&mut dd.mat, alpha, &a.vec, &b.vec).unwrap();
}

/// Performs the dyadic product between two Tensor2 resulting in a Tensor4 (with update)
///
/// Computes:
///
/// ```text
/// D += α a ⊗ b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ += α aᵢⱼ bₖₗ
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Dₘₙ += α aₘ bₙ
/// ```
///
/// # Output
///
/// * `dd` -- the tensor `D`; with the same [Mandel] as `a` and `b`
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b` and `dd`
/// * `b` -- second tensor; with the same [Mandel] as `a` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_lab::Matrix;
/// use russell_tensor::{t2_dyad_t2_update, Mandel, StrError, Tensor2, Tensor4};
///
/// fn main() -> Result<(), StrError> {
///     #[rustfmt::skip]
///     let a = Tensor2::from_matrix(&[
///         [ 1.0, 10.0, 0.0],
///         [ 2.0,  1.0, 0.0],
///         [ 0.0,  0.0, 2.0],
///     ], Mandel::General)?;
///
///     #[rustfmt::skip]
///     let b = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mat = Matrix::filled(9, 9, 0.5);
///     let mut dd = Tensor4::from_matrix(&mat, Mandel::General)?;
///     t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
///
///     assert_eq!(
///         format!("{:.1}", dd.as_matrix()),
///         "┌                                              ┐\n\
///          │  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 │\n\
///          │  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 │\n\
///          │  2.5  4.5  6.5  8.5 10.5 12.5 14.5 16.5 18.5 │\n\
///          │ 10.5 20.5 30.5 40.5 50.5 60.5 70.5 80.5 90.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  2.5  4.5  6.5  8.5 10.5 12.5 14.5 16.5 18.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          └                                              ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_dyad_t2_update(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_outer_update(&mut dd.mat, alpha, &a.vec, &b.vec).unwrap();
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2
///
/// ```text
/// b = α D : a
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bᵢⱼ = α Σ Σ Dᵢⱼₖₗ aₖₗ
///         k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// bₘ = α Σ Dₘₙ aₙ
///        n
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t4_ddot_t2, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::new(Mandel::General);
///     t4_ddot_t2(&mut b, 1.0, &dd, &a);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                      ┐\n\
///          │  285.0 2850.0    0.0 │\n\
///          │ -570.0 -285.0    0.0 │\n\
///          │    0.0    0.0  570.0 │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t4_ddot_t2(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    mat_vec_mul(&mut b.vec, alpha, &dd.mat, &a.vec).unwrap();
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2 with update
///
/// Computes:
///
/// ```text
/// b = α D : a + β b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bᵢⱼ = α Σ Σ Dᵢⱼₖₗ aₖₗ + β bᵢⱼ
///         k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// bₘ = α Σ Dₘₙ aₙ + β bₘ
///        n
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `beta` -- the other scalar multiplier
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t4_ddot_t2_update, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::from_matrix(&[
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///         [0.0, 0.0, 1.0],
///     ], Mandel::General)?;
///     t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1000.0);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                      ┐\n\
///          │ 1285.0 2850.0    0.0 │\n\
///          │ -570.0  715.0    0.0 │\n\
///          │    0.0    0.0 1570.0 │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t4_ddot_t2_update(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2, beta: f64) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    mat_vec_mul_update(&mut b.vec, alpha, &dd.mat, &a.vec, beta).unwrap();
}

/// Performs the double-dot (ddot) operation between a Tensor2 and a Tensor4
///
/// Computes:
///
/// ```text
/// b = α a : D
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bₖₗ = α Σ Σ aᵢⱼ Dᵢⱼₖₗ
///         i j
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t2_ddot_t4, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::new(Mandel::General);
///     t2_ddot_t4(&mut b, 1.0, &a, &dd);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                   ┐\n\
///          │  31.0 124.0 186.0 │\n\
///          │ 217.0  62.0 155.0 │\n\
///          │ 279.0 248.0  93.0 │\n\
///          └                   ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_ddot_t4(b: &mut Tensor2, alpha: f64, a: &Tensor2, dd: &Tensor4) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_mat_mul(&mut b.vec, alpha, &a.vec, &dd.mat).unwrap();
}

/// Computes Tensor2 double-dot Tensor4 double-dot Tensor2
///
/// Computes:
///
/// ```text
/// s = a : D : b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// s = Σ Σ Σ Σ aᵢⱼ Dᵢⱼₖₗ bₖₗ
///     i j k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// s = Σ Σ aₘ Dₘₙ bₙ
///     m n
/// ```
///
/// Note: the Lagrange multiplier in Plasticity needs this operation.
///
/// # Input
///
/// * `a` -- the first second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `b` -- the second second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Output
///
/// Returns the scalar results.
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t2_ddot_t4_ddot_t2(a: &Tensor2, dd: &Tensor4, b: &Tensor2) -> f64 {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    let dim = a.vec.dim();
    let mut s = 0.0;
    for m in 0..dim {
        for n in 0..dim {
            s += a.vec[m] * dd.mat.get(m, n) * b.vec[n];
        }
    }
    s
}

/// Computes Tensor4 double-dot Tensor2 dyadic Tensor2 double-dot Tensor4
///
/// Computes:
///
/// ```text
/// E = α (D : a) ⊗ (b : D)
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α Σ Σ Σ Σ (Dᵢⱼₛₜ aₛₜ) (bₒₚ Dₒₚₖₗ)
///           s t o p
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α Σ Σ (Dₘₐ aₐ) (bₑ Dₑₙ)
///         a e
/// ```
///
/// Note: the elastoplastic modulus in Plasticity needs this operation.
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as the other tensors
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the first second-order tensor; with the same [Mandel] as the other tensors
/// * `b` -- the second second-order tensor; with the same [Mandel] as the other tensors
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as the other tensors
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t4_ddot_t2_dyad_t2_ddot_t4(ee: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2, dd: &Tensor4) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    let dim = a.vec.dim();
    ee.mat.fill(0.0);
    for m in 0..dim {
        for n in 0..dim {
            for p in 0..dim {
                for q in 0..dim {
                    ee.mat
                        .add(m, n, alpha * dd.mat.get(m, p) * a.vec[p] * b.vec[q] * dd.mat.get(q, n));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mandel, SamplesTensor4, MN_TO_IJKL};
    use russell_lab::{approx_eq, mat_approx_eq, Matrix};

    #[test]
    #[should_panic]
    fn t2_dyad_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        t2_dyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    fn t2_dyad_t2_works() {
        // general dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 │\n\
             │ 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 │\n\
             │ 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             │ 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 │\n\
             │ 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 │\n\
             │ 7.0 7.0 7.0 7.0 7.0 7.0 7.0 7.0 7.0 │\n\
             └                                     ┘"
        );

        // sym-3D dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 │\n\
             │ 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 │\n\
             │ 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             └                                     ┘"
        );

        // sym-2D dyad sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 │\n\
             │ 5.0 5.0 5.0 5.0 0.0 0.0 5.0 0.0 0.0 │\n\
             │ 9.0 9.0 9.0 9.0 0.0 0.0 9.0 0.0 0.0 │\n\
             │ 2.0 2.0 2.0 2.0 0.0 0.0 2.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 2.0 2.0 2.0 2.0 0.0 0.0 2.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             └                                     ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_update_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_update_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
    }

    #[test]
    fn t2_dyad_t2_update_works() {
        // general dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], Mandel::General).unwrap();
        let mat = Matrix::filled(9, 9, 0.1);
        let mut dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        t2_dyad_t2_update(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = "┌                                     ┐\n\
                       │ 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 │\n\
                       │ 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 │\n\
                       │ 9.1 9.1 9.1 9.1 9.1 9.1 9.1 9.1 9.1 │\n\
                       │ 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 │\n\
                       │ 6.1 6.1 6.1 6.1 6.1 6.1 6.1 6.1 6.1 │\n\
                       │ 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 │\n\
                       │ 4.1 4.1 4.1 4.1 4.1 4.1 4.1 4.1 4.1 │\n\
                       │ 8.1 8.1 8.1 8.1 8.1 8.1 8.1 8.1 8.1 │\n\
                       │ 7.1 7.1 7.1 7.1 7.1 7.1 7.1 7.1 7.1 │\n\
                       └                                     ┘";
        assert_eq!(format!("{:.1}", mat), correct);
    }

    fn check_dyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, j) * b.get(k, l));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_dyad_t2_works_extra() {
        // general dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [18.0, 10.0, 2.0, 16.0, 8.0, 14.0, 12.0, 4.0, 6.0],
            [90.0, 50.0, 10.0, 80.0, 40.0, 70.0, 60.0, 20.0, 30.0],
            [162.0, 90.0, 18.0, 144.0, 72.0, 126.0, 108.0, 36.0, 54.0],
            [36.0, 20.0, 4.0, 32.0, 16.0, 28.0, 24.0, 8.0, 12.0],
            [108.0, 60.0, 12.0, 96.0, 48.0, 84.0, 72.0, 24.0, 36.0],
            [54.0, 30.0, 6.0, 48.0, 24.0, 42.0, 36.0, 12.0, 18.0],
            [72.0, 40.0, 8.0, 64.0, 32.0, 56.0, 48.0, 16.0, 24.0],
            [144.0, 80.0, 16.0, 128.0, 64.0, 112.0, 96.0, 32.0, 48.0],
            [126.0, 70.0, 14.0, 112.0, 56.0, 98.0, 84.0, 28.0, 42.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_dyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric dyad symmetric
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 4.0, 2.0, 10.0, 8.0, 12.0, 10.0, 8.0, 12.0],
            [12.0, 8.0, 4.0, 20.0, 16.0, 24.0, 20.0, 16.0, 24.0],
            [18.0, 12.0, 6.0, 30.0, 24.0, 36.0, 30.0, 24.0, 36.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_dyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D dyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 4.0, 2.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            [12.0, 8.0, 4.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [18.0, 12.0, 6.0, 24.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_dyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
    }

    #[test]
    fn t4_ddot_t2_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], Mandel::Symmetric2D).unwrap();
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
        let out = b.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -46.0 -154.0    0.0 │\n\
             │ -154.0  -64.0    0.0 │\n\
             │    0.0    0.0  -82.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_update_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1.0);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_update_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1.0);
    }

    #[test]
    fn t4_ddot_t2_update_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let mut b = Tensor2::from_matrix(&[
            [-1000.0, -1000.0,     0.0],
            [-1000.0, -1000.0,     0.0],
            [    0.0,     0.0, -1000.0],
        ], Mandel::Symmetric2D).unwrap();
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 2.0);
        let out = b.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                         ┐\n\
             │ -2046.0 -2154.0     0.0 │\n\
             │ -2154.0 -2064.0     0.0 │\n\
             │     0.0     0.0 -2082.0 │\n\
             └                         ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
    }

    #[test]
    fn t2_ddot_t4_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], Mandel::Symmetric2D).unwrap();
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
        let out = b.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -90.0 -144.0    0.0 │\n\
             │ -144.0  -96.0    0.0 │\n\
             │    0.0    0.0 -102.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_ddot_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4_ddot_t2(&a, &dd, &b);
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_ddot_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4_ddot_t2(&a, &dd, &b);
    }

    #[test]
    fn t2_ddot_t4_ddot_t2_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mat = Matrix::filled(9, 9, -1.0);
        let dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        let s = t2_ddot_t4_ddot_t2(&a, &dd, &b);
        approx_eq(s, -2025.0, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel3() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    fn t4_ddot_t2_dyad_t2_ddot_t4_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mat = Matrix::filled(9, 9, -1.0);
        let dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        let mut ee = Tensor4::new(Mandel::General);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
        let correct = [
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
        ];
        mat_approx_eq(&ee.as_matrix(), &correct, 1e-11);
    }
}
