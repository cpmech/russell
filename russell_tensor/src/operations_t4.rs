use super::Tensor4;
use russell_lab::{mat_add, mat_mat_mul};

#[allow(unused)]
use crate::Mandel; // for documentation

/// Adds two fourth-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t4_add(c: &mut Tensor4, alpha: f64, a: &Tensor4, beta: f64, b: &Tensor4) {
    assert_eq!(b.mandel, a.mandel);
    assert_eq!(c.mandel, a.mandel);
    mat_add(&mut c.mat, alpha, &a.mat, beta, &b.mat).unwrap();
}

/// Performs the double-dot (ddot) operation between two Tensor4
///
/// Computes:
///
/// ```text
/// E = α C : D
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α Σ Σ Cᵢⱼₛₜ : Dₛₜₖₗ
///           s t
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α Σ Cₘₐ  Dₐₙ
///         m
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as `cc` and `dd`
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
/// use russell_lab::approx_eq;
/// use russell_tensor::{Mandel, t4_ddot_t4, StrError, Tensor4};
///
/// fn main() -> Result<(), StrError> {
///     let cc = Tensor4::from_matrix(
///         &[
///             [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///         ],
///         Mandel::General,
///     )?;
///
///     let dd = Tensor4::from_matrix(
///         &[
///             [-1.0, 1.0 / 3.0, 5.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///         ],
///         Mandel::General,
///     )?;
///
///     let mut ee = Tensor4::new(Mandel::General);
///     t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
///
///     let out = ee.as_matrix();
///     for i in 0..9 {
///         for j in 0..9 {
///             if i == j {
///                 approx_eq(out.get(i, j), 1.0, 1e-15);
///             } else {
///                 approx_eq(out.get(i, j), 0.0, 1e-15);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn t4_ddot_t4(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4) {
    assert_eq!(cc.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, 0.0).unwrap();
}

/// Performs the double-dot (ddot) operation between two Tensor4 with update
///
/// Computes:
///
/// ```text
/// E = α C : D + β E
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α (Σ Σ Cᵢⱼₛₜ : Dₛₜₖₗ) + β Eᵢⱼₖₗ
///            s t
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α (Σ Cₘₐ  Dₐₙ) + β Eₘₙ
///          m
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as `cc` and `dd`
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
pub fn t4_ddot_t4_update(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4, beta: f64) {
    assert_eq!(cc.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, beta).unwrap();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mandel, SamplesTensor4};
    use russell_lab::{mat_approx_eq, Matrix};

    #[test]
    #[should_panic]
    fn t4_add_panics_on_different_mandel1() {
        let a = Tensor4::new(Mandel::Symmetric2D);
        let b = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut c = Tensor4::new(Mandel::Symmetric2D);
        t4_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    #[should_panic]
    fn t4_add_panics_on_different_mandel2() {
        let a = Tensor4::new(Mandel::Symmetric2D);
        let b = Tensor4::new(Mandel::Symmetric2D);
        let mut c = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        t4_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    fn t4_add_works() {
        let mut a = Tensor4::new(Mandel::Symmetric2D);
        let mut b = Tensor4::new(Mandel::Symmetric2D);
        let mut c = Tensor4::new(Mandel::Symmetric2D);
        a.sym_set(0, 0, 0, 0, 1.0);
        b.sym_set(0, 0, 0, 0, 1.0);
        t4_add(&mut c, 2.0, &a, 3.0, &b);
        #[rustfmt::skip]
        let correct = &[
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        mat_approx_eq(&c.as_matrix(), correct, 1e-14);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_panics_on_different_mandel1() {
        let cc = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_panics_on_different_mandel2() {
        let cc = Tensor4::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    fn t4_ddot_t4_works() {
        let cc = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 2.0, &cc, &cc);
        let out = ee.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                ┐\n\
             │  820.0  872.0  924.0 1288.0    0.0    0.0 1288.0    0.0    0.0 │\n\
             │ 1120.0 1202.0 1284.0 1858.0    0.0    0.0 1858.0    0.0    0.0 │\n\
             │ 1420.0 1532.0 1644.0 2428.0    0.0    0.0 2428.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             └                                                                ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_update_panics_on_different_mandel1() {
        let cc = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_update_panics_on_different_mandel2() {
        let cc = Tensor4::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    fn t4_ddot_t4_update_works() {
        let cc = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        let mut mat = Matrix::new(9, 9);
        mat.set(0, 0, 0.1);
        mat.set(1, 1, 0.1);
        mat.set(2, 2, 0.1);
        let mut ee = Tensor4::from_matrix(&mat, Mandel::Symmetric2D).unwrap();
        t4_ddot_t4_update(&mut ee, 2.0, &cc, &cc, 2.0);
        let out = ee.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                ┐\n\
             │  820.2  872.0  924.0 1288.0    0.0    0.0 1288.0    0.0    0.0 │\n\
             │ 1120.0 1202.2 1284.0 1858.0    0.0    0.0 1858.0    0.0    0.0 │\n\
             │ 1420.0 1532.0 1644.2 2428.0    0.0    0.0 2428.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             └                                                                ┘"
        );
    }
}
