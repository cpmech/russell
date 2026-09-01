use super::Tensor4;

#[cfg(feature = "heap")]
use russell_lab::{mat_add, mat_mat_mul};

#[cfg(not(feature = "heap"))]
use russell_lab::{small_mat_add, small_mat_mat_mul};

/// Adds two fourth-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
pub fn t4_add<const N: usize>(c: &mut Tensor4<N>, alpha: f64, a: &Tensor4<N>, beta: f64, b: &Tensor4<N>) {
    #[cfg(feature = "heap")]
    {
        mat_add(&mut c.mat, alpha, &a.mat, beta, &b.mat).unwrap();
    }
    #[cfg(not(feature = "heap"))]
    {
        small_mat_add(&mut c.mat, alpha, &a.mat, beta, &b.mat);
    }
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
/// Or, in Kelvin-Mandel basis:
///
/// ```text
/// Eₘₙ = α Σ Cₘₐ  Dₐₙ
///         a
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `cc` -- the input fourth-order tensor
/// * `dd` -- the fourth-order tensor
///
/// # Examples
///
/// ```
/// use russell_lab::approx_eq;
/// use russell_tensor::{t4_ddot_t4, StrError, Tensor4};
///
/// fn main() -> Result<(), StrError> {
///     let cc = Tensor4::<9>::from_std_matrix(&[
///         [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///     ])?;
///
///     let dd = Tensor4::<9>::from_std_matrix(&[
///         [-1.0, 1.0 / 3.0, 5.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///     ])?;
///
///     let mut ee = Tensor4::<9>::new();
///     t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
///
///     let out = ee.as_std_matrix();
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
pub fn t4_ddot_t4<const N: usize>(ee: &mut Tensor4<N>, alpha: f64, cc: &Tensor4<N>, dd: &Tensor4<N>) {
    #[cfg(feature = "heap")]
    {
        mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, 0.0).unwrap();
    }
    #[cfg(not(feature = "heap"))]
    {
        let dim = ee.dim();
        small_mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, 0.0, dim);
    }
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
/// Or, in Kelvin-Mandel basis:
///
/// ```text
/// Eₘₙ = α (Σ Cₘₐ  Dₐₙ) + β Eₘₙ
///          a
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `cc` -- the input fourth-order tensor
/// * `dd` -- the fourth-order tensor
/// * `beta` -- the other scalar multiplier
pub fn t4_ddot_t4_update<const N: usize>(ee: &mut Tensor4<N>, alpha: f64, cc: &Tensor4<N>, dd: &Tensor4<N>, beta: f64) {
    #[cfg(feature = "heap")]
    {
        mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, beta).unwrap();
    }
    #[cfg(not(feature = "heap"))]
    {
        let dim = ee.dim();
        small_mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, beta, dim);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplesTensor4;
    use russell_lab::{Matrix, mat_approx_eq};

    #[test]
    fn t4_add_works() {
        let mut a = Tensor4::<4>::new();
        let mut b = Tensor4::<4>::new();
        let mut c = Tensor4::<4>::new();
        a.sym_set_std(0, 0, 0, 0, 1.0);
        b.sym_set_std(0, 0, 0, 0, 1.0);
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
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-14);
    }

    #[test]
    fn t4_ddot_t4_works() {
        let cc = Tensor4::<4>::from_std_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX).unwrap();
        let mut ee = Tensor4::<4>::new();
        t4_ddot_t4(&mut ee, 2.0, &cc, &cc);
        let out = ee.as_std_matrix();
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
    fn t4_ddot_t4_update_works() {
        let cc = Tensor4::<4>::from_std_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX).unwrap();
        let mut mat = Matrix::new(9, 9);
        mat.set(0, 0, 0.1);
        mat.set(1, 1, 0.1);
        mat.set(2, 2, 0.1);
        let mut ee = Tensor4::<4>::from_std_matrix(&mat).unwrap();
        t4_ddot_t4_update(&mut ee, 2.0, &cc, &cc, 2.0);
        let out = ee.as_std_matrix();
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
