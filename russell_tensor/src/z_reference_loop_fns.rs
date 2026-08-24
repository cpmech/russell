//! Loop-based reference implementations of the tensor operations used by the
//! benchmarks. These are simple, obviously-correct implementations (using the
//! `M_TO_IJ` / `MN_TO_IJKL` index mappings) that are used in tests to
//! cross-check the optimized (unrolled) production implementations.

use crate::{M_TO_IJ, MN_TO_IJKL, ONE_BY_3, SQRT_2, SQRT_3, TOL_J2, TWO_BY_3};
use crate::{Rep, Tensor2, Tensor4};
use russell_lab::Matrix;

/// Computes the self-sum-dyadic (ssd) operation using loops
///
/// `Dᵢⱼₖₗ = s (Aᵢₖ Aⱼₗ + Aᵢₗ Aⱼₖ)`
///
/// Reference implementation of [`crate::ssd_fn`].
pub fn ssd_fn_loops(dd: &mut Tensor4, s: f64, aa: &Tensor2) {
    assert_eq!(dd.rep(), Rep::Symmetric);
    let ndim = dd.dim();
    for m in 0..ndim {
        for n in 0..ndim {
            dd.set(m, n, 0.0);
        }
    }
    for m in 0..6 {
        let (i, j) = M_TO_IJ[m];
        let fm = if i == j { 1.0 } else { SQRT_2 };
        for n in 0..6 {
            let (k, l) = M_TO_IJ[n];
            let fn_ = if k == l { 1.0 } else { SQRT_2 };
            let dijkl = aa.get_std(i, k) * aa.get_std(j, l) + aa.get_std(i, l) * aa.get_std(j, k);
            dd.set(m, n, s * fm * fn_ * dijkl);
        }
    }
}

/// Computes the quad-sum-dyadic (qsd) operation using loops
///
/// `Dᵢⱼₖₗ = s (Aᵢₖ Bⱼₗ + Aᵢₗ Bⱼₖ + Bᵢₖ Aⱼₗ + Bᵢₗ Aⱼₖ)`
///
/// Reference implementation of [`crate::qsd_fn`].
pub fn qsd_fn_loops(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.rep(), Rep::Symmetric);
    assert_eq!(bb.rep(), aa.rep());
    let ndim = dd.dim();
    for m in 0..ndim {
        for n in 0..ndim {
            dd.set(m, n, 0.0);
        }
    }
    for m in 0..6 {
        let (i, j) = M_TO_IJ[m];
        let fm = if i == j { 1.0 } else { SQRT_2 };
        for n in 0..6 {
            let (k, l) = M_TO_IJ[n];
            let fn_ = if k == l { 1.0 } else { SQRT_2 };
            let dijkl = aa.get_std(i, k) * bb.get_std(j, l)
                + aa.get_std(i, l) * bb.get_std(j, k)
                + bb.get_std(i, k) * aa.get_std(j, l)
                + bb.get_std(i, l) * aa.get_std(j, k);
            dd.set(m, n, s * fm * fn_ * dijkl);
        }
    }
}

/// Computes the derivative of the squared tensor using loops
///
/// `∂A²ᵢⱼ/∂Aₖₗ = Aᵢₖ δⱼₗ + δᵢₖ Aₗⱼ`
///
/// Reference implementation of [`crate::deriv_squared_tensor`].
pub fn deriv_squared_tensor_loops(da2_da: &mut Tensor4, a: &Tensor2) {
    assert_eq!(da2_da.rep(), Rep::General);
    let a = a.as_std_matrix();
    let mut mat = Matrix::new(9, 9);
    for m in 0..9 {
        for n in 0..9 {
            let (i, j, k, l) = MN_TO_IJKL[m][n];
            let djl = if j == l { 1.0 } else { 0.0 };
            let dik = if i == k { 1.0 } else { 0.0 };
            mat.set(m, n, a.get(i, k) * djl + dik * a.get(l, j));
        }
    }
    da2_da.set_std_matrix(&mat).unwrap();
}

/// Computes the second derivative of the J3 invariant using loops
///
/// `d²J3/dσ⊗dσ = ½ qsd(s,I) − ⅔ (s ⊗ I + I ⊗ s)`, with `s = deviator(σ)`
///
/// Reference implementation of [`crate::deriv2_invariant_jj3`].
pub fn deriv2_invariant_jj3_loops(d2: &mut Tensor4, sigma: &Tensor2) {
    assert_eq!(d2.rep(), Rep::Symmetric);
    assert!(sigma.rep().symmetric());
    let mut s = Tensor2::new(sigma.rep());
    sigma.deviator(&mut s);
    let ndim = d2.dim();
    for m in 0..ndim {
        for n in 0..ndim {
            d2.set(m, n, 0.0);
        }
    }
    for m in 0..6 {
        let (i, j) = M_TO_IJ[m];
        let fm = if i == j { 1.0 } else { SQRT_2 };
        for n in 0..6 {
            let (k, l) = M_TO_IJ[n];
            let fn_ = if k == l { 1.0 } else { SQRT_2 };
            let dik = if i == k { 1.0 } else { 0.0 };
            let djl = if j == l { 1.0 } else { 0.0 };
            let dil = if i == l { 1.0 } else { 0.0 };
            let djk = if j == k { 1.0 } else { 0.0 };
            let dij = if i == j { 1.0 } else { 0.0 };
            let dkl = if k == l { 1.0 } else { 0.0 };
            let qsd = s.get_std(i, k) * djl + s.get_std(i, l) * djk + dik * s.get_std(j, l) + dil * s.get_std(j, k);
            let d2_ijkl = 0.5 * qsd - TWO_BY_3 * (s.get_std(i, j) * dkl + dij * s.get_std(k, l));
            d2.set(m, n, fm * fn_ * d2_ijkl);
        }
    }
}

/// Computes the second derivative of the Lode invariant using loops
///
/// `d²l/dσ⊗dσ = a·d²J3 − b·J3·d²J2 − b·(dJ3⊗dJ2 + dJ2⊗dJ3) + c·J3·(dJ2⊗dJ2)`
///
/// Reference implementation of [`crate::deriv2_invariant_lode`].
pub fn deriv2_invariant_lode_loops(d2: &mut Tensor4, sigma: &Tensor2) -> Option<f64> {
    assert_eq!(d2.rep(), Rep::Symmetric);
    assert!(sigma.rep().symmetric());
    let jj2 = sigma.invariant_jj2();
    if jj2 <= TOL_J2 {
        return None;
    }
    let jj3 = sigma.invariant_jj3();
    let a = 1.5 * SQRT_3 / jj2.powf(1.5);
    let b = 2.25 * SQRT_3 / jj2.powf(2.5);
    let c = 5.625 * SQRT_3 / jj2.powf(3.5);

    // deviator s = dJ2/dσ
    let mut s = Tensor2::new(sigma.rep());
    sigma.deviator(&mut s);

    // dJ3/dσ = s·s − (2/3) J2 I  (standard 3x3)
    let mut d3 = Matrix::new(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += s.get_std(i, k) * s.get_std(k, j);
            }
            let dij = if i == j { 1.0 } else { 0.0 };
            d3.set(i, j, acc - TWO_BY_3 * jj2 * dij);
        }
    }

    // assemble the 6x6 result via loops
    let ndim = d2.dim();
    for m in 0..ndim {
        for n in 0..ndim {
            d2.set(m, n, 0.0);
        }
    }
    for m in 0..6 {
        let (i, j) = M_TO_IJ[m];
        let fm = if i == j { 1.0 } else { SQRT_2 };
        for n in 0..6 {
            let (k, l) = M_TO_IJ[n];
            let fn_ = if k == l { 1.0 } else { SQRT_2 };
            let dik = if i == k { 1.0 } else { 0.0 };
            let djl = if j == l { 1.0 } else { 0.0 };
            let dil = if i == l { 1.0 } else { 0.0 };
            let djk = if j == k { 1.0 } else { 0.0 };
            let dij = if i == j { 1.0 } else { 0.0 };
            let dkl = if k == l { 1.0 } else { 0.0 };
            // d2J2 = Psymdev
            let psd = 0.5 * (dik * djl + dil * djk) - ONE_BY_3 * dij * dkl;
            // d2J3 = 0.5 qsd(s,I) − (2/3)(s⊗I + I⊗s)
            let qsd = s.get_std(i, k) * djl + s.get_std(i, l) * djk + dik * s.get_std(j, l) + dil * s.get_std(j, k);
            let d2j3 = 0.5 * qsd - TWO_BY_3 * (s.get_std(i, j) * dkl + dij * s.get_std(k, l));
            // dJ2 = s, dJ3 = d3
            let dj2_ij = s.get_std(i, j);
            let dj2_kl = s.get_std(k, l);
            let dj3_ij = d3.get(i, j);
            let dj3_kl = d3.get(k, l);
            let val = a * d2j3 - b * jj3 * psd - b * (dj3_ij * dj2_kl + dj2_ij * dj3_kl) + c * jj3 * (dj2_ij * dj2_kl);
            d2.set(m, n, fm * fn_ * val);
        }
    }
    Some(jj2)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{
        deriv_squared_tensor_loops, deriv2_invariant_jj3_loops, deriv2_invariant_lode_loops, qsd_fn_loops, ssd_fn_loops,
    };
    use crate::{Rep, Tensor2, Tensor4};
    use crate::{deriv_squared_tensor, deriv2_invariant_jj3, deriv2_invariant_lode, qsd_fn, ssd_fn};
    use russell_lab::mat_approx_eq;

    const GENERAL_A: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    const GENERAL_B: [[f64; 3]; 3] = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    const SYMMETRIC_A: [[f64; 3]; 3] = [[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]];
    const SYMMETRIC_B: [[f64; 3]; 3] = [[3.0, 5.0, 6.0], [5.0, 2.0, 4.0], [6.0, 4.0, 1.0]];
    const SYM2D_A: [[f64; 3]; 3] = [[1.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    const SYM2D_B: [[f64; 3]; 3] = [[3.0, 4.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_same_t4(a: &Tensor4, b: &Tensor4, tol: f64) {
        mat_approx_eq(&a.as_std_matrix(), &b.as_std_matrix(), tol);
    }

    #[test]
    fn ssd_fn_loops_matches() {
        for (mat, rep) in [
            (&GENERAL_A, Rep::General),
            (&SYMMETRIC_A, Rep::Symmetric),
            (&SYM2D_A, Rep::Symmetric2D),
        ] {
            let a = Tensor2::from_std_matrix(mat, rep).unwrap();
            let mut dd = Tensor4::new(Rep::Symmetric);
            let mut dd_ref = Tensor4::new(Rep::Symmetric);
            ssd_fn(&mut dd, 2.0, &a);
            ssd_fn_loops(&mut dd_ref, 2.0, &a);
            assert_same_t4(&dd, &dd_ref, 1e-12);
        }
    }

    #[test]
    fn qsd_fn_loops_matches() {
        for (mat_a, mat_b, rep) in [
            (&GENERAL_A, &GENERAL_B, Rep::General),
            (&SYMMETRIC_A, &SYMMETRIC_B, Rep::Symmetric),
            (&SYM2D_A, &SYM2D_B, Rep::Symmetric2D),
        ] {
            let a = Tensor2::from_std_matrix(mat_a, rep).unwrap();
            let b = Tensor2::from_std_matrix(mat_b, rep).unwrap();
            let mut dd = Tensor4::new(Rep::Symmetric);
            let mut dd_ref = Tensor4::new(Rep::Symmetric);
            qsd_fn(&mut dd, 2.0, &a, &b);
            qsd_fn_loops(&mut dd_ref, 2.0, &a, &b);
            assert_same_t4(&dd, &dd_ref, 1e-12);
        }
    }

    #[test]
    fn deriv_squared_tensor_loops_matches() {
        let a = Tensor2::from_std_matrix(&GENERAL_A, Rep::General).unwrap();
        let mut da2_da = Tensor4::new(Rep::General);
        deriv_squared_tensor(&mut da2_da, &a);
        let mut da2_da_ref = Tensor4::new(Rep::General);
        deriv_squared_tensor_loops(&mut da2_da_ref, &a);
        assert_same_t4(&da2_da, &da2_da_ref, 1e-12);
    }

    #[test]
    fn deriv2_invariant_jj3_loops_matches() {
        for (mat, rep) in [(&SYMMETRIC_A, Rep::Symmetric), (&SYM2D_A, Rep::Symmetric2D)] {
            let sigma = Tensor2::from_std_matrix(mat, rep).unwrap();
            let mut d2 = Tensor4::new(Rep::Symmetric);
            let mut aux = crate::AuxDeriv2InvariantJ3::new();
            deriv2_invariant_jj3(&mut d2, &mut aux, &sigma);
            let mut d2_ref = Tensor4::new(Rep::Symmetric);
            deriv2_invariant_jj3_loops(&mut d2_ref, &sigma);
            assert_same_t4(&d2, &d2_ref, 1e-11);
        }
    }

    #[test]
    fn deriv2_invariant_lode_loops_matches() {
        for (mat, rep) in [(&SYMMETRIC_A, Rep::Symmetric), (&SYM2D_A, Rep::Symmetric2D)] {
            let sigma = Tensor2::from_std_matrix(mat, rep).unwrap();
            let mut d2 = Tensor4::new(Rep::Symmetric);
            let mut aux = crate::AuxDeriv2InvariantLode::new();
            let res = deriv2_invariant_lode(&mut d2, &mut aux, &sigma);
            let mut d2_ref = Tensor4::new(Rep::Symmetric);
            let res_ref = deriv2_invariant_lode_loops(&mut d2_ref, &sigma);
            assert!(res.is_some());
            assert_eq!(res.unwrap(), res_ref.unwrap());
            assert_same_t4(&d2, &d2_ref, 1e-10);
        }
    }

    #[test]
    fn deriv2_invariant_lode_loops_returns_none() {
        let sigma =
            Tensor2::from_std_matrix(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], Rep::Symmetric).unwrap();
        let mut d2 = Tensor4::new(Rep::Symmetric);
        assert_eq!(deriv2_invariant_lode_loops(&mut d2, &sigma), None);
    }
}
