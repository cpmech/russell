use super::{Spectral2, Tensor2};
use russell_lab::{
    Matrix, StrError, Vector, mat_mat_mul, mat_svd, small_mat_inv, small_mat_mat_mul, small_mat_t_mat_mul,
};

/// Calculates the polar decomposition F = R U using the eigenvalues of C = Fᵀ · F
///
/// ```text
/// C = Fᵀ · F
/// C = Σ λₖ Pₖ          (λ and P are the eigenvalues and eigenprojectors of C)
/// U = Σ √λₖ Pₖ
/// R = F · U⁻¹
/// ```
///
/// # Output
///
/// * `rr` -- the rotation tensor R
/// * `uu` -- the right stretch tensor U
///
/// # Input
///
/// * `ff` -- the deformation gradient F
///
/// # Errors
///
/// Returns an error if U cannot be inverted.
pub(crate) fn polar_decomp_eigen(rr: &mut Tensor2<9>, uu: &mut Tensor2<6>, ff: &Tensor2<9>) -> Result<(), StrError> {
    /* (*Mathematica:*)
    PolarDecEigen[ff_] := Module[{cc, vals, vecs, uuPrinc, uu, rr},
       cc = Transpose[ff] . ff;
       {vals, vecs} = Eigensystem[N[cc]];
       uuPrinc = DiagonalMatrix[Sqrt[vals]];
       uu = Transpose[vecs] . uuPrinc . vecs;
       rr = ff . Inverse[uu];
       {rr, uu}];
    */
    // F as a 3x3 matrix (stack)
    let mut f = [[0.0; 3]; 3];
    ff.to_std_matrix_slice(&mut f);

    // C = Fᵀ · F (stack)
    let mut cc = [[0.0; 3]; 3];
    small_mat_t_mat_mul(&mut cc, 1.0, &f, &f, 0.0, 3);

    // eigen-decomposition of C: C = Σ λₖ Pₖ
    let c = Tensor2::<6>::from_std_matrix(&cc)?;
    let mut spec = Spectral2::new();
    spec.decompose(&c)?;

    // U = Σ √λₖ Pₖ
    let sqrt_l0 = spec.lam[0].sqrt();
    let sqrt_l1 = spec.lam[1].sqrt();
    let sqrt_l2 = spec.lam[2].sqrt();
    for m in 0..6 {
        uu.vec[m] = sqrt_l0 * spec.proj[0].vec[m] + sqrt_l1 * spec.proj[1].vec[m] + sqrt_l2 * spec.proj[2].vec[m];
    }

    // R = F · U⁻¹ (stack)
    let mut u3 = [[0.0; 3]; 3];
    uu.to_std_matrix_slice(&mut u3);
    let mut ui = [[0.0; 3]; 3];
    small_mat_inv(&mut ui, &u3, 3)?;
    let mut r = [[0.0; 3]; 3];
    small_mat_mat_mul(&mut r, 1.0, &f, &ui, 0.0, 3);

    // set the rotation output tensor
    rr.set_std_matrix(&r)?;
    Ok(())
}

/// Calculates the polar decomposition F = R U using the singular value decomposition
///
/// ```text
/// F = P · D · Qᵀ     (SVD)
/// U = Q · D · Qᵀ
/// R = P · Qᵀ
/// ```
///
/// # Output
///
/// * `rr` -- the rotation tensor R
/// * `uu` -- the right stretch tensor U
///
/// # Input
///
/// * `ff` -- the deformation gradient F
pub(crate) fn polar_decomp_svd(rr: &mut Tensor2<9>, uu: &mut Tensor2<6>, ff: &Tensor2<9>) -> Result<(), StrError> {
    /* (*Mathematica:*)
    PolarDecSVD[ff_] := Module[{pp, dd, qq, uu, rr},
       {pp, dd, qq} = SingularValueDecomposition[N[ff]];(* F = P.D.Q^T *)
       uu = qq . dd . Transpose[qq];
       rr = pp . Transpose[qq];
       {rr, uu}];
    */
    // check

    // SVD: F = P · D · Qᵀ (a is overwritten by dgesvd)
    let mut a = ff.as_std_matrix();
    let mut s = Vector::new(3);
    let mut p = Matrix::new(3, 3);
    let mut qt = Matrix::new(3, 3); // Qᵀ (i.e., Vᵀ)
    mat_svd(&mut s, &mut p, &mut qt, &mut a)?;

    // U = Q · D · Qᵀ = V · D · Vᵀ (compute the upper triangle and mirror to guarantee symmetry)
    let mut u = Matrix::new(3, 3);
    for i in 0..3 {
        for j in i..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += qt.get(k, i) * s[k] * qt.get(k, j);
            }
            u.set(i, j, sum);
            u.set(j, i, sum);
        }
    }

    // R = P · Qᵀ
    let mut r = Matrix::new(3, 3);
    mat_mat_mul(&mut r, 1.0, &p, &qt, 0.0)?;

    // set the output tensors
    rr.set_std_matrix(&r)?;
    uu.set_std_matrix(&u)?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{polar_decomp_eigen, polar_decomp_svd};
    use crate::Tensor2;
    use russell_lab::{Matrix, mat_approx_eq, mat_mat_mul, mat_t_mat_mul};

    const TOL: f64 = 1e-12;

    fn check_polar(ff: &Tensor2<9>, rr: &Tensor2<9>, uu: &Tensor2<6>, tol: f64) {
        // check that F = R · U
        let f = ff.as_std_matrix();
        let r = rr.as_std_matrix();
        let u = uu.as_std_matrix();
        let mut ru = Matrix::new(3, 3);
        mat_mat_mul(&mut ru, 1.0, &r, &u, 0.0).unwrap();
        mat_approx_eq(&ru, &f, tol);

        // check that R is orthogonal (Rᵀ · R = I)
        let mut rtr = Matrix::new(3, 3);
        mat_t_mat_mul(&mut rtr, 1.0, &r, &r, 0.0).unwrap();
        mat_approx_eq(&rtr, &Matrix::identity(3), tol);

        // check that U is symmetric (U = Uᵀ)
        for i in 0..3 {
            for j in 0..3 {
                assert!((u.get(i, j) - u.get(j, i)).abs() < tol);
            }
        }
    }

    fn check_ref(ff: &Tensor2<9>, r_ref: &[[f64; 3]; 3], u_ref: &[[f64; 3]; 3], tol: f64) {
        // eigen
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        polar_decomp_eigen(&mut rr, &mut uu, ff).unwrap();
        check_polar(ff, &rr, &uu, tol);
        mat_approx_eq(&rr.as_std_matrix(), r_ref, tol);
        mat_approx_eq(&uu.as_std_matrix(), u_ref, tol);

        // svd
        let mut rr = Tensor2::<9>::new();
        let mut uu = Tensor2::<6>::new();
        polar_decomp_svd(&mut rr, &mut uu, ff).unwrap();
        check_polar(ff, &rr, &uu, tol);
        mat_approx_eq(&rr.as_std_matrix(), r_ref, tol);
        mat_approx_eq(&uu.as_std_matrix(), u_ref, tol);
    }

    // Python reference (numpy + scipy):
    // ```python
    // import numpy as np
    // from scipy import linalg
    //
    // def polar_eig(F):              # eigenvalues of C = Fᵀ·F
    //     C = F.T @ F
    //     lam, V = np.linalg.eigh(C)  # ascending eigenvalues; columns = eigenvectors
    //     U = (V * np.sqrt(lam)) @ V.T
    //     R = F @ np.linalg.inv(U)
    //     return R, U
    //
    // def polar_svd(F):              # singular value decomposition
    //     P, s, Qh = np.linalg.svd(F)  # F = P · diag(s) · Qh
    //     U = (Qh.T * s) @ Qh
    //     R = P @ Qh
    //     return R, U
    //
    // R, U = polar_eig(F)   # (or polar_svd(F); they agree to ~1e-15)
    // ```
    #[test]
    fn polar_decomp_classic_works() {
        // well-conditioned general matrix
        #[rustfmt::skip]
        let ff = Tensor2::<9>::from_std_matrix(&[
            [ 1.0,  0.495,  0.5  ],
            [-0.333, 1.0,   -0.247],
            [ 0.959, 0.0,    1.5  ],
        ]).unwrap();
        #[rustfmt::skip]
        let r_ref = [
            [ 0.9143288659766733,  0.3769304925214390, -0.1480747400786342],
            [-0.3738918867839085,  0.9261806148757240,  0.0489318467421089],
            [ 0.1555878589060806,  0.0106241440111795,  0.9877649243241274],
        ];
        #[rustfmt::skip]
        let u_ref = [
            [1.1880436209666460, 0.0787009018745443,  0.7828975173830820],
            [0.0787009018745443, 1.1127612086738370, -0.0243651495968139],
            [0.7828975173830820, -0.0243651495968139, 1.3955238503015750],
        ];
        check_ref(&ff, &r_ref, &u_ref, TOL);

        // general matrix (all components distinct)
        #[rustfmt::skip]
        let ff = Tensor2::<9>::from_std_matrix(&[
            [2.0, 1.0,   0.5],
            [0.3, 3.0,   1.0],
            [0.7, -0.2,  2.5],
        ]).unwrap();
        #[rustfmt::skip]
        let r_ref = [
            [ 0.9812167785042613,  0.1797641567869235, -0.0699891528481767],
            [-0.1577488954874382,  0.9565376734603720,  0.2452569371567530],
            [ 0.1110356679369841, -0.2296095102248696,  0.9669284116521154],
        ];
        #[rustfmt::skip]
        let u_ref = [
            [1.9928338559181810, 0.4857629584545492, 0.6104486636071538],
            [0.4857629584545494, 3.0952990792130150, 0.4723959762916591],
            [0.6104486636071539, 0.4723959762916588, 2.6275833898629540],
        ];
        check_ref(&ff, &r_ref, &u_ref, TOL);

        // reflection (det < 0): R has det = -1
        #[rustfmt::skip]
        let ff = Tensor2::<9>::from_std_matrix(&[
            [1.0,  0.5,  0.0],
            [0.2,  1.0,  0.3],
            [0.0, -0.4, -1.0],
        ]).unwrap();
        #[rustfmt::skip]
        let r_ref = [
            [ 0.9883141202500686,  0.1508649447384873, -0.0217937643234417],
            [-0.1517146219954520,  0.9874085229738411, -0.0448004713300558],
            [-0.0147605280091824, -0.0475833711255416, -0.9987582037736755],
        ];
        #[rustfmt::skip]
        let u_ref = [
            [ 0.9579711958509785,  0.3483466493332560, -0.0307538585894529],
            [ 0.3483466493332560,  1.0818743437933010,  0.3438059280176936],
            [-0.0307538585894528,  0.3438059280176936,  0.9853180623746591],
        ];
        check_ref(&ff, &r_ref, &u_ref, TOL);
    }
}
