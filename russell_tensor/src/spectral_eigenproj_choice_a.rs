/// Computes the three eigenprojectors of a symmetric 3x3 matrix
/// using the deviatoric polynomial expansion.
///
/// `s`: Deviatoric tensor S
/// `j2`: Second invariant of S
/// `lambda_s`: Eigenvalues of S (sorted or unsorted)
pub fn compute_eigenprojectors_choice_a(s: &[[f64; 3]; 3], j2: f64, lambda_s: &[f64; 3]) -> [[[f64; 3]; 3]; 3] {
    // Isotropic/Spherical state limit
    // When J2 approaches zero, any orthogonal basis is valid.
    // We return standard Cartesian dyads to prevent division by zero.
    if j2 < 1e-14 {
        return [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ];
    }

    // Compute S^2 manually to avoid allocating matrix structures
    let mut s2 = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s2[i][j] = s[i][0] * s[0][j] + s[i][1] * s[1][j] + s[i][2] * s[2][j];
        }
    }

    let mut projectors = [[[0.0; 3]; 3]; 3];
    let mut computed = [false; 3];

    // Scale the tolerance by J2 to maintain robustness across physical unit scales (e.g., Pa vs MPa)
    let den_tol = 1e-12 * j2;

    for k in 0..3 {
        let sk = lambda_s[k];
        let den = 3.0 * sk * sk - j2;

        // If denominator is safely away from zero, compute the projector directly
        if den.abs() > den_tol {
            let c0 = (sk * sk - j2) / den;
            let c1 = sk / den;
            let c2 = 1.0 / den;

            for i in 0..3 {
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    projectors[k][i][j] = c0 * delta + c1 * s[i][j] + c2 * s2[i][j];
                }
            }
            computed[k] = true;
        }
    }

    // Coalescing limit (Lode angle approaches +/- 30 degrees).
    // Exactly one projector (the distinct root) will have been computed.
    // The combined eigenspace of the two repeated roots is exactly (I - E_distinct).
    if !computed[0] || !computed[1] || !computed[2] {
        // Locate the index of the successfully computed distinct projector
        let distinct_idx = computed.iter().position(|&x| x).unwrap();

        let mut remainder = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                remainder[i][j] = delta - projectors[distinct_idx][i][j];
            }
        }

        // To generate two mutually orthogonal rank-1 projectors from this 2D space,
        // we extract the column of the remainder matrix with the maximum norm.
        let mut max_col = 0;
        let mut max_norm = 0.0;
        for col in 0..3 {
            let norm_sq = remainder[0][col] * remainder[0][col]
                + remainder[1][col] * remainder[1][col]
                + remainder[2][col] * remainder[2][col];
            if norm_sq > max_norm {
                max_norm = norm_sq;
                max_col = col;
            }
        }

        // Normalize the chosen vector (v1)
        let inv_norm = 1.0 / max_norm.sqrt();
        let v1 = [
            remainder[0][max_col] * inv_norm,
            remainder[1][max_col] * inv_norm,
            remainder[2][max_col] * inv_norm,
        ];

        // Form the first repeated-root projector E_a = v1 ⊗ v1
        // Form the second repeated-root projector E_b = Remainder - E_a
        let mut e_a = [[0.0; 3]; 3];
        let mut e_b = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                e_a[i][j] = v1[i] * v1[j];
                e_b[i][j] = remainder[i][j] - e_a[i][j];
            }
        }

        // Assign the extracted projectors to the two uncomputed eigenvalues
        let mut uncomputed_iter = computed.iter().enumerate().filter(|(_, c)| !**c).map(|(i, _)| i);
        if let (Some(idx1), Some(idx2)) = (uncomputed_iter.next(), uncomputed_iter.next()) {
            projectors[idx1] = e_a;
            projectors[idx2] = e_b;
        }
    }

    projectors
}
