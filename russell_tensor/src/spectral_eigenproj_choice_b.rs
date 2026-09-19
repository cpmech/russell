pub fn compute_eigenprojectors_choice_b(
    s_mat: &[[f64; 3]; 3],
    scale: f64,
    j2: f64,
    lambda_s: &[f64; 3], // Must be sorted (e.g., descending: s1 >= s2 >= s3)
) -> [[[f64; 3]; 3]; 3] {
    let i_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let zero_mat = [[0.0; 3]; 3];

    // Precompute S^2 for the deviatoric polynomial E_k = c0*I + c1*S + c2*S^2
    let mut s2_mat = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                s2_mat[i][j] += s_mat[i][k] * s_mat[k][j];
            }
        }
    }

    // Evaluates the continuous polynomial projector for a strictly distinct eigenvalue
    let eval_projector = |s_k: f64| -> [[f64; 3]; 3] {
        let den = 3.0 * s_k * s_k - j2;
        let c0 = (s_k * s_k - j2) / den;
        let c1 = s_k / den;
        let c2 = 1.0 / den;

        let mut proj = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                proj[i][j] = c0 * i_mat[i][j] + c1 * s_mat[i][j] + c2 * s2_mat[i][j];
            }
        }
        proj
    };

    let diff12 = (lambda_s[0] - lambda_s[1]).abs();
    let diff23 = (lambda_s[1] - lambda_s[2]).abs();
    // let larger = if diff12 > diff23 { diff12 } else { diff23 };
    let tol_jj2 = 10.0 * f64::EPSILON.sqrt() * j2;
    let tol_diff = 10.0 * f64::EPSILON.sqrt() * scale;
    println!(
        "diff12 = {:.5e}, diff23 = {:.5e}, tol_jj2 = {:.5e}, tol_diff = {:.5e}",
        diff12, diff23, tol_jj2, tol_diff
    );

    if j2 < tol_jj2 || j2 <= f64::EPSILON {
        println!("Case 1");
        // Case 1: Purely spherical (J2 approaches 0).
        // Entire space is the eigenspace. E1 gets the Identity; E2 and E3 are empty.
        [i_mat, zero_mat, zero_mat]
    } else if diff12 < tol_diff {
        println!("Case 2");
        // Case 2: lambda_1 ≈ lambda_2 != lambda_3
        // Compute projector for the distinct eigenvalue (s3)
        let e3 = eval_projector(lambda_s[2]);

        // Use partition of unity for the repeated roots
        let mut e12 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                e12[i][j] = i_mat[i][j] - e3[i][j];
            }
        }
        // Assign the shared plane's projector to E1, zero to E2
        [e12, zero_mat, e3]
    } else if diff23 < tol_diff {
        println!("Case 3");
        // Case 3: lambda_1 != lambda_2 ≈ lambda_3
        // Compute projector for the distinct eigenvalue (s1)
        let e1 = eval_projector(lambda_s[0]);

        let mut e23 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                e23[i][j] = i_mat[i][j] - e1[i][j];
            }
        }
        // Assign the shared plane's projector to E2, zero to E3
        [e1, e23, zero_mat]
    } else {
        println!("Case 4");
        // Case 4: Three distinct eigenvalues
        let e1 = eval_projector(lambda_s[0]);
        let e2 = eval_projector(lambda_s[1]);

        // Calculate the third projector purely via partition of unity to avoid
        // accumulating floating-point errors from a third polynomial evaluation.
        let mut e3 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                e3[i][j] = i_mat[i][j] - e1[i][j] - e2[i][j];
            }
        }
        [e1, e2, e3]
    }
}
