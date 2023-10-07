use russell_lab::*;

fn main() -> Result<(), StrError> {
    // data for matrix "A"
    #[rustfmt::skip]
    let data = [
        [ 1.96, -6.49, -0.47, -7.20, -0.65],
        [-6.49,  3.80, -6.39,  1.50, -6.34],
        [-0.47, -6.39,  4.17, -1.51,  2.67],
        [-7.20,  1.50, -1.51,  5.70,  1.80],
        [-0.65, -6.34,  2.67,  1.80, -7.10],
    ];

    // lower and upper parts
    let mut a_lower = Matrix::from_lower(&data)?;
    let mut a_upper = Matrix::from_upper(&data)?;

    // allocate space for the eigenvalues
    let m = a_lower.nrow();
    let mut l_lower = Vector::new(m);
    let mut l_upper = Vector::new(m);

    // compute eigen-{values,vectors}
    mat_eigen_sym(&mut l_lower, &mut a_lower, false)?;
    mat_eigen_sym(&mut l_upper, &mut a_upper, true)?;

    // dsyev(true, true, n, &mut a_full, &mut w_full)?;
    // dsyev(true, true, n, &mut a_upper, &mut w_upper)?;
    // dsyev(true, false, n, &mut a_lower, &mut w_lower)?;

    // check eigenvalues
    let l_correct = &[
        -11.065575263268386,
        -6.228746932398536,
        0.864027975272061,
        8.865457108365517,
        16.094837112029339,
    ];
    vec_approx_eq(l_lower.as_data(), l_correct, 1e-14);
    vec_approx_eq(l_upper.as_data(), l_correct, 1e-14);

    // check eigen-decomposition
    // A ⋅ v[col] = λ[col] ⋅ v[col]
    let a = Matrix::from(&data);
    check_eigen_sym(&a, &l_lower, &a_lower, 1e-14);
    check_eigen_sym(&a, &l_upper, &a_upper, 1e-14);
    Ok(())
}

// Checks eigenvalues and eigenvectors of a symmetric matrix
//
// ```text
// col is the column-index in the v_matrix, 0 ≤ col < n
//
// A ⋅ v[col] = λ[col] ⋅ v[col]
//
// error_i = | (A⋅v[col])_i - (λ[col]⋅v[col])_i |
// ```
fn check_eigen_sym(a: &Matrix, lambda: &Vector, v_matrix: &Matrix, tol: f64) {
    let (m, n) = a.dims();
    assert_eq!(m, n);
    for col in 0..n {
        for i in 0..n {
            let mut a_times_v_i = 0.0;
            for k in 0..n {
                a_times_v_i += a.get(i, k) * v_matrix.get(k, col);
            }
            let error = f64::abs(a_times_v_i - lambda[col] * v_matrix.get(i, col));
            if error > tol {
                panic!(
                    "A ⋅ v[{}] = λ[{}] ⋅ v[{}] failed at index {}. error = {:?}",
                    col, col, col, i, error
                );
            }
        }
    }
}
