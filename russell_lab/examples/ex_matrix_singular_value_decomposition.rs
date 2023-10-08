use russell_lab::*;

fn main() -> Result<(), StrError> {
    // matrix
    let s33 = math::SQRT_3 / 3.0;
    #[rustfmt::skip]
    let mut a = Matrix::from(&[
        [-s33, -s33, 1.0],
        [ s33, -s33, 1.0],
        [-s33,  s33, 1.0],
        [ s33,  s33, 1.0],
    ]);
    let a_copy = a.clone();

    // allocate output structures
    let (m, n) = a.dims();
    let min_mn = if m < n { m } else { n };
    let mut s = Vector::new(min_mn);
    let mut u = Matrix::new(m, m);
    let mut vt = Matrix::new(n, n);

    // perform SVD
    mat_svd(&mut s, &mut u, &mut vt, &mut a)?;

    // check
    let s_correct = &[2.0, 2.0 / math::SQRT_3, 2.0 / math::SQRT_3];
    vec_approx_eq(s.as_data(), s_correct, 1e-14);

    // check SVD: a == u * s * vt
    let mut usv = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..min_mn {
                usv.add(i, j, u.get(i, k) * s[k] * vt.get(k, j));
            }
        }
    }
    mat_approx_eq(&usv, &a_copy, 1e-14);
    Ok(())
}
