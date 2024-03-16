use num_complex::Complex64;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // matrix
    let data = [
        [cpx!(1.0, 1.0), cpx!(2.0, -1.0), cpx!(3.0, 0.0)],
        [cpx!(2.0, -1.0), cpx!(4.0, 1.0), cpx!(5.0, -1.0)],
        [cpx!(3.0, 0.0), cpx!(5.0, -1.0), cpx!(6.0, 1.0)],
    ];

    let mut a = ComplexMatrix::from(&data);
    let a_copy = ComplexMatrix::from(&data);

    // allocate output data
    let (m, n) = a.dims();
    let min_mn = if m < n { m } else { n };
    let mut s = Vector::new(min_mn);
    let mut u = ComplexMatrix::new(m, m);
    let mut vh = ComplexMatrix::new(n, n);

    // calculate SVD
    complex_mat_svd(&mut s, &mut u, &mut vh, &mut a).unwrap();

    // check SVD
    let mut usv = ComplexMatrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..min_mn {
                usv.add(i, j, u.get(i, k) * s[k] * vh.get(k, j));
            }
        }
    }
    complex_mat_approx_eq(&usv, &a_copy, 1e-14);
    Ok(())
}
