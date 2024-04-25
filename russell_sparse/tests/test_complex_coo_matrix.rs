use russell_lab::*;
use russell_sparse::prelude::*;

#[test]
fn test_complex_coo_matrix() -> Result<(), StrError> {
    let mut coo = ComplexCooMatrix::new(3, 3, 4, Sym::YesLower)?;
    coo.put(0, 0, cpx!(1.0, 0.1))?;
    coo.put(1, 0, cpx!(2.0, 0.2))?;
    coo.put(1, 1, cpx!(3.0, 0.3))?;
    coo.put(2, 1, cpx!(4.0, 0.4))?;
    let mut a = ComplexMatrix::new(3, 3);
    coo.to_dense(&mut a).unwrap();
    let correct = "┌                      ┐\n\
                   │ 1+0.1i 2+0.2i   0+0i │\n\
                   │ 2+0.2i 3+0.3i 4+0.4i │\n\
                   │   0+0i 4+0.4i   0+0i │\n\
                   └                      ┘";
    assert_eq!(format!("{}", a), correct);

    let u = ComplexVector::from(&[cpx!(3.0, 0.0), cpx!(2.0, 2.0), cpx!(1.0, 0.0)]);
    let mut v = ComplexVector::new(3);
    coo.mat_vec_mul(&mut v, cpx!(1.0, 0.0), &u)?;
    complex_vec_approx_eq(&v, &[cpx!(6.6, 4.7), cpx!(15.4, 7.6), cpx!(7.2, 8.8)], 1e-15);

    coo.mat_vec_mul(&mut v, cpx!(1.0, 2.0), &u)?;
    complex_vec_approx_eq(&v, &[cpx!(-2.8, 17.9), cpx!(0.2, 38.4), cpx!(-10.4, 23.2)], 1e-14);
    Ok(())
}
