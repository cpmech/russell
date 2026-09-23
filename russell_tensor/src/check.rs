use super::{Tensor1, Tensor2, Tensor3, Tensor4};

/// Panics if two Tensor1 are not approximately equal to each other
///
/// The comparison is made component-wise on the standard vector.
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_tensor::{t1_approx_eq, Tensor1};
///
/// fn main() {
///     let u = Tensor1::from(&[3.0000001, 2.0, 0.0]);
///     let v = Tensor1::from(&[3.0, 2.0, 0.0]);
///     t1_approx_eq(&u, &v, 1e-6);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_tensor::{t1_approx_eq, Tensor1};
///
/// fn main() {
///     let u = Tensor1::from(&[3.0000001, 2.0, 0.0]);
///     let v = Tensor1::from(&[4.0, 2.0, 0.0]);
///     t1_approx_eq(&u, &v, 1e-6);
/// }
/// ```
pub fn t1_approx_eq(u: &Tensor1, v: &Tensor1, tol: f64) {
    for m in 0..3 {
        let diff = f64::abs(u.vec[m] - v.vec[m]);
        if diff.is_nan() {
            panic!("t1_approx_eq found NaN");
        }
        if diff.is_infinite() {
            panic!("t1_approx_eq found Inf");
        }
        if diff > tol {
            panic!(
                "first-order tensors are not approximately equal. @ ({}) diff = {:?}",
                m, diff
            );
        }
    }
}

/// Panics if two Tensor2 are not approximately equal to each other
///
/// The comparison is made component-wise on the Kelvin-Mandel vector.
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_tensor::{t2_approx_eq, Tensor2};
///
/// fn main() {
///     let mut u = Tensor2::<6>::new();
///     u.set(0, 3.0000001);
///     u.set(1, 2.0);
///     let mut v = Tensor2::<6>::new();
///     v.set(0, 3.0);
///     v.set(1, 2.0);
///     t2_approx_eq(&u, &v, 1e-6);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_tensor::{t2_approx_eq, Tensor2};
///
/// fn main() {
///     let mut u = Tensor2::<6>::new();
///     u.set(0, 3.0000001);
///     u.set(1, 2.0);
///     let mut v = Tensor2::<6>::new();
///     v.set(0, 4.0);
///     v.set(1, 2.0);
///     t2_approx_eq(&u, &v, 1e-6);
/// }
/// ```
pub fn t2_approx_eq<const N: usize>(u: &Tensor2<N>, v: &Tensor2<N>, tol: f64) {
    for m in 0..N {
        let diff = f64::abs(u.vec[m] - v.vec[m]);
        if diff.is_nan() {
            panic!("t2_approx_eq found NaN");
        }
        if diff.is_infinite() {
            panic!("t2_approx_eq found Inf");
        }
        if diff > tol {
            panic!(
                "second-order tensors are not approximately equal. @ ({}) diff = {:?}",
                m, diff
            );
        }
    }
}

/// Panics if two Tensor3 are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_tensor::{t3_approx_eq, Tensor3};
///
/// fn main() {
///     let mut a = Tensor3::<6, 3>::new();
///     a.set(0, 0, 1.0);
///     let mut b = Tensor3::<6, 3>::new();
///     b.set(0, 0, 1.01);
///     t3_approx_eq(&a, &b, 0.011);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_tensor::{t3_approx_eq, Tensor3};
///
/// fn main() {
///     let mut a = Tensor3::<6, 3>::new();
///     a.set(0, 0, 1.0);
///     let mut b = Tensor3::<6, 3>::new();
///     b.set(0, 0, 1.01);
///     t3_approx_eq(&a, &b, 0.001);
/// }
/// ```
pub fn t3_approx_eq<const M: usize, const N: usize>(u: &Tensor3<M, N>, v: &Tensor3<M, N>, tol: f64) {
    for i in 0..M {
        for j in 0..N {
            let diff = f64::abs(u.mat[i][j] - v.mat[i][j]);
            if diff.is_nan() {
                panic!("t3_approx_eq found NaN");
            }
            if diff.is_infinite() {
                panic!("t3_approx_eq found Inf");
            }
            if diff > tol {
                panic!(
                    "third-order tensors are not approximately equal. @ ({},{}) diff = {:?}",
                    i, j, diff
                );
            }
        }
    }
}

/// Panics if two Tensor4 are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_tensor::{t4_approx_eq, Tensor4};
///
/// fn main() {
///     let mut a = Tensor4::<9>::new();
///     a.set(0, 0, 1.0);
///     let mut b = Tensor4::<9>::new();
///     b.set(0, 0, 1.01);
///     t4_approx_eq(&a, &b, 0.011);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_tensor::{t4_approx_eq, Tensor4};
///
/// fn main() {
///     let mut a = Tensor4::<9>::new();
///     a.set(0, 0, 1.0);
///     let mut b = Tensor4::<9>::new();
///     b.set(0, 0, 1.01);
///     t4_approx_eq(&a, &b, 0.001);
/// }
/// ```
pub fn t4_approx_eq<const N: usize>(u: &Tensor4<N>, v: &Tensor4<N>, tol: f64) {
    for i in 0..N {
        for j in 0..N {
            let diff = f64::abs(u.mat[i][j] - v.mat[i][j]);
            if diff.is_nan() {
                panic!("t4_approx_eq found NaN");
            }
            if diff.is_infinite() {
                panic!("t4_approx_eq found Inf");
            }
            if diff > tol {
                panic!(
                    "fourth-order tensors are not approximately equal. @ ({},{}) diff = {:?}",
                    i, j, diff
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{t1_approx_eq, t2_approx_eq, t3_approx_eq, t4_approx_eq};
    use crate::{Tensor1, Tensor2, Tensor3, Tensor4};

    #[test]
    fn t1_approx_eq_works() {
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        let v = Tensor1::from(&[1.0, 2.0, 3.01]);
        t1_approx_eq(&u, &v, 0.011);
    }

    #[test]
    #[should_panic(expected = "first-order tensors are not approximately equal. @ (2) diff =")]
    fn t1_approx_eq_panics() {
        let u = Tensor1::from(&[1.0, 2.0, 3.0]);
        let v = Tensor1::from(&[1.0, 2.0, 3.01]);
        t1_approx_eq(&u, &v, 0.009);
    }

    #[test]
    fn t2_approx_eq_works() {
        let mut u = Tensor2::<6>::new();
        u.set(0, 1.0);
        u.set(1, 2.0);
        u.set(2, 3.0);
        u.set(3, 4.0);
        u.set(4, 5.0);
        u.set(5, 6.0);
        let mut v = Tensor2::<6>::new();
        v.set(0, 1.0);
        v.set(1, 2.0);
        v.set(2, 3.0);
        v.set(3, 4.0);
        v.set(4, 5.0);
        v.set(5, 6.01);
        t2_approx_eq(&u, &v, 0.011);
    }

    #[test]
    #[should_panic(expected = "second-order tensors are not approximately equal. @ (5) diff =")]
    fn t2_approx_eq_panics() {
        let mut u = Tensor2::<6>::new();
        u.set(0, 1.0);
        u.set(1, 2.0);
        u.set(2, 3.0);
        u.set(3, 4.0);
        u.set(4, 5.0);
        u.set(5, 6.0);
        let mut v = Tensor2::<6>::new();
        v.set(0, 1.0);
        v.set(1, 2.0);
        v.set(2, 3.0);
        v.set(3, 4.0);
        v.set(4, 5.0);
        v.set(5, 6.01);
        t2_approx_eq(&u, &v, 0.009);
    }

    #[test]
    fn t3_approx_eq_works() {
        let mut a = Tensor3::<6, 3>::new();
        a.set(0, 0, 1.0);
        a.set(5, 2, 2.0);
        let mut b = Tensor3::<6, 3>::new();
        b.set(0, 0, 1.01);
        b.set(5, 2, 2.0);
        t3_approx_eq(&a, &b, 0.011);
    }

    #[test]
    #[should_panic(expected = "third-order tensors are not approximately equal. @ (0,0) diff =")]
    fn t3_approx_eq_panics() {
        let mut a = Tensor3::<6, 3>::new();
        a.set(0, 0, 1.0);
        let mut b = Tensor3::<6, 3>::new();
        b.set(0, 0, 1.01);
        t3_approx_eq(&a, &b, 0.001);
    }

    #[test]
    fn t4_approx_eq_works() {
        let mut a = Tensor4::<9>::new();
        a.set(0, 0, 1.0);
        a.set(8, 8, 2.0);
        let mut b = Tensor4::<9>::new();
        b.set(0, 0, 1.01);
        b.set(8, 8, 2.0);
        t4_approx_eq(&a, &b, 0.011);
    }

    #[test]
    #[should_panic(expected = "fourth-order tensors are not approximately equal. @ (0,0) diff =")]
    fn t4_approx_eq_panics() {
        let mut a = Tensor4::<9>::new();
        a.set(0, 0, 1.0);
        let mut b = Tensor4::<9>::new();
        b.set(0, 0, 1.01);
        t4_approx_eq(&a, &b, 0.001);
    }
}
