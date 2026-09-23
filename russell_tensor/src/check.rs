use super::Tensor2;
use russell_lab::{AsArray1D, array_approx_eq};

/// Panics if a Tensor2 and an array containing KELVIN-MANDEL components are not approximately equal to each other
///
/// The comparison is made against the Kelvin-Mandel vector of the tensor.
///
/// # Panics
///
/// 1. Will panic if the dimensions are different
/// 2. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 3. Will panic if the absolute difference of components is greater than the tolerance
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
///     let v = [3.0, 2.0, 0.0, 0.0, 0.0, 0.0];
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
///     let v = [4.0, 2.0, 0.0, 0.0, 0.0, 0.0];
///     t2_approx_eq(&u, &v, 1e-6);
/// }
/// ```
pub fn t2_approx_eq<'a, const N: usize, T>(u: &Tensor2<N>, v: &'a T, tol: f64)
where
    T: AsArray1D<'a, f64>,
{
    array_approx_eq(u.as_vec(), v.as_slice(), tol);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::t2_approx_eq;
    use crate::Tensor2;

    #[test]
    fn t2_approx_eq_works() {
        let mut u = Tensor2::<6>::new();
        u.set(0, 1.0);
        u.set(1, 2.0);
        u.set(2, 3.0);
        u.set(3, 4.0);
        u.set(4, 5.0);
        u.set(5, 6.0);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.01];
        t2_approx_eq(&u, &v, 0.011);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff[5] =")]
    fn t2_approx_eq_panics() {
        let mut u = Tensor2::<6>::new();
        u.set(0, 1.0);
        u.set(1, 2.0);
        u.set(2, 3.0);
        u.set(3, 4.0);
        u.set(4, 5.0);
        u.set(5, 6.0);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.01];
        t2_approx_eq(&u, &v, 0.009);
    }
}
