use crate::math::{chebyshev_lobatto_points_standard, chebyshev_tn, PI};
use crate::StrError;
use crate::Vector;

/// Defines the tolerance to make sure that the range [xa, xb] is not zero
pub const TOL_RANGE: f64 = 1.0e-5;

/// Implements the Chebyshev interpolant and associated functions
///
/// The problem coordinates are `x ∈ [xa, xb]` and the grid coordinates are `z ∈ [-1, 1]`
/// Thus, consider the mapping:
///
/// ```text
///        2 x - xb - xa
/// z(x) = —————————————
///           xb - xa
/// ```
///
/// And
///
/// ```text
///        xb + xa + (xb - xa) z
/// x(z) = —————————————————————
///                 2
/// ```
///
/// The interpolated values are:
///
/// ```text
/// Uⱼ = f(Xⱼ(Zⱼ))
///
/// where xa ≤ Xⱼ ≤ xb
/// and   -1 ≤ Zⱼ ≤ 1
/// ```
///
/// # Notes
///
/// 1. This structure is meant for interpolating data and finding (all) the roots of an equation.
///    On the other hand, [crate::InterpLagrange] is meant for implementing spectral methods
///    for solving partial differential equations. Therefore, [crate::InterpLagrange] implements
///    the derivative matrices, whereas this structure does not.
/// 2. The [crate::InterpLagrange] renders the same results when using Chebyshev-Gauss-Lobatto points.
/// 3. Only Chebyshev-Gauss-Lobatto points are considered here.
/// 4. The Chebyshev-Gauss-Lobatto coordinates are sorted from +1 to -1 (as in Reference # 1).
///
/// # References
///
/// 1. Canuto C, Hussaini MY, Quarteroni A, Zang TA (2006) Spectral Methods: Fundamentals in
///    Single Domains. Springer. 563p
pub struct InterpChebyshev {
    /// Holds the polynomial degree N
    nn: usize,

    /// Holds the number of points (= N + 1)
    np: usize,

    /// Holds the lower bound
    xa: f64,

    /// Holds the upper bound
    xb: f64,

    /// Holds the difference xb - xa
    dx: f64,

    /// Holds the expansion coefficients (standard Chebyshev-Gauss-Lobatto)
    a: Vector,

    /// Holds the function evaluation at the standard (from 1 to -1) Chebyshev-Gauss-Lobatto points
    uu: Vector,

    /// Holds the constant y=c value for a zeroth-order function
    constant_fx: f64,

    /// Indicates that `a` and `uu` are set and ready for `eval`
    ready: bool,
}

impl InterpChebyshev {
    /// Returns the standard (from 1 to -1) Chebyshev-Gauss-Lobatto points
    ///
    /// # Input
    ///
    /// * `nn` -- polynomial degree N
    pub fn points(nn: usize) -> Vector {
        chebyshev_lobatto_points_standard(nn)
    }

    /// Allocates a new instance with uninitialized values
    ///
    /// **Important:** Make sure to call [InterpChebyshev::set_uu_value()] before
    /// calling [InterpChebyshev::eval()] to evaluate the interpolated function.
    ///
    /// # Input
    ///
    /// * `nn` -- polynomial degree N
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    pub fn new(nn: usize, xa: f64, xb: f64) -> Result<Self, StrError> {
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np = nn + 1;
        Ok(InterpChebyshev {
            nn,
            np,
            xa,
            xb,
            dx: xb - xa,
            a: Vector::new(np),
            uu: Vector::new(np),
            constant_fx: 0.0,
            ready: false,
        })
    }

    /// Sets a component of the U vector and computes the expansion coefficients
    ///
    /// **Note:** This function will compute the expansion coefficients when
    /// the last component is set; i.e., when `i == nn` (nn is the degree N).
    /// Therefore, it is recommended to call this function sequentially
    /// from 0 to N (N is available via [InterpChebyshev::get_degree()]).
    ///
    /// # Input
    ///
    /// * `i` -- the index of the Chebyshev-Gauss-Lobatto point in `[0, N]`
    ///   with `N` being the polynomial degree. The grid points can be obtained
    ///   using the [InterpChebyshev::points()] function.
    /// * `uui` -- the i-th component of the `U` vector; i.e., `Uᵢ`
    ///
    /// # Panics
    ///
    /// A panic will occur if `i` is out of range (it must be in `[0, N]`)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let nn = 2;
    ///     let (xa, xb) = (-4.0, 4.0);
    ///     let mut interp = InterpChebyshev::new(nn, xa, xb).unwrap();
    ///     let zz = InterpChebyshev::points(nn);
    ///     let dx = xb - xa;
    ///     let np = nn + 1;
    ///     for i in 0..np {
    ///         let x = (xb + xa + dx * zz[i]) / 2.0;
    ///         interp.set_uu_value(i, x * x - 1.0);
    ///     }
    ///     approx_eq(interp.eval(0.0)?, -1.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn set_uu_value(&mut self, i: usize, uui: f64) {
        self.uu[i] = uui;
        if i == self.nn {
            chebyshev_coefficients(self.a.as_mut_data(), self.uu.as_mut_data(), self.nn);
            self.ready = true;
        } else {
            self.ready = false;
        }
    }

    /// Allocates a new instance with given f(x) function
    ///
    /// # Input
    ///
    /// * `nn` -- polynomial degree N
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    /// * `args` -- extra arguments for the f(x) function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(1.0 / (1.0 + 16.0 * x * x));
    ///     let (xa, xb) = (-1.0, 1.0);
    ///
    ///     // interpolant
    ///     let degree = 10;
    ///     let args = &mut 0;
    ///     let interp = InterpChebyshev::new_with_f(degree, xa, xb, args, f)?;
    ///
    ///     // check
    ///     approx_eq(interp.eval(0.0).unwrap(), 1.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn new_with_f<F, A>(nn: usize, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<Self, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np = nn + 1;
        let mut interp = InterpChebyshev {
            nn,
            np,
            xa,
            xb,
            dx: xb - xa,
            a: Vector::new(np),
            uu: Vector::new(np),
            constant_fx: 0.0,
            ready: true,
        };
        if nn == 0 {
            interp.constant_fx = f((xa + xb) / 2.0, args)?;
        } else {
            chebyshev_data_vector(interp.uu.as_mut_data(), nn, xa, xb, args, &mut f)?;
            chebyshev_coefficients(interp.a.as_mut_data(), interp.uu.as_mut_data(), nn);
        }
        Ok(interp)
    }

    /// Allocates a new instance with given U vector (function evaluated at grid points)
    ///
    /// # Input
    ///
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    /// * `uu` -- the data vector such that `Uᵢ = f(xᵢ)`; i.e., the function evaluated at the
    ///   **standard** (from 1 to -1) Chebyshev-Gauss-Lobatto coordinates. These coordinates
    ///   are available via the [InterpChebyshev::points()] function.
    pub fn new_with_uu(xa: f64, xb: f64, uu: &[f64]) -> Result<Self, StrError> {
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np = uu.len();
        if np == 0 {
            return Err("the number of points = uu.len() must be ≥ 1");
        }
        let nn = np - 1;
        let mut interp = InterpChebyshev {
            nn,
            np,
            xa,
            xb,
            dx: xb - xa,
            a: Vector::new(np),
            uu: Vector::from(&uu),
            constant_fx: 0.0,
            ready: true,
        };
        if nn == 0 {
            interp.constant_fx = uu[0];
        } else {
            chebyshev_coefficients(interp.a.as_mut_data(), interp.uu.as_mut_data(), nn);
        }
        Ok(interp)
    }

    /// Allocates a new instance using adaptive interpolation
    ///
    /// # Input
    ///
    /// * `nn_max` -- maximum polynomial degree N (≤ 2048)
    /// * `tol` -- tolerance to truncate the Chebyshev series (e.g., 1e-8)
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    /// * `args` -- extra arguments for the f(x) function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Method
    ///
    /// A simple method is considered here, in which the polynomial degree N is increased
    /// linearly until the two last expansion coefficients of the Chebyshev series are small
    /// in absolute value. This termination strategy corresponds to the Battles and Trefethen
    /// line in Table 3.3 (page 55) of Reference # 1.
    ///
    /// # References
    ///
    /// 1. Boyd JP (2014) Solving Transcendental Equations: The Chebyshev Polynomial Proxy
    ///    and Other Numerical Rootfinders, Perturbation Series, and Oracles, SIAM, pp460
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
    ///     let (xa, xb) = (-1.0, 1.0);
    ///
    ///     // interpolant
    ///     let nn_max = 200;
    ///     let tol = 1e-8;
    ///     let args = &mut 0;
    ///     let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f)?;
    ///
    ///     // check
    ///     assert_eq!(interp.get_degree(), 2);
    ///     Ok(())
    /// }
    /// ```
    pub fn new_adapt<F, A>(nn_max: usize, tol: f64, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<Self, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if nn_max > 2048 {
            return Err("the maximum degree N must be ≤ 2048");
        }
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np_max = nn_max + 1;
        let mut work_a = vec![0.0; np_max];
        let mut work_uu = vec![0.0; np_max];
        let mut an_prev = 0.0;
        for nn in 1..=nn_max {
            chebyshev_data_vector(&mut work_uu, nn, xa, xb, args, &mut f)?;
            chebyshev_coefficients(&mut work_a, &work_uu, nn);
            let an = work_a[nn];
            if nn > 1 && f64::abs(an_prev) < tol && f64::abs(an) < tol {
                let nn_final = nn - 2; // -2 because the last two coefficients are zero
                return Ok(InterpChebyshev::new_with_f(nn_final, xa, xb, args, f)?);
            }
            an_prev = an;
        }
        Err("adaptive interpolation did not converge")
    }

    /// Evaluates the interpolated f(x) function
    ///
    /// This function uses the Clenshaw algorithm (Reference # 1) to
    /// avoid calling the trigonometric functions.
    ///
    /// # Reference
    ///
    /// 1. Clenshaw CW (1954) A note on the summation of Chebyshev series,
    ///    Mathematics of Computation, 9:118-120
    pub fn eval(&self, x: f64) -> Result<f64, StrError> {
        if !self.ready {
            return Err("all U components must be set first");
        }
        if self.nn == 0 {
            return Ok(self.constant_fx);
        }
        let z = f64::max(-1.0, f64::min(1.0, (2.0 * x - self.xb - self.xa) / self.dx));
        let z2 = z * 2.0;
        let mut b_k = 0.0;
        let mut b_k_plus_1 = 0.0;
        let mut b_k_plus_2: f64;
        for k in (1..self.np).rev() {
            b_k_plus_2 = b_k_plus_1;
            b_k_plus_1 = b_k;
            b_k = z2 * b_k_plus_1 - b_k_plus_2 + self.a[k];
        }
        let res = b_k * z - b_k_plus_1 + self.a[0];
        Ok(res)
    }

    /// Evaluates the interpolated f(x) function using the slower trigonometric functions
    pub fn eval_using_trig(&self, x: f64) -> Result<f64, StrError> {
        if !self.ready {
            return Err("all U components must be set first");
        }
        if self.nn == 0 {
            return Ok(self.constant_fx);
        }
        let z = f64::max(-1.0, f64::min(1.0, (2.0 * x - self.xb - self.xa) / self.dx));
        let mut sum = 0.0;
        for k in 0..self.np {
            sum += self.a[k] * chebyshev_tn(k, z);
        }
        Ok(sum)
    }

    /// Estimates the max error by comparing the interpolation against a reference solution
    ///
    /// **Note:** The error is calculated by the max abs difference between the interpolation
    /// and the reference (e.g., analytical) solution at `nstation` discrete points.
    ///
    /// # Input
    ///
    /// * `nstation` -- number of stations (e.g, `10_000`) to compute `max(|f(x) - I{f}(x)|)`
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Output
    ///
    /// * `err_f` -- is the max interpolation error in `[xa, xb]`
    pub fn estimate_max_error<F, A>(&self, nstation: usize, args: &mut A, mut f: F) -> Result<f64, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let mut err_f = 0.0;
        let stations = Vector::linspace(self.xa, self.xb, nstation).unwrap();
        for p in 0..nstation {
            let fi = self.eval(stations[p])?;
            err_f = f64::max(err_f, f64::abs(fi - f(stations[p], args)?));
        }
        Ok(err_f)
    }

    /// Returns the polynomial degree N
    pub fn get_degree(&self) -> usize {
        self.nn
    }

    /// Returns the range
    ///
    /// Returns `(xa, xb, dx)` with `dx = xb - xa`
    pub fn get_range(&self) -> (f64, f64, f64) {
        (self.xa, self.xb, self.dx)
    }

    /// Returns an access to the expansion coefficients (a)
    pub fn get_coefficients(&self) -> &Vector {
        &self.a
    }

    /// Returns the ready flag
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

/// Computes the data vector (function evaluations at Chebyshev-Gauss-Lobatto points)
fn chebyshev_data_vector<F, A>(
    work_uu: &mut [f64],
    nn: usize,
    xa: f64,
    xb: f64,
    args: &mut A,
    f: &mut F,
) -> Result<(), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    // check
    let np = nn + 1;
    assert!(nn > 0);
    assert!(xb > xa);
    assert!(work_uu.len() >= np);

    // data vector U
    let nf = nn as f64;
    let uu = &mut work_uu[0..np];
    for k in 0..np {
        let kf = k as f64;
        let x = (xb + xa + (xb - xa) * f64::cos(PI * kf / nf)) / 2.0;
        uu[k] = (*f)(x, args)?;
    }
    Ok(())
}

/// Computes the Chebyshev-Gauss-Lobatto coefficients
fn chebyshev_coefficients(work_a: &mut [f64], work_uu: &[f64], nn: usize) {
    // check
    let np = nn + 1;
    assert!(nn > 0);
    assert!(work_a.len() >= np);
    assert!(work_uu.len() >= np);

    // coefficients a
    let nf = nn as f64;
    let a = &mut work_a[0..np];
    let uu = &work_uu[0..np];
    for j in 0..np {
        let jf = j as f64;
        let qj = if j == 0 || j == nn { 2.0 } else { 1.0 };
        a[j] = 0.0;
        for k in 0..np {
            let kf = k as f64;
            let qk = if k == 0 || k == nn { 2.0 } else { 1.0 };
            a[j] += uu[k] * 2.0 * f64::cos(PI * jf * kf / nf) / (qj * qk * nf);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::InterpChebyshev;
    use crate::math::PI;
    use crate::{approx_eq, vec_approx_eq, NoArgs, Vector, TOL_RANGE};

    #[allow(unused)]
    use plotpy::{Curve, Legend, Plot};

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            InterpChebyshev::new(2, 0.0, 0.0).err(),
            Some("xb must be greater than xa + ϵ")
        );
    }

    #[test]
    fn new_works() {
        let nn = 2;
        let (xa, xb) = (-4.0, 4.0);
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        let np = nn + 1;
        assert_eq!(interp.xa, xa);
        assert_eq!(interp.xb, xb);
        assert_eq!(interp.dx, 8.0);
        assert_eq!(interp.nn, nn);
        assert_eq!(interp.np, np);
        assert_eq!(interp.a.dim(), np);
        assert_eq!(interp.uu.dim(), np);
        assert_eq!(interp.constant_fx, 0.0);
        assert_eq!(interp.ready, false);
    }

    #[test]
    fn new_with_f_captures_errors() {
        let f = |x: f64, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        assert_eq!(
            InterpChebyshev::new_with_f(2, 0.0, 0.0, args, f).err(),
            Some("xb must be greater than xa + ϵ")
        );
        let f = |_: f64, _: &mut NoArgs| Err("stop");
        assert_eq!(InterpChebyshev::new_with_f(0, 0.0, 1.0, args, f).err(), Some("stop"));
        assert_eq!(InterpChebyshev::new_with_f(1, 0.0, 1.0, args, f).err(), Some("stop"));
    }

    #[test]
    fn new_with_f_and_estimate_max_error_work() {
        let f = |x: f64, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);
        let args = &mut 0;
        // N = 2
        let nn = 2;
        let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
        let err = interp.estimate_max_error(100, args, f).unwrap();
        println!("N = 2, err = {:e}", err);
        assert!(err < 1e-14);
        // N = 1
        let nn = 1;
        let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
        let err = interp.estimate_max_error(100, args, f).unwrap();
        println!("N = 1, err = {:e}", err);
        assert!(err > 15.0);
    }

    #[test]
    fn new_with_uu_captures_errors() {
        let uu = Vector::new(0);
        assert_eq!(
            InterpChebyshev::new_with_uu(0.0, 0.0, uu.as_data()).err(),
            Some("xb must be greater than xa + ϵ")
        );
        assert_eq!(
            InterpChebyshev::new_with_uu(0.0, 1.0, uu.as_data()).err(),
            Some("the number of points = uu.len() must be ≥ 1")
        );
    }

    #[test]
    fn new_with_uu_works_n0() {
        let (xa, xb) = (0.0, 1.0);
        let uu = &[1.0];
        let interp = InterpChebyshev::new_with_uu(xa, xb, uu).unwrap();
        assert_eq!(interp.eval(3.0).unwrap(), 1.0);
    }

    #[test]
    fn new_with_uu_works() {
        let f = |x, _: &mut NoArgs| Ok(-1.0 + f64::sqrt(1.0 + 2.0 * x * 1200.0));

        let (xa, xb) = (0.0, 1.0);
        let nn = 10;
        let zz = InterpChebyshev::points(nn);
        let dx = xb - xa;
        let args = &mut 0;
        let uu = zz.get_mapped(|z| {
            let x = (xb + xa + dx * z) / 2.0;
            f(x, args).unwrap()
        });
        let interp = InterpChebyshev::new_with_uu(xa, xb, uu.as_data()).unwrap();

        // check
        let np = nn + 1;
        for i in 0..np {
            let x = (xb + xa + dx * zz[i]) / 2.0;
            let fxi = interp.eval(x).unwrap();
            approx_eq(fxi, interp.uu[i], 1e-13);
        }
        let err = interp.estimate_max_error(100, args, f).unwrap();
        println!("err = {}", err);
        assert!(err < 0.73);

        // plot f(x)
        /*
        let xx = Vector::linspace(xa, xb, 201).unwrap();
        let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
        let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
        let mut curve_ana = Curve::new();
        let mut curve_int = Curve::new();
        curve_ana.set_label("analytical");
        curve_int
            .set_label(&format!("interpolated,N={}", nn))
            .set_line_style(":")
            .set_marker_style(".")
            .set_marker_every(5);
        curve_ana.draw(xx.as_data(), yy_ana.as_data());
        curve_int.draw(xx.as_data(), yy_int.as_data());
        let mut plot = Plot::new();
        let mut legend = Legend::new();
        legend.set_num_col(4);
        legend.set_outside(true);
        legend.draw();
        plot.add(&curve_ana)
            .add(&curve_int)
            .add(&legend)
            .set_cross(0.0, 0.0, "gray", "-", 1.5)
            .grid_and_labels("x", "f(x)")
            .save("/tmp/russell_lab/test_new_with_uu.svg")
            .unwrap();
        */
    }

    #[test]
    fn eval_captures_errors() {
        let nn = 2;
        let (xa, xb) = (-4.0, 4.0);
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        assert_eq!(interp.eval(0.0).err(), Some("all U components must be set first"));
    }

    #[test]
    fn eval_using_trig_captures_errors() {
        let nn = 2;
        let (xa, xb) = (-4.0, 4.0);
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        assert_eq!(
            interp.eval_using_trig(0.0).err(),
            Some("all U components must be set first")
        );
    }

    #[test]
    fn new_adapt_captures_errors() {
        struct Args {
            count: usize,
        }
        let f = move |x: f64, a: &mut Args| {
            a.count += 1;
            if a.count == 3 {
                return Err("stop with count = 3");
            }
            if a.count == 18 {
                return Err("stop with count = 18");
            }
            Ok(x * x - 1.0)
        };
        let mut args = Args { count: 0 };
        let (xa, xb) = (-4.0, 4.0);
        let tol = 1e-3;
        assert_eq!(
            InterpChebyshev::new_adapt(2049, tol, xa, xb, &mut args, f).err(),
            Some("the maximum degree N must be ≤ 2048")
        );
        assert_eq!(
            InterpChebyshev::new_adapt(2, tol, xa, xa + TOL_RANGE, &mut args, f).err(),
            Some("xb must be greater than xa + ϵ")
        );
        assert_eq!(
            InterpChebyshev::new_adapt(1, tol, xa, xb, &mut args, f).err(),
            Some("adaptive interpolation did not converge")
        );
        assert_eq!(
            InterpChebyshev::new_adapt(2, tol, xa, xb, &mut args, f).err(),
            Some("stop with count = 3")
        );
        assert_eq!(
            InterpChebyshev::new_adapt(4, tol, xa, xb, &mut args, f).err(),
            Some("stop with count = 18")
        );
    }

    #[test]
    fn new_adapt_and_eval_work() {
        let functions = [
            |_: f64, _: &mut NoArgs| Ok(2.0),
            |x: f64, _: &mut NoArgs| Ok(x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x - 1.0),
            |x: f64, _: &mut NoArgs| Ok(x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x * x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x)),
            |x: f64, _: &mut NoArgs| Ok(0.092834 * f64::sin(77.0001 + 19.87 * x)),
            |x: f64, _: &mut NoArgs| Ok(f64::ln(2.0 * f64::cos(x / 2.0))),
        ];
        let ranges = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-2.34567, 12.34567),
            (-0.995 * PI, 0.995 * PI),
        ];
        let tols_adapt = [0.0, 0.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-6, 1e-6, 1e-6];
        let tols_eval = [0.0, 0.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-14, 1e-14, 1e-14];
        let nn_max = 400;
        let tol = 1e-7;
        let args = &mut 0;
        for (index, f) in functions.into_iter().enumerate() {
            // adaptive interpolation
            let (xa, xb) = ranges[index];
            let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f).unwrap();
            let nn = interp.get_degree();

            // check adaptive interpolation
            let err = interp.estimate_max_error(1000, args, f).unwrap();
            println!("{:0>3} : N = {:>3} : err = {:.2e}", index, nn, err);
            assert!(err <= tols_adapt[index]);

            // check eval and eval_using_trig
            let stations_for_eval = Vector::linspace(xa, xb, 100).unwrap();
            for x in &stations_for_eval {
                let res1 = interp.eval(*x).unwrap();
                let res2 = interp.eval_using_trig(*x).unwrap();
                let err = f64::abs(res1 - res2);
                assert!(err <= tols_eval[index]);
            }

            // plot f(x)
            /*
            let xx = Vector::linspace(xa, xb, 201).unwrap();
            let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
            let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
            let mut curve_ana = Curve::new();
            let mut curve_int = Curve::new();
            curve_ana.set_label("analytical");
            curve_int
                .set_label(&format!("interpolated,N={}", nn))
                .set_line_style(":")
                .set_marker_style(".")
                .set_marker_every(5);
            curve_ana.draw(xx.as_data(), yy_ana.as_data());
            curve_int.draw(xx.as_data(), yy_int.as_data());
            let mut plot = Plot::new();
            let mut legend = Legend::new();
            legend.set_num_col(4);
            legend.set_outside(true);
            legend.draw();
            plot.add(&curve_ana)
                .add(&curve_int)
                .add(&legend)
                .set_cross(0.0, 0.0, "gray", "-", 1.5)
                .grid_and_labels("x", "f(x)")
                .save(&format!(
                    "/tmp/russell_lab/test_interp_chebyshev_new_adapt_{:0>3}.svg",
                    index
                ))
                .unwrap();
            */
        }
    }

    #[test]
    fn estimate_max_error_captures_errors_1() {
        let f = |x: f64, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        let nn = 2;
        let (xa, xb) = (-4.0, 4.0);
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        assert_eq!(
            interp.estimate_max_error(100, args, f).err(),
            Some("all U components must be set first")
        );
    }

    #[test]
    fn estimate_max_error_captures_errors_2() {
        struct Args {
            count: usize,
        }
        let f = move |x: f64, a: &mut Args| {
            a.count += 1;
            if a.count == 1 {
                return Err("stop with count = 1");
            }
            Ok(x * x - 1.0)
        };
        let mut args = Args { count: 0 };
        let (xa, xb) = (0.0, 1.0);
        let uu = &[1.0];
        let interp = InterpChebyshev::new_with_uu(xa, xb, uu).unwrap();
        assert_eq!(
            interp.estimate_max_error(2, &mut args, f).err(),
            Some("stop with count = 1")
        );
    }

    #[test]
    fn set_uu_value_works() {
        let nn = 2;
        let (xa, xb) = (-4.0, 4.0);
        let mut interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        let zz = InterpChebyshev::points(nn);
        let dx = xb - xa;
        let np = nn + 1;
        for i in 0..np {
            let x = (xb + xa + dx * zz[i]) / 2.0;
            interp.set_uu_value(i, x * x - 1.0);
        }
        // check
        let f = |x: f64, _: &mut NoArgs| Ok(x * x - 1.0);
        let args = &mut 0;
        let err = interp.estimate_max_error(100, args, f).unwrap();
        assert!(err < 1e-14);
    }

    #[test]
    fn getters_work() {
        let f = |x: f64, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);
        let nn = 2;
        let args = &mut 0;
        let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
        assert_eq!(interp.get_degree(), 2);
        assert_eq!(interp.get_range(), (-4.0, 4.0, 8.0));
        assert_eq!(interp.is_ready(), true);
        vec_approx_eq(interp.get_coefficients(), &[7.0, 0.0, 8.0], 1e-15);
    }
}
