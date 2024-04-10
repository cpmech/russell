use crate::math::{chebyshev_gauss_points, chebyshev_lobatto_points, neg_one_pow_n};
use crate::StrError;
use crate::{mat_vec_mul, Matrix, Vector};

/// Defines the type of the interpolation grid in 1D
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpGrid {
    Uniform,
    ChebyshevGauss,
    ChebyshevGaussLobatto,
}

/// Holds additional parameters for the interpolation functions
#[derive(Clone, Copy, Debug)]
pub struct InterpParams {
    /// Polynomial degree `N` satisfying `1 ≤ N ≤ 2048`
    pub nn: usize,

    /// The type of grid
    pub grid_type: InterpGrid,

    /// Do not use the eta normalization
    pub no_eta_normalization: bool,

    /// Constant to apply the alternative normalization for higher N (e.g., 700)
    pub eta_cutoff: usize,

    /// Number of stations (points) for the Lebesgue estimate (e.g., 10,000)
    pub lebesgue_estimate_nstation: usize,

    /// Number of stations (points) for the interpolation error estimate (e.g., 10,000)
    pub error_estimate_nstation: usize,
}

impl InterpParams {
    /// Allocates a new instance with default values
    pub fn new() -> Self {
        InterpParams {
            nn: 20,
            grid_type: InterpGrid::ChebyshevGaussLobatto,
            no_eta_normalization: false,
            eta_cutoff: 700,
            lebesgue_estimate_nstation: 10_000,
            error_estimate_nstation: 10_000,
        }
    }

    /// Validates the parameters
    fn validate(&self) -> Result<(), StrError> {
        if self.nn < 1 || self.nn > 2048 {
            return Err("the polynomial degree must be in [1, 2048]");
        }
        if self.lebesgue_estimate_nstation < 2 {
            return Err("lebesgue_estimate_nstation must be ≥ 2");
        }
        if self.error_estimate_nstation < 2 {
            return Err("error_estimate_nstation must be ≥ 2");
        }
        Ok(())
    }
}

/// Implements a polynomial interpolant in Lagrange Form
///
/// **Note:** The barycentric form (discussed below) is the only one considered here.
///
/// A polynomial interpolant `I^X_N{f}` for the function `f(x)`, associated with a grid `X`, of degree `N`,
/// and with `N+1` points, is expressed in the Lagrange form as
/// (see Eq 2.2.19 of Ref #1, page 73; and Eq 3.31 of Ref #2, page 73):
///
/// ```text
///                         N
///                       —————
///            X          \             X
/// pnu(x) := I {f}(x) =  /      U  ⋅  ℓ (x)
///            N          —————   j     j
///                       j = 0
///
/// with Uⱼ := f(xⱼ)
/// ```
///
/// where `ℓ^X_j(x)` is the j-th Lagrange cardinal polynomial associated with grid X and given by
/// (see Eq 2.3.27 of Ref #1, page 80; and Eq 3.32 of Ref #2, page 74):
///
/// ```text
///           N
///         ━━━━━
///  X      ┃   ┃  x  - Xₖ
/// ℓ (x) = ┃   ┃  ———————
///  j      k = 0  Xⱼ - Xₖ
///         k ≠ j
///
/// 0 ≤ j ≤ N
/// ```
///
/// In barycentric form, the interpolant is expressed as
/// (see Eq 3.36 of Ref #2, page 74):
///
/// ```text
///                       N       λⱼ
///                       Σ  Uⱼ ——————
///            X         j=0    x - Xⱼ
/// pnu(x) := I {f}(x) = —————————————
///            N           N     λₖ
///                        Σ   ——————
///                       k=0  x - Xₖ
///
/// with Uⱼ := f(xⱼ)
/// ```
///
/// where (see Eq 2.4.34 of Ref #1, page 90; and Eq 3.34 of Ref #2, page 74)
///
/// ```text
///             1
/// λⱼ = ————————————————
///         N
///         𝚷   (Xⱼ - Xₖ)
///      k=0,k≠j
/// ```
///
/// Let us define (see Eq 2.4.34 of Ref #1, page 90):
///
/// ```text
///              λⱼ
///            ——————
///  X         x - Xⱼ
/// ψ (x) = ———————————
///  j       N     λₖ
///          Σ   ——————
///         k=0  x - Xₖ
/// ```
///
/// Then:
///
/// ```text
///           N
/// pnu(x) =  Σ  Uⱼ ⋅ pⱼ(x)
///          j=0
///
/// with pⱼ(x) = ℓⱼ(x) = ψⱼ(x)
/// ```
///
/// To minimize round-off problems, an option to normalize the barycentric weights `λₖ` is available.
/// The strategy is to normalize lambda using the so-called eta-factors (`η`) as follows (See Ref #3 and #4).
/// First the eta coefficients are computed:
///
/// ```text
///         N
/// ηⱼ =    Σ   ln(|Xⱼ - Xₖ|)
///      k=0,k≠j
/// ```
///
/// Then, the lambda weights are computed as:
///
/// ```text
///      aⱼ ⋅ bⱼ          
/// λⱼ = ———————
///        lf0
/// ```
///
/// with:
///
/// ```text
///          j+N
/// aⱼ = (-1)
/// bⱼ = exp(mⱼ)
/// mⱼ = -ηⱼ
/// c0 = (2^(N-1))/N
/// ```
///
/// For higher degrees, e.g., N > 700, the alternative normalization is applied:
///
/// ```text
///      ⎛ aⱼ bⱼ ⎞   ⎛ bⱼ ⎞   ⎛ bⱼ ⎞
/// λⱼ = ⎜ ————— ⎟ ⋅ ⎜ —— ⎟ ⋅ ⎜ —— ⎟
///      ⎝   c0  ⎠   ⎝ c1 ⎠   ⎝ c2 ⎠
/// ```
///
/// with:
///
/// ```text
/// bⱼ = exp(mⱼ/3)
/// c0 = c1 = 2^(N/3)
/// c2 = (2^(N/3-1))/N
/// c0 ⋅ c1 ⋅ c2 = (2^(N-1))/N
/// ```
///
/// # Properties
///
/// The Lagrange polynomial `ℓᵢ` corresponding to node xᵢ has the Kronecker property:
///
/// ```text
///                    ⎧ 1  if i = j
/// pᵢ(xⱼ) := ℓᵢ(xⱼ) = ⎨
///                    ⎩ 0  if i ≠ j
/// ```
///
/// Also:
///
/// ```text
///  N
///  Σ  pⱼ(x) = 1
/// j=0
/// ```
///
/// # References
///
/// 1. Canuto C, Hussaini MY, Quarteroni A, Zang TA (2006) Spectral Methods: Fundamentals in
///    Single Domains. Springer. 563p
/// 2. Kopriva DA (2009) Implementing Spectral Methods for Partial Differential Equations
///    Springer, 404p
/// 3. Costa B, Don WS (2000) On the computation of high order pseudospectral derivatives,
///    Applied Numerical Mathematics, 33:151-159.
/// 4. Baltensperger R, Trummer MR (2003) Spectral differencing with a twist,
///    SIAM Journal of Scientific Computing, 24(5):1465-1487
/// 5. Berrut JP, Trefethen LN (2004) Barycentric Lagrange Interpolation,
///    SIAM Review Vol. 46, No. 3, pp. 501-517
#[derive(Clone, Debug)]
pub struct InterpLagrange {
    /// Additional parameters
    params: InterpParams,

    /// number of points (equal to `N + 1`)
    npoint: usize,

    /// Point coordinates in `[-1, 1]`
    xx: Vector,

    /// Eta coefficients to normalize lambda
    eta: Vector,

    /// Lambda coefficients of the barycentric formula
    lambda: Vector,

    /// Matrix (D1) with the first-order derivative coefficients `(dℓⱼ/dx)(xₖ)`
    dd1: Matrix,

    /// Matrix (D2) with the second-order derivative coefficients `(d²ℓⱼ/dx²)(xₖ)`
    dd2: Matrix,
}

impl InterpLagrange {
    /// Allocates a new instance
    pub fn new(params: InterpParams) -> Result<Self, StrError> {
        // check the parameters
        params.validate()?;

        // interp struct
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange {
            params,
            npoint,
            xx: match params.grid_type {
                InterpGrid::Uniform => Vector::linspace(-1.0, 1.0, npoint).unwrap(),
                InterpGrid::ChebyshevGauss => chebyshev_gauss_points(params.nn),
                InterpGrid::ChebyshevGaussLobatto => chebyshev_lobatto_points(params.nn),
            },
            eta: Vector::new(if params.no_eta_normalization { 0 } else { npoint }),
            lambda: Vector::new(npoint),
            dd1: Matrix::new(0, 0), // indicates: "not computed yet"
            dd2: Matrix::new(0, 0), // indicates: "not computed yet"
        };

        // without normalization
        //
        //             1
        // λⱼ = ————————————————
        //         N
        //         𝚷   (Xⱼ - Xₖ)
        //      k=0,k≠j
        //
        if params.no_eta_normalization {
            for j in 0..npoint {
                let mut prod = 1.0;
                for k in 0..npoint {
                    if k != j {
                        prod *= interp.xx[j] - interp.xx[k];
                    }
                }
                interp.lambda[j] = 1.0 / prod;
            }
        } else {
            // with normalization
            //
            //         N
            // ηⱼ =    Σ   ln(|Xⱼ - Xₖ|)
            //      k=0,k≠j
            //
            for j in 0..npoint {
                for k in 0..npoint {
                    if k != j {
                        interp.eta[j] += f64::ln(f64::abs(interp.xx[j] - interp.xx[k]));
                    }
                }
            }
            // factors
            let nnf = params.nn as f64;
            let (c0, c1, c2) = if params.nn > params.eta_cutoff {
                (
                    f64::powf(2.0, nnf / 3.0),
                    f64::powf(2.0, nnf / 3.0),
                    f64::powf(2.0, nnf / 3.0 - 1.0) / nnf,
                )
            } else {
                (f64::powf(2.0, nnf - 1.0) / nnf, 0.0, 0.0)
            };
            // lambda
            for j in 0..npoint {
                let aj = neg_one_pow_n((j + params.nn) as i32);
                let mj = -interp.eta[j];
                if params.nn > params.eta_cutoff {
                    //      ⎛ aⱼ bⱼ ⎞   ⎛ bⱼ ⎞   ⎛ bⱼ ⎞
                    // λⱼ = ⎜ ————— ⎟ ⋅ ⎜ —— ⎟ ⋅ ⎜ —— ⎟
                    //      ⎝   c0  ⎠   ⎝ c1 ⎠   ⎝ c2 ⎠
                    let bj = f64::exp(mj / 3.0);
                    interp.lambda[j] = aj * bj / c0;
                    interp.lambda[j] *= bj / c1;
                    interp.lambda[j] *= bj / c2;
                } else {
                    //      aⱼ ⋅ bⱼ
                    // λⱼ = ———————
                    //        lf0
                    let bj = f64::exp(mj);
                    interp.lambda[j] = aj * bj / c0;
                }
                assert!(interp.lambda[j].is_finite());
            }
        }

        // done
        Ok(interp)
    }

    /// Computes the i-th polynomial associated with grid X
    ///
    /// Calculates `pⱼ` in:
    ///
    /// ```text
    ///           N
    /// pnu(x) =  Σ  Uⱼ ⋅ pⱼ(x)
    ///          j=0
    ///
    /// with pⱼ(x) = ℓⱼ(x) = ψⱼ(x)
    /// ```
    ///
    /// Barycentric form:
    ///
    /// ```text
    ///                      λⱼ
    ///                    ——————
    ///          X         x - Xⱼ
    /// pⱼ(x) = ψ (x) = ———————————
    ///          j       N     λₖ
    ///                  Σ   ——————
    ///                 k=0  x - Xₖ
    /// ```
    ///
    /// # Input
    ///
    /// * `j` -- index of the Xⱼ point; must satisfy 0 ≤ j ≤ N
    /// * `x` -- the coordinate to evaluate the polynomial
    ///
    /// # Panics
    ///
    /// This function will panic if `j > N`.
    pub fn poly(&self, j: usize, x: f64) -> f64 {
        assert!(j <= self.params.nn);
        if f64::abs(x - self.xx[j]) < 10.0 * f64::EPSILON {
            return 1.0;
        }
        let mut sum = 0.0;
        for k in 0..self.npoint {
            sum += self.lambda[k] / (x - self.xx[k]);
        }
        self.lambda[j] / (x - self.xx[j]) / sum
    }

    /// Evaluates the function f(x) over all nodes
    ///
    /// # Output
    ///
    /// * `uu` -- the "data" vector `U` of size equal to N + 1
    ///
    /// # Input
    ///
    /// * `f` -- the function f(x) implemented as `(i: usize, x: f64) -> f64`
    ///
    /// # Panics
    ///
    /// Will panic if `uu.dim()` is not equal to the number of points `N + 1`
    pub fn evaluate_f_over_nodes<F>(&self, uu: &mut Vector, mut f: F)
    where
        F: FnMut(usize, f64) -> f64,
    {
        assert_eq!(uu.dim(), self.npoint);
        for j in 0..self.npoint {
            uu[j] = f(j, self.xx[j]);
        }
    }

    /// Performs the interpolation
    ///
    /// Calculates:
    ///
    /// ```text
    ///           N
    /// pnu(x) =  Σ  Uⱼ ⋅ pⱼ(x)
    ///          j=0
    ///
    /// with pⱼ(x) = ℓⱼ(x) = ψⱼ(x)
    /// ```
    ///
    /// # Input
    ///
    /// * `uu` -- the "data" vector `U` of size equal to N + 1
    ///
    /// # Panics
    ///
    /// Will panic if `uu.dim()` is not equal to the number of points `N + 1`
    pub fn execute(&self, x: f64, uu: &Vector) -> f64 {
        assert_eq!(uu.dim(), self.npoint);
        let mut res = 0.0;
        for j in 0..self.npoint {
            res += uu[j] * self.poly(j, x);
        }
        res
    }

    /// Computes the differentiation matrix D1
    ///
    /// ```text
    /// dI{f}(x) │       N  dℓⱼ(x) │
    /// ———————— │    =  Σ  —————— │   ⋅ f(xⱼ)  =  D1ₖⱼ ⋅ f(xⱼ)
    ///    dx    │x=xₖ  j=0   dx   │x=xₖ          
    /// ```
    pub fn calc_dd1_matrix(&mut self) {
        // allocate matrix
        if self.dd1.dims().0 == self.npoint {
            return; // already calculated
        }
        self.dd1 = Matrix::new(self.npoint, self.npoint);

        if self.params.no_eta_normalization {
            // calculate D1 using λⱼ directly
            for k in 0..self.npoint {
                let mut row_sum = 0.0;
                for j in 0..self.npoint {
                    if k != j {
                        let v = (self.lambda[j] / self.lambda[k]) / (self.xx[k] - self.xx[j]);
                        self.dd1.set(k, j, v);
                        row_sum += v;
                    }
                }
                self.dd1.set(k, k, -row_sum); // negative sum trick
            }
        } else {
            // calculate D1 using ηⱼ
            for k in 0..self.npoint {
                let mut row_sum = 0.0;
                for j in 0..self.npoint {
                    if k != j {
                        let r = neg_one_pow_n((k + j) as i32) * f64::exp(self.eta[k] - self.eta[j]);
                        let v = r / (self.xx[k] - self.xx[j]);
                        self.dd1.set(k, j, v);
                        row_sum += v;
                    }
                }
                self.dd1.set(k, k, -row_sum); // negative sum trick
            }
        }
    }

    /// Computes the differentiation matrix D2
    ///
    /// ```text
    /// d²I{f}(x) │       N  d²ℓⱼ(x) │    
    /// ————————— │    =  Σ  ——————— │   ⋅ f(xⱼ)  =  D2ₖⱼ f(xⱼ)
    ///    dx²    │x=xₖ  j=0   dx²   │x=xₖ
    /// ```
    pub fn calc_dd2_matrix(&mut self) {
        // calculate D1
        self.calc_dd1_matrix();

        // allocate matrix
        if self.dd2.dims().0 == self.npoint {
            return; // already calculated
        }
        self.dd2 = Matrix::new(self.npoint, self.npoint);

        // compute D2 from D1 (recursion formula; see Eqs 9 and 13 of Ref #3)
        for k in 0..self.npoint {
            let mut row_sum = 0.0;
            for j in 0..self.npoint {
                if k != j {
                    let v = 2.0 * self.dd1.get(k, j) * (self.dd1.get(k, k) - 1.0 / (self.xx[k] - self.xx[j]));
                    self.dd2.set(k, j, v);
                    row_sum += v;
                }
            }
            self.dd2.set(k, k, -row_sum); // negative sum trick
        }
    }

    /// Computes the maximum error due to differentiation using the D1 matrix
    ///
    /// # Input
    ///
    /// * `uu` -- the "data" vector `U` of size equal to N + 1
    /// * `dfdx_ana` -- function `(i: usize, x: f64) -> f64` with the analytical solution
    ///
    /// # Panics
    ///
    /// Will panic if `uu.dim()` is not equal to the number of points `N + 1`
    pub fn max_error_dd1<F>(&self, uu: &Vector, mut dfdx_ana: F) -> f64
    where
        F: FnMut(usize, f64) -> f64,
    {
        // derivative of interpolation @ xᵢ
        assert_eq!(uu.dim(), self.npoint);
        let mut v = Vector::new(self.npoint);
        mat_vec_mul(&mut v, 1.0, &self.dd1, uu).unwrap();

        // compute error
        let mut max_err = 0.0;
        for i in 0..self.npoint {
            let v_ana = dfdx_ana(i, self.xx[i]);
            let diff = f64::abs(v[i] - v_ana);
            if diff > max_err {
                max_err = diff;
            }
        }
        max_err
    }

    /// Computes the maximum error due to differentiation using the D2 matrix
    ///
    /// Computes the error @ `X[i]`
    ///
    /// # Input
    ///
    /// * `uu` -- the "data" vector `U` of size equal to N + 1
    /// * `d2fdx2_ana` -- function `(i: usize, x: f64) -> f64` with the analytical solutions
    ///
    /// # Panics
    ///
    /// Will panic if `uu.dim()` is not equal to the number of points `N + 1`
    pub fn max_error_dd2<F>(&self, uu: &Vector, mut d2fdx2_ana: F) -> f64
    where
        F: FnMut(usize, f64) -> f64,
    {
        // derivative of interpolation @ xᵢ
        assert_eq!(uu.dim(), self.npoint);
        let mut v = Vector::new(self.npoint);
        mat_vec_mul(&mut v, 1.0, &self.dd2, uu).unwrap();

        // compute error
        let mut max_err = 0.0;
        for i in 0..self.npoint {
            let v_ana = d2fdx2_ana(i, self.xx[i]);
            let diff = f64::abs(v[i] - v_ana);
            if diff > max_err {
                max_err = diff;
            }
        }
        max_err
    }

    /// Estimates the Lebesgue constant ΛN
    pub fn estimate_lebesgue_constant(&self) -> f64 {
        let n_station = self.params.lebesgue_estimate_nstation;
        let mut lambda_times_nn = 0.0;
        for j in 0..n_station {
            let x = -1.0 + 2.0 * (j as f64) / ((n_station - 1) as f64);
            let mut sum = f64::abs(self.poly(0, x));
            for i in 1..self.npoint {
                sum += f64::abs(self.poly(i, x));
            }
            if sum > lambda_times_nn {
                lambda_times_nn = sum;
            }
        }
        lambda_times_nn
    }

    /// Estimates the maximum error of the interpolation
    ///
    /// Computes:
    ///
    /// ```text
    /// max_err = max(|f(x) - I{f}(x)|)
    /// ```
    ///
    /// # Input
    ///
    /// * `uu` -- the "data" vector `U` of size equal to N + 1
    /// * `f` -- function `(x: f64, i: usize) -> f64`
    ///
    /// # Output
    ///
    /// Returns `(max_err, x_loc)` where `x_loc` is the location of the max error
    ///
    /// # Panics
    ///
    /// Will panic if `uu.dim()` is not equal to the number of points `N + 1`
    pub fn estimate_max_err<F>(&self, uu: &Vector, mut f: F) -> (f64, f64)
    where
        F: FnMut(f64, usize) -> f64,
    {
        assert_eq!(uu.dim(), self.npoint);
        let mut max_err = 0.0;
        let mut x_loc = 0.0;
        let den = (self.params.error_estimate_nstation - 1) as f64;
        for i in 0..self.params.error_estimate_nstation {
            let x = -1.0 + 2.0 * (i as f64) / den;
            let fx = f(x, i);
            let ix = self.execute(x, uu);
            let e = f64::abs(fx - ix);
            if e > max_err {
                max_err = e;
                x_loc = x;
            }
        }
        (max_err, x_loc)
    }

    /// Executes a loop over the grid points
    ///
    /// Loops over `(i: usize, x: f64)`
    ///
    /// # Input
    ///
    /// * `callback` -- a function of `(j, x)` where `j` is the point number,
    ///   and `x` is the Cartesian coordinates of the point.
    pub fn loop_over_grid_points<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        for j in 0..self.npoint {
            callback(j, self.xx[j]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{InterpGrid, InterpLagrange, InterpParams};
    use crate::{approx_eq, deriv1_approx_eq, deriv2_approx_eq, Vector};

    #[test]
    fn new_works() {
        let mut params = InterpParams::new();
        params.nn = 2;
        let interp = InterpLagrange::new(params).unwrap();
        assert_eq!(interp.npoint, 3);
        assert_eq!(interp.xx.as_data(), &[-1.0, 0.0, 1.0]);
        assert_eq!(interp.eta.dim(), 3);
        assert_eq!(interp.lambda.dim(), 3);
        assert_eq!(interp.dd1.dims(), (0, 0));
        assert_eq!(interp.dd2.dims(), (0, 0));
    }

    // --- lambda and psi -----------------------------------------------------------------------------

    fn check_lambda(params: InterpParams, tol: f64) {
        let nnf = params.nn as f64;
        let npoint = params.nn + 1;
        let interp = InterpLagrange::new(params).unwrap();
        // ```text
        //      N-1
        //     2
        // m = ————    TODO: check this formula
        //      N
        // ```
        let m = f64::powf(2.0, nnf - 1.0) / nnf;
        for i in 0..npoint {
            let mut d = 1.0;
            for j in 0..npoint {
                if i != j {
                    d *= interp.xx[i] - interp.xx[j]
                }
            }
            approx_eq(interp.lambda[i], 1.0 / d / m, tol);
        }
    }

    #[test]
    fn lambda_is_correct() {
        let mut params = InterpParams::new();
        for nn in 1..20 {
            params.nn = nn;
            for (tol, grid_type) in [
                (1e-12, InterpGrid::Uniform),
                (1e-14, InterpGrid::ChebyshevGauss),
                (1e-14, InterpGrid::ChebyshevGaussLobatto),
            ] {
                // println!("nn = {}, grid = {:?}", nn, grid_type);
                params.grid_type = grid_type;
                check_lambda(params, tol);
            }
        }
    }

    fn check_psi(params: InterpParams, tol_comparison: f64) {
        let npoint = params.nn + 1;

        // interpolant
        let interp = InterpLagrange::new(params).unwrap();

        // check Kronecker property (barycentric)
        for i in 0..npoint {
            let mut sum = 0.0;
            for j in 0..npoint {
                let psi = interp.poly(i, interp.xx[j]);
                let mut ana = 1.0;
                if i != j {
                    ana = 0.0;
                }
                assert_eq!(psi, ana);
                sum += psi;
            }
            assert_eq!(sum, 1.0);
        }

        // Cardinal form:
        //
        // ```text
        //           N
        //         ━━━━━
        //  X      ┃   ┃  x  - Xₖ
        // ℓ (x) = ┃   ┃  ———————
        //  j      k = 0  Xⱼ - Xₖ
        //         k ≠ j
        // ```

        // compare barycentric versus cardinal
        let xx = Vector::linspace(-1.0, 1.0, 20).unwrap();
        for x in xx {
            for j in 0..npoint {
                let psi_j = interp.poly(j, x);
                let mut ell_j = 1.0;
                for k in 0..npoint {
                    if j != k {
                        ell_j *= (x - interp.xx[k]) / (interp.xx[j] - interp.xx[k]);
                    }
                }
                approx_eq(psi_j, ell_j, tol_comparison);
            }
        }
    }

    #[test]
    fn psi_is_correct() {
        let mut params = InterpParams::new();
        for nn in 1..20 {
            params.nn = nn;
            for (tol, grid_type) in [
                (1e-11, InterpGrid::Uniform),
                (1e-14, InterpGrid::ChebyshevGauss),
                (1e-14, InterpGrid::ChebyshevGaussLobatto),
            ] {
                // println!("nn = {}, grid = {:?}", nn, grid_type);
                params.grid_type = grid_type;
                check_psi(params, tol);
            }
        }
    }

    // --- polynomial interpolation -------------------------------------------------------------------

    #[test]
    fn poly_at_nodes_is_exact_1() {
        let f = |_, x| f64::cos(f64::exp(2.0 * x));
        let mut params = InterpParams::new();
        params.nn = 5;
        params.grid_type = InterpGrid::Uniform;
        let npoint = params.nn + 1;
        let interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.execute(x, &uu), f(i, x)));
    }

    #[test]
    fn poly_at_nodes_is_exact_2() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut params = InterpParams::new();
        params.nn = 8;
        let npoint = params.nn + 1;
        let interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.execute(x, &uu), f(i, x)));
    }

    fn check_execute<F>(params: InterpParams, mut f: F)
    where
        F: Copy + FnMut(usize, f64) -> f64,
    {
        let npoint = params.nn + 1;
        let interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        // check the interpolation @ nodes
        interp.loop_over_grid_points(|j, x| {
            assert_eq!(interp.execute(x, &uu), f(j, x));
        });
    }

    #[test]
    fn poly_works() {
        let f = |_, x| f64::cos(f64::exp(2.0 * x));
        let mut params = InterpParams::new();
        for nn in 1..20 {
            params.nn = nn;
            for grid_type in [
                InterpGrid::Uniform,
                InterpGrid::ChebyshevGauss,
                InterpGrid::ChebyshevGaussLobatto,
            ] {
                // println!("nn = {}, grid = {:?}", nn, grid_type);
                params.grid_type = grid_type;
                check_execute(params, f);
            }
        }
    }

    // --- derivatives --------------------------------------------------------------------------------

    fn check_dd1_matrix(params: InterpParams, tol: f64) {
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();
        interp.calc_dd1_matrix();
        struct Args {}
        let args = &mut Args {};
        for i in 0..npoint {
            let xi = interp.xx[i];
            for j in 0..npoint {
                deriv1_approx_eq(interp.dd1.get(i, j), xi, args, tol, |x, _| Ok(interp.poly(j, x)));
            }
        }
    }

    #[test]
    fn dd1_matrix_works() {
        // with eta
        #[rustfmt::skip]
        let nn_and_tols = [
            (2, 1e-12),
            (5, 1e-9),
            (10, 1e-8),
        ];
        let mut params = InterpParams::new();
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd1_matrix(params, tol);
        }
        // no eta
        #[rustfmt::skip]
        let nn_and_tols = [
            (2, 1e-12),
            (5, 1e-9),
            (10, 1e-8),
        ];
        params.no_eta_normalization = true;
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd1_matrix(params, tol);
        }
    }

    fn check_dd2_matrix(params: InterpParams, tol: f64) {
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();
        interp.calc_dd2_matrix();
        struct Args {}
        let args = &mut Args {};
        for i in 0..npoint {
            let xi = interp.xx[i];
            for j in 0..npoint {
                deriv2_approx_eq(interp.dd2.get(i, j), xi, args, tol, |x, _| Ok(interp.poly(j, x)));
            }
        }
    }

    #[test]
    fn dd2_matrix_works() {
        #[rustfmt::skip]
        let nn_and_tols = [
            (2, 1e-9),
            (5, 1e-8),
            (10, 1e-8),
        ];
        let mut params = InterpParams::new();
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd2_matrix(params, tol);
        }
    }

    fn check_dd1_error<F, G>(params: InterpParams, tol: f64, f: F, dfdx_ana: G)
    where
        F: FnMut(usize, f64) -> f64,
        G: FnMut(usize, f64) -> f64,
    {
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.calc_dd1_matrix();
        let max_diff = interp.max_error_dd1(&uu, dfdx_ana);
        if max_diff > tol {
            panic!("D1⋅U failed; max_diff = {:?}", max_diff);
        }
    }

    #[test]
    fn dd1_times_uu_works() {
        let mut params = InterpParams::new();
        let f = |_, x| f64::powf(x, 8.0);
        let g = |_, x| 8.0 * f64::powf(x, 7.0);
        for (nn, grid_type, tol) in [
            (8, InterpGrid::Uniform, 1e-13),
            (8, InterpGrid::ChebyshevGauss, 1e-14),
            (8, InterpGrid::ChebyshevGaussLobatto, 1e-13),
        ] {
            // println!("nn = {}, grid = {:?}", nn, grid_type);
            params.nn = nn;
            params.grid_type = grid_type;
            check_dd1_error(params, tol, f, g);
        }
    }

    fn check_dd2_error<F, H>(params: InterpParams, tol: f64, f: F, d2fdx2_ana: H)
    where
        F: FnMut(usize, f64) -> f64,
        H: FnMut(usize, f64) -> f64,
    {
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.calc_dd2_matrix();
        let max_diff = interp.max_error_dd2(&uu, d2fdx2_ana);
        if max_diff > tol {
            panic!("D2⋅U failed; max_diff = {:?}", max_diff);
        }
    }

    #[test]
    fn dd2_times_uu_works() {
        let mut params = InterpParams::new();
        let f = |_, x| f64::powf(x, 8.0);
        // let g = |_, x| 8.0 * f64::powf(x, 7.0);
        let h = |_, x| 56.0 * f64::powf(x, 6.0);
        for (nn, grid_type, tol) in [
            (8, InterpGrid::Uniform, 1e-11),
            (8, InterpGrid::ChebyshevGauss, 1e-12),
            (8, InterpGrid::ChebyshevGaussLobatto, 1e-12),
        ] {
            // println!("nn = {}, grid = {:?}", nn, grid_type);
            params.nn = nn;
            params.grid_type = grid_type;
            check_dd2_error(params, tol, f, h);
        }
    }

    // --- Lebesgue -----------------------------------------------------------------------------------

    #[test]
    fn lebesgue_works_uniform() {
        let mut params = InterpParams::new();
        params.nn = 5;
        params.grid_type = InterpGrid::Uniform;
        let tol = 1e-3;
        params.lebesgue_estimate_nstation = 210; // 1e-15 is achieved with 10_000
        let interp = InterpLagrange::new(params).unwrap();
        approx_eq(interp.estimate_lebesgue_constant(), 3.106301040275436e+00, tol);
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut params = InterpParams::new();
        params.nn = 8;
        params.grid_type = InterpGrid::ChebyshevGauss;
        let npoint = params.nn + 1;
        params.lebesgue_estimate_nstation = 10;
        let interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.execute(x, &uu), f(i, x)));
        let nn_and_lebesgue = [
            (4, 1.988854381999833e+00),
            (8, 2.361856787767076e+00),
            (24, 3.011792612349363e+00),
        ];
        for (nn, lambda_times_nn) in nn_and_lebesgue {
            params.nn = nn;
            let interp = InterpLagrange::new(params).unwrap();
            approx_eq(interp.estimate_lebesgue_constant(), lambda_times_nn, 1e-14);
        }
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss_lobatto() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut params = InterpParams::new();
        params.nn = 8;
        params.grid_type = InterpGrid::ChebyshevGaussLobatto;
        let npoint = params.nn + 1;
        let tol = 1e-3; // 1e-15 is achieved with 10_000
        params.lebesgue_estimate_nstation = 200; // 1e-15 is achieved with 10_000
        let interp = InterpLagrange::new(params).unwrap();
        let mut uu = Vector::new(npoint);
        interp.evaluate_f_over_nodes(&mut uu, f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.execute(x, &uu), f(i, x)));
        let nn_and_lebesgue = [
            (4, 1.798761778849085e+00),
            (8, 2.274730699116020e+00),
            (24, 2.984443326362511e+00),
        ];
        for (nn, lambda_times_nn) in nn_and_lebesgue {
            params.nn = nn;
            let interp = InterpLagrange::new(params).unwrap();
            approx_eq(interp.estimate_lebesgue_constant(), lambda_times_nn, tol);
        }
    }
}
