use crate::math::{chebyshev_gauss_points, chebyshev_lobatto_points, neg_one_pow_n};
use crate::StrError;
use crate::{Matrix, Vector};

/// Tiny number to consider x an Xⱼ identical
const DX_EPSILON: f64 = 10.0 * f64::EPSILON;

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
}

impl InterpParams {
    /// Allocates a new instance with default values
    ///
    /// # Input
    ///
    /// * `nn` -- the polynomial degree `N`; thus the number of grid nodes will be `N + 1`.
    ///   **Note:** `nn` must be in `[1, 2048]`
    pub fn new(nn: usize) -> Result<Self, StrError> {
        if nn < 1 || nn > 2048 {
            return Err("the polynomial degree must be in [1, 2048]");
        }
        Ok(InterpParams {
            nn,
            grid_type: InterpGrid::ChebyshevGaussLobatto,
            no_eta_normalization: false,
            eta_cutoff: 700,
            lebesgue_estimate_nstation: 10_000,
        })
    }

    /// Validates the parameters
    fn validate(&self) -> Result<(), StrError> {
        if self.nn < 1 || self.nn > 2048 {
            return Err("the polynomial degree must be in [1, 2048]");
        }
        if self.lebesgue_estimate_nstation < 2 {
            return Err("lebesgue_estimate_nstation must be ≥ 2");
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
/// (see Eq 2.2.19 of Reference #1, page 73; and Eq 3.31 of Reference #2, page 73):
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
/// (see Eq 2.3.27 of Reference #1, page 80; and Eq 3.32 of Reference #2, page 74):
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
/// (see Eq 3.36 of Reference #2, page 74):
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
/// where (see Eq 2.4.34 of Reference #1, page 90; and Eq 3.34 of Reference #2, page 74)
///
/// ```text
///             1
/// λⱼ = ————————————————
///         N
///         𝚷   (Xⱼ - Xₖ)
///      k=0,k≠j
/// ```
///
/// Let us define (see Eq 2.4.34 of Reference #1, page 90):
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
/// The strategy is to normalize lambda using the so-called eta-factors (`η`) as follows (See Reference #3 and #4).
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

    /// Computes the i-th polynomial associated with grid X (barycentric form)
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
    /// * `x` -- the coordinate to evaluate the polynomial; must satisfy -1 ≤ j ≤ 1
    pub fn psi(&self, j: usize, x: f64) -> Result<f64, StrError> {
        if j > self.params.nn {
            return Err("j must be in 0 ≤ j ≤ N");
        }
        if x < -1.0 || x > 1.0 {
            return Err("x must be in -1 ≤ x ≤ 1");
        }
        if f64::abs(x - self.xx[j]) < DX_EPSILON {
            return Ok(1.0);
        }
        let mut sum = 0.0;
        for k in 0..self.npoint {
            sum += self.lambda[k] / (x - self.xx[k]);
        }
        Ok(self.lambda[j] / (x - self.xx[j]) / sum)
    }

    /// Evaluates the interpolation
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
    /// * `x` -- the coordinate to evaluate the polynomial; must satisfy -1 ≤ j ≤ 1
    /// * `uu` -- the "data" vector `U` of size equal to `N + 1`
    pub fn eval(&self, x: f64, uu: &Vector) -> Result<f64, StrError> {
        if x < -1.0 || x > 1.0 {
            return Err("x must be in -1 ≤ x ≤ 1");
        }
        if uu.dim() != self.npoint {
            return Err("the dimension of the data vector U must be equal to N + 1");
        }
        let mut res = 0.0;
        for j in 0..self.npoint {
            res += uu[j] * self.psi(j, x).unwrap();
        }
        Ok(res)
    }

    /// Evaluates the first derivative using the interpolating polynomial
    ///
    /// Calculates the first derivative of `pnu(x)` described in [InterpLagrange::eval()]
    ///
    /// For `x` not coinciding with any node:
    ///
    /// ```text
    ///             N    λⱼ   pnu(x) - Uⱼ
    ///             Σ  —————— ———————————
    /// d pnu(x)   j=0 x - Xⱼ   x - Xⱼ
    /// ———————— = ——————————————————————
    ///    dx            N     λⱼ
    ///                  Σ   ——————
    ///                 j=0  x - Xⱼ
    /// ```
    ///
    /// See Equation 3.45 of Reference #2.
    ///
    /// If `x` coincides with a node
    ///
    /// ```text
    /// d pnu(Xₖ)     1   N     Uₖ - Uⱼ
    /// ————————— = - ——  Σ  λⱼ ———————
    ///     dx        λₖ j=0    Xₖ - Xⱼ
    ///                  j≠k
    /// ```
    ///
    /// See Equation 3.46 of Reference #2. See also [InterpLagrange::calc_dd1_matrix()].
    ///
    /// # Input
    ///
    /// * `x` -- the coordinate to evaluate the derivative; must satisfy -1 ≤ j ≤ 1
    /// * `uu` -- the "data" vector `U` of size equal to `N + 1`
    pub fn eval_deriv1(&self, x: f64, uu: &Vector) -> Result<f64, StrError> {
        if x < -1.0 || x > 1.0 {
            return Err("x must be in -1 ≤ x ≤ 1");
        }
        if uu.dim() != self.npoint {
            return Err("the dimension of the data vector U must be equal to N + 1");
        }
        let mut at_node = false;
        let mut at_node_index = 0;
        if x == -1.0 {
            at_node = true;
            at_node_index = 0;
        } else if x == 1.0 {
            at_node = true;
            at_node_index = self.params.nn;
        } else {
            for j in 0..self.npoint {
                let dx = x - self.xx[j];
                if f64::abs(dx) < DX_EPSILON {
                    at_node = true;
                    at_node_index = j;
                    break;
                }
            }
        }
        if at_node {
            let k = at_node_index;
            let mut sum = 0.0;
            for j in 0..self.npoint {
                if j != k {
                    sum += self.lambda[j] * (uu[k] - uu[j]) / (self.xx[k] - self.xx[j]);
                }
            }
            Ok(-sum / self.lambda[k])
        } else {
            let pnu_x = self.eval(x, uu).unwrap();
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..self.npoint {
                let dx = x - self.xx[j];
                let a = self.lambda[j] / dx;
                num += a * (pnu_x - uu[j]) / dx;
                den += a;
            }
            Ok(num / den)
        }
    }

    /// Evaluates the second derivative using the interpolating polynomial
    ///
    /// Calculates the second derivative of `pnu(x)` described in [InterpLagrange::eval()]
    ///
    /// For `x` not coinciding with any node:
    ///
    /// ```text
    ///             N     λⱼ     2    ⎛           Uⱼ - pnu(x) ⎞
    ///             Σ   —————— —————— ⎜ pnu'(x) + ——————————— ⎟
    /// d²pnu(x)   j=0  x - Xⱼ x - Xⱼ ⎝              x - Xⱼ   ⎠
    /// ———————— = ————————————————————————————————————————————
    ///    dx²                    N     λⱼ
    ///                           Σ   ——————
    ///                          j=0  x - Xⱼ
    /// ```
    ///
    /// Note: The above equation was derived using Mathematica.
    ///
    /// If `x` coincides with a node (see [InterpLagrange::calc_dd2_matrix()]):
    ///
    /// ```text
    /// d²pnu(Xₖ)        N
    /// ————————— = - 2  Σ  D1ₖⱼ [D1ₖₖ - (Xₖ - Xⱼ)⁻¹] (Uₖ - Uⱼ)
    ///    dx²          j=0
    ///                 j≠k
    /// ```
    ///
    /// where:
    ///
    /// ```text
    /// D1ₖⱼ = (λⱼ/λₖ) / (Xₖ-Xⱼ)
    ///
    /// and
    ///
    /// D1ₖₖ = - Σ_(j=0,j≠k)^N D1ₖⱼ   (negative sum trick)
    /// ```
    ///
    /// See also [InterpLagrange::calc_dd1_matrix()].
    ///
    /// # Input
    ///
    /// * `x` -- the coordinate to evaluate the derivative; must satisfy -1 ≤ j ≤ 1
    /// * `uu` -- the "data" vector `U` of size equal to `N + 1`
    pub fn eval_deriv2(&self, x: f64, uu: &Vector) -> Result<f64, StrError> {
        if x < -1.0 || x > 1.0 {
            return Err("x must be in -1 ≤ x ≤ 1");
        }
        if uu.dim() != self.npoint {
            return Err("the dimension of the data vector U must be equal to N + 1");
        }
        let mut at_node = false;
        let mut at_node_index = 0;
        if x == -1.0 {
            at_node = true;
            at_node_index = 0;
        } else if x == 1.0 {
            at_node = true;
            at_node_index = self.params.nn;
        } else {
            for j in 0..self.npoint {
                let dx = x - self.xx[j];
                if f64::abs(dx) < DX_EPSILON {
                    at_node = true;
                    at_node_index = j;
                    break;
                }
            }
        }
        if at_node {
            // compute dkk (using NST: negative sum trick)
            let k = at_node_index;
            let mut sum = 0.0;
            for j in 0..self.npoint {
                if j != k {
                    sum += self.lambda[j] / (self.xx[k] - self.xx[j]);
                }
            }
            let d1kk = -sum / self.lambda[k];
            // compute the second derivative
            sum = 0.0;
            for j in 0..self.npoint {
                if j != k {
                    let dx = self.xx[k] - self.xx[j];
                    let d1kj = (self.lambda[j] / self.lambda[k]) / dx;
                    sum += d1kj * (d1kk - 1.0 / dx) * (uu[k] - uu[j]);
                }
            }
            Ok(-2.0 * sum)
        } else {
            let pnu_x = self.eval(x, uu).unwrap();
            let d_pnu_x = self.eval_deriv1(x, uu).unwrap();
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..self.npoint {
                let dx = x - self.xx[j];
                let a = self.lambda[j] / dx;
                let b = 2.0 / dx;
                let c = d_pnu_x + (uu[j] - pnu_x) / dx;
                num += a * b * c;
                den += a;
            }
            Ok(num / den)
        }
    }

    /// Computes the differentiation matrix D1
    ///
    /// Calculates:
    ///
    /// ```text
    /// dI{f}(x) │       N  dℓⱼ(x) │
    /// ———————— │    =  Σ  —————— │   ⋅ f(xⱼ)  =  D1ₖⱼ ⋅ f(xⱼ)
    ///    dx    │x=xₖ  j=0   dx   │x=xₖ          
    /// ```
    ///
    /// Where (no_eta form):
    ///
    /// ```text
    ///        ⎧ (λⱼ/λₖ) / (Xₖ-Xⱼ)     if k ≠ j
    /// D1ₖⱼ = ⎨
    ///        ⎩ - Σ_(m=0,m≠k)^N D1ₖₘ  if k = j  (negative sum trick, NST)
    /// ```
    ///
    /// See Eqs 6, 7, and 9 in Reference #3. Note that `cⱼ = λⱼ⁻¹` in Reference #3;
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
    /// Calculates:
    ///
    /// ```text
    /// d²I{f}(x) │       N  d²ℓⱼ(x) │    
    /// ————————— │    =  Σ  ——————— │   ⋅ f(xⱼ)  =  D2ₖⱼ f(xⱼ)
    ///    dx²    │x=xₖ  j=0   dx²   │x=xₖ
    /// ```
    ///
    /// Where (recursive formula):
    ///
    /// ```text
    ///        ⎧ 2 D1ₖⱼ [D1ₖₖ - (Xₖ - Xⱼ)⁻¹]  if k ≠ j
    /// D2ₖⱼ = ⎨
    ///        ⎩ - Σ_(m=0,m≠k)^N D2ₖₘ         if k = j  (negative sum trick, NST)
    /// ```
    ///
    /// See Eqs 9 and 13 in Reference #3.
    pub fn calc_dd2_matrix(&mut self) {
        // calculate D1
        self.calc_dd1_matrix();

        // allocate matrix
        if self.dd2.dims().0 == self.npoint {
            return; // already calculated
        }
        self.dd2 = Matrix::new(self.npoint, self.npoint);

        // compute D2 from D1 (recursion formula; see Eqs 9 and 13 of Reference #3)
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

    /// Estimates the Lebesgue constant ΛN
    pub fn estimate_lebesgue_constant(&self) -> f64 {
        let n_station = self.params.lebesgue_estimate_nstation;
        let mut lambda_times_nn = 0.0;
        for j in 0..n_station {
            let x = -1.0 + 2.0 * (j as f64) / ((n_station - 1) as f64);
            let mut sum = f64::abs(self.psi(0, x).unwrap());
            for i in 1..self.npoint {
                sum += f64::abs(self.psi(i, x).unwrap());
            }
            if sum > lambda_times_nn {
                lambda_times_nn = sum;
            }
        }
        lambda_times_nn
    }

    /// Returns a reference to the grid nodes
    pub fn get_points(&self) -> &Vector {
        &self.xx
    }

    /// Returns the (min, max) coordinates
    pub fn get_xrange(&self) -> (f64, f64) {
        (-1.0, 1.0)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{InterpGrid, InterpLagrange, InterpParams};
    use crate::{approx_eq, deriv1_approx_eq, deriv2_approx_eq, mat_vec_mul, Vector};

    #[test]
    fn params_new_and_validate_capture_errors() {
        assert_eq!(
            InterpParams::new(0).err(),
            Some("the polynomial degree must be in [1, 2048]")
        );
        assert_eq!(
            InterpParams::new(2049).err(),
            Some("the polynomial degree must be in [1, 2048]")
        );
        let params = InterpParams {
            nn: 0,
            grid_type: InterpGrid::Uniform,
            no_eta_normalization: false,
            eta_cutoff: 0,
            lebesgue_estimate_nstation: 2,
        };
        assert_eq!(
            params.validate().err(),
            Some("the polynomial degree must be in [1, 2048]")
        );
    }

    #[test]
    fn new_captures_errors() {
        let mut params = InterpParams::new(1).unwrap();
        params.lebesgue_estimate_nstation = 1;
        assert_eq!(
            InterpLagrange::new(params).err(),
            Some("lebesgue_estimate_nstation must be ≥ 2")
        );
    }

    #[test]
    fn new_works() {
        let params = InterpParams::new(2).unwrap();
        let interp = InterpLagrange::new(params).unwrap();
        assert_eq!(interp.npoint, 3);
        assert_eq!(interp.xx.as_data(), &[-1.0, 0.0, 1.0]);
        assert_eq!(interp.eta.dim(), 3);
        assert_eq!(interp.lambda.dim(), 3);
        assert_eq!(interp.dd1.dims(), (0, 0));
        assert_eq!(interp.dd2.dims(), (0, 0));
    }

    #[test]
    fn getters_work() {
        let params = InterpParams::new(2).unwrap();
        let interp = InterpLagrange::new(params).unwrap();
        assert_eq!(interp.get_points().as_data(), &[-1.0, 0.0, 1.0]);
        assert_eq!(interp.get_xrange(), (-1.0, 1.0));
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
        let mut params = InterpParams::new(1).unwrap();
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
                let psi = interp.psi(i, interp.xx[j]).unwrap();
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
                let psi_j = interp.psi(j, x).unwrap();
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
        let mut params = InterpParams::new(1).unwrap();
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

    fn check_eval<F>(params: InterpParams, tol: f64, mut f: F)
    where
        F: Copy + FnMut(f64) -> f64,
    {
        // interpolant
        let npoint = params.nn + 1;
        let interp = InterpLagrange::new(params).unwrap();

        // compute data points
        let mut uu = Vector::new(npoint);
        for (i, x) in interp.get_points().into_iter().enumerate() {
            uu[i] = f(*x);
        }

        // check the interpolation @ nodes
        for x in interp.get_points() {
            assert_eq!(interp.eval(*x, &uu).unwrap(), f(*x));
        }

        // check the interpolation over the xrange
        let nstation = 20;
        let station = Vector::linspace(-1.0, 1.0, nstation).unwrap();
        for i in 0..nstation {
            approx_eq(interp.eval(station[i], &uu).unwrap(), f(station[i]), tol);
        }
    }

    #[test]
    fn eval_works_1() {
        let f = |x| f64::cos(f64::exp(2.0 * x));
        let mut params = InterpParams::new(5).unwrap();
        for grid_type in [
            InterpGrid::Uniform,
            InterpGrid::ChebyshevGauss,
            InterpGrid::ChebyshevGaussLobatto,
        ] {
            params.grid_type = grid_type;
            check_eval(params, 1.5, f); // TODO: check this
        }
    }

    #[test]
    fn eval_works_2() {
        // Runge equation
        let f = |x| 1.0 / (1.0 + 16.0 * x * x);
        let mut params = InterpParams::new(8).unwrap();
        for grid_type in [
            InterpGrid::Uniform,
            InterpGrid::ChebyshevGauss,
            InterpGrid::ChebyshevGaussLobatto,
        ] {
            params.grid_type = grid_type;
            check_eval(params, 0.69, f);
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
                if i == 0 || i == params.nn {
                    // TODO: find another method because we
                    // cannot use central differences @ x=-1 and x=1
                } else {
                    deriv1_approx_eq(interp.dd1.get(i, j), xi, args, tol, |x, _| {
                        Ok(interp.psi(j, x).unwrap())
                    });
                }
            }
        }
    }

    #[test]
    fn dd1_matrix_works() {
        // with eta
        let nn_and_tols = [(2, 1e-12), (5, 1e-9), (10, 1e-8)];
        let mut params = InterpParams::new(1).unwrap();
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd1_matrix(params, tol);
        }
        // with eta and low cutoff
        let nn_and_tols = [(2, 1e-12), (5, 1e-9), (10, 1e-8)];
        params.eta_cutoff = 0;
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd1_matrix(params, tol);
        }
        // no eta
        let nn_and_tols = [(2, 1e-12), (5, 1e-9), (10, 1e-8)];
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
                if i == 0 || i == params.nn {
                    // TODO: find another method because we
                    // cannot use central differences @ x=-1 and x=1
                } else {
                    deriv2_approx_eq(interp.dd2.get(i, j), xi, args, tol, |x, _| {
                        Ok(interp.psi(j, x).unwrap())
                    });
                }
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
        let mut params = InterpParams::new(1).unwrap();
        for (nn, tol) in nn_and_tols {
            params.nn = nn;
            // println!("nn = {:?}", nn);
            check_dd2_matrix(params, tol);
        }
    }

    fn check_dd1_error<F, G>(params: InterpParams, tol: f64, mut f: F, mut g: G)
    where
        F: FnMut(f64) -> f64,
        G: FnMut(f64) -> f64,
    {
        // interpolant
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();

        // compute data points
        let mut uu = Vector::new(npoint);
        for (i, x) in interp.get_points().into_iter().enumerate() {
            uu[i] = f(*x);
        }

        // derivative of interpolation @ all nodes
        interp.calc_dd1_matrix();
        let mut num = Vector::new(npoint);
        mat_vec_mul(&mut num, 1.0, &interp.dd1, &uu).unwrap();

        // check the maximum error due to differentiation using the D1 matrix
        let mut max_err = 0.0;
        for i in 0..npoint {
            let ana = g(interp.xx[i]);
            let diff = f64::abs(num[i] - ana);
            if diff > max_err {
                max_err = diff;
            }
        }
        approx_eq(max_err, 0.0, tol);
    }

    #[test]
    fn dd1_times_uu_works() {
        let mut params = InterpParams::new(1).unwrap();
        let f = |x| f64::powf(x, 8.0);
        let g = |x| 8.0 * f64::powf(x, 7.0);
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

    fn check_dd2_error<F, H>(params: InterpParams, tol: f64, mut f: F, mut h: H)
    where
        F: FnMut(f64) -> f64,
        H: FnMut(f64) -> f64,
    {
        // interpolant
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();

        // compute data points
        let mut uu = Vector::new(npoint);
        for (i, x) in interp.get_points().into_iter().enumerate() {
            uu[i] = f(*x);
        }

        // derivative of interpolation @ all nodes
        interp.calc_dd2_matrix();
        let mut num = Vector::new(npoint);
        mat_vec_mul(&mut num, 1.0, &interp.dd2, &uu).unwrap();

        // check the maximum error due to differentiation using the D2 matrix
        let mut max_err = 0.0;
        for i in 0..npoint {
            let ana = h(interp.xx[i]);
            let diff = f64::abs(num[i] - ana);
            if diff > max_err {
                max_err = diff;
            }
        }
        approx_eq(max_err, 0.0, tol);
    }

    #[test]
    fn dd2_times_uu_works() {
        let mut params = InterpParams::new(1).unwrap();
        let f = |x| f64::powf(x, 8.0);
        let h = |x| 56.0 * f64::powf(x, 6.0);
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

    #[test]
    fn dd_matrices_are_computed_just_once() {
        // interpolant
        let nn = 8;
        let params = InterpParams::new(nn).unwrap();
        let mut interp = InterpLagrange::new(params).unwrap();

        // calculate D1 and D2
        interp.calc_dd1_matrix();
        interp.calc_dd2_matrix();

        // D1 and D2 should not be computed again; use debug
        interp.calc_dd1_matrix();
        interp.calc_dd2_matrix();
    }

    // --- derivatives of polynomial function ---------------------------------------------------------

    fn check_eval_deriv1<F, G>(params: InterpParams, tol: f64, mut f: F, mut g: G)
    where
        F: Copy + FnMut(f64) -> f64,
        G: Copy + FnMut(f64) -> f64,
    {
        // interpolant
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();

        // compute data points
        let mut uu = Vector::new(npoint);
        for (i, x) in interp.get_points().into_iter().enumerate() {
            uu[i] = f(*x);
        }

        // calculate the derivative at all nodes
        interp.calc_dd1_matrix();
        let mut d1_at_nodes = Vector::new(npoint);
        mat_vec_mul(&mut d1_at_nodes, 1.0, &interp.dd1, &uu).unwrap();

        // check the interpolation over the xrange
        let nstation = 20;
        let stations = Vector::linspace(-1.0, 1.0, nstation).unwrap();
        for i in 0..nstation {
            // println!("x[{}] = {:?}", i, stations[i]);
            let d1 = interp.eval_deriv1(stations[i], &uu).unwrap();
            approx_eq(d1, g(stations[i]), tol);
            if i == 0 || i == (nstation - 1) {
                // at node
                let j = if i == 0 { 0 } else { npoint - 1 };
                assert!(f64::abs(stations[i] - interp.xx[j]) < 10.0 * f64::EPSILON);
                // println!("at node {}: x[{}] = {:?}: d1 = {:?} =? {:?}", j, i, stations[i], d1, d1_at_nodes[j]);
                approx_eq(d1, d1_at_nodes[j], 1e-14);
            }
        }

        // check at node near the middle
        let x_mid = interp.xx[params.nn / 2];
        // println!("x_mid = {:?}", x_mid);
        let d1 = interp.eval_deriv1(x_mid, &uu).unwrap();
        approx_eq(d1, g(x_mid), tol);
    }

    #[test]
    fn eval_deriv1_works() {
        let f = |x| f64::powf(x, 8.0);
        let g = |x| 8.0 * f64::powf(x, 7.0);
        let params = InterpParams::new(8).unwrap();
        check_eval_deriv1(params, 1e-13, f, g)
    }

    fn check_eval_deriv2<F, H>(params: InterpParams, tol: f64, mut f: F, mut h: H)
    where
        F: Copy + FnMut(f64) -> f64,
        H: Copy + FnMut(f64) -> f64,
    {
        // interpolant
        let npoint = params.nn + 1;
        let mut interp = InterpLagrange::new(params).unwrap();

        // compute data points
        let mut uu = Vector::new(npoint);
        for (i, x) in interp.get_points().into_iter().enumerate() {
            uu[i] = f(*x);
        }

        // calculate the derivative at all nodes
        interp.calc_dd2_matrix();
        let mut d2_at_nodes = Vector::new(npoint);
        mat_vec_mul(&mut d2_at_nodes, 1.0, &interp.dd2, &uu).unwrap();

        // check the interpolation over the xrange
        let nstation = 20;
        let stations = Vector::linspace(-1.0, 1.0, nstation).unwrap();
        for i in 0..nstation {
            // println!("x[{}] = {:?}", i, stations[i]);
            let d2 = interp.eval_deriv2(stations[i], &uu).unwrap();
            approx_eq(d2, h(stations[i]), tol);
            if i == 0 || i == (nstation - 1) {
                // at node
                let j = if i == 0 { 0 } else { npoint - 1 };
                assert!(f64::abs(stations[i] - interp.xx[j]) < 10.0 * f64::EPSILON);
                // println!("at node {}: x[{}] = {:?}: d1 = {:?} =? {:?}", j, i, stations[i], d2, d2_at_nodes[j]);
                approx_eq(d2, d2_at_nodes[j], 1e-12);
            }
        }

        // check at node near the middle
        let x_mid = interp.xx[params.nn / 2];
        // println!("x_mid = {:?}", x_mid);
        let d2 = interp.eval_deriv2(x_mid, &uu).unwrap();
        approx_eq(d2, h(x_mid), tol);
    }

    #[test]
    fn eval_deriv2_works() {
        let f = |x| f64::powf(x, 8.0);
        let h = |x| 56.0 * f64::powf(x, 6.0);
        let params = InterpParams::new(8).unwrap();
        check_eval_deriv2(params, 1e-12, f, h)
    }

    // --- Lebesgue -----------------------------------------------------------------------------------

    #[test]
    fn lebesgue_works_uniform() {
        let mut params = InterpParams::new(5).unwrap();
        params.grid_type = InterpGrid::Uniform;
        let tol = 1e-3;
        params.lebesgue_estimate_nstation = 210; // 1e-15 is achieved with 10_000
        let interp = InterpLagrange::new(params).unwrap();
        approx_eq(interp.estimate_lebesgue_constant(), 3.106301040275436e+00, tol);
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss() {
        let mut params = InterpParams::new(1).unwrap();
        let data = [
            (4, 1e-14, 1.988854381999833e+00),
            (8, 1e-15, 2.361856787767076e+00),
            (24, 1e-14, 3.011792612349363e+00),
        ];
        for (nn, tol, reference) in data {
            println!("nn = {}", nn);
            params.nn = nn;
            params.grid_type = InterpGrid::ChebyshevGauss;
            params.lebesgue_estimate_nstation = 10_000;
            let interp = InterpLagrange::new(params).unwrap();
            approx_eq(interp.estimate_lebesgue_constant(), reference, tol);
        }
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss_lobatto() {
        let mut params = InterpParams::new(1).unwrap();
        let data = [
            (4, 1e-15, 1.798761778849085e+00),
            (8, 1e-15, 2.274730699116020e+00),
            (24, 1e-14, 2.984443326362511e+00),
        ];
        for (nn, tol, reference) in data {
            println!("nn = {}", nn);
            params.nn = nn;
            params.grid_type = InterpGrid::ChebyshevGaussLobatto;
            params.lebesgue_estimate_nstation = 10_000;
            let interp = InterpLagrange::new(params).unwrap();
            approx_eq(interp.estimate_lebesgue_constant(), reference, tol);
        }
    }

    // --- errors -------------------------------------------------------------------------------------

    #[test]
    fn functions_check_ranges() {
        let params = InterpParams::new(2).unwrap();
        let interp = InterpLagrange::new(params).unwrap();
        let uu = Vector::new(0);
        // psi
        assert_eq!(interp.psi(100, -1.0).err(), Some("j must be in 0 ≤ j ≤ N"));
        assert_eq!(interp.psi(0, -2.0).err(), Some("x must be in -1 ≤ x ≤ 1"));
        // eval
        assert_eq!(interp.eval(-2.0, &uu).err(), Some("x must be in -1 ≤ x ≤ 1"));
        assert_eq!(
            interp.eval(-1.0, &uu).err(),
            Some("the dimension of the data vector U must be equal to N + 1")
        );
        // eval_deriv1
        assert_eq!(interp.eval_deriv1(-2.0, &uu).err(), Some("x must be in -1 ≤ x ≤ 1"));
        assert_eq!(
            interp.eval_deriv1(-1.0, &uu).err(),
            Some("the dimension of the data vector U must be equal to N + 1")
        );
        // eval_deriv2
        assert_eq!(interp.eval_deriv2(-2.0, &uu).err(), Some("x must be in -1 ≤ x ≤ 1"));
        assert_eq!(
            interp.eval_deriv2(-1.0, &uu).err(),
            Some("the dimension of the data vector U must be equal to N + 1")
        );
    }
}
