#![allow(non_snake_case)]

use crate::math::{chebyshev_gauss_points, chebyshev_lobatto_points, neg_one_pow_n};
use crate::StrError;
use crate::{mat_vec_mul, Matrix, Vector};

/// Defines the type of the interpolation grid in 1D
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridType {
    Uniform,
    ChebyshevGauss,
    ChebyshevGaussLobatto,
}

/// Implements a polynomial interpolant in Lagrange Form
///
/// A polynomial interpolant `I^X_N{f}` for the function `f(x)`, associated with a grid `X`, of degree `N`,
/// and with `N+1` points, is expressed in the Lagrange form as
/// (see Eq 2.2.19 of Ref #1, page 73; and Eq 3.31 of Ref #2, page 73):
///
/// ```text
///                         N
///                       —————
///            X          \             X
/// pnu(x) := I {f}(x) =  /      u  ⋅  ℓ (x)
///            N          —————   j     j
///                       j = 0
///
/// with uⱼ := f(xⱼ)
/// ```
///
/// where `ℓ^X_j(x)` is the j-th Lagrange cardinal polynomial associated with grid X and given by
/// (see Eq 2.3.27 of Ref #1, page 80; and Eq 3.32 of Ref #2, page 74):
///
/// ```text
///                      N
///                    ━━━━━
///             X      ┃   ┃  x  - Xᵢ
/// ell (x) := ℓ (x) = ┃   ┃  ———————
///    j        j      i = 0  Xⱼ - Xᵢ
///                    i ≠ j
///
/// 0 ≤ j ≤ N
/// ```
///
/// In barycentric form, the interpolant is expressed as
/// (see Eq 3.36 of Ref #2, page 74):
///
/// ```text
///                       N       λⱼ 
///                       Σ  uⱼ ——————
///            X         j=0    x - Xⱼ
/// pnu(x) := I {f}(x) = —————————————
///            N           N     λₖ
///                        Σ   ——————
///                       k=0  x - Xₖ
///
/// with uⱼ := f(xⱼ)
/// ```
///
/// where (see Eq 2.4.34 of Ref #1, page 90; and Eq 3.34 of Ref #2, page 74)
///
/// ```text
///             1
/// λⱼ = ————————————————
///         N
///         𝚷   (Xⱼ - Xᵢ)
///      i=0,i≠j
/// ```
///
/// Let us define (see Eq 2.4.34 of Ref #1, page 90):
///
/// ```text
///                         λⱼ
///                       ——————
///                       x - Xⱼ
/// bee (x) := ψ (x) = ———————————
///    j        j       N     λₖ
///                     Σ   ——————
///                    k=0  x - Xₖ
/// ```
///
/// Then:
///
/// ```text
///           N
/// pnu(x) =  Σ  uⱼ ⋅ pⱼ(x)
///          j=0
///
/// with pⱼ(x) = ell[j](x) = bee[j](x)
/// ```
///
/// An option to normalize the barycentric weights `λₖ` is available---they are
/// normalized and computed from `η` as follows:
///
/// ```text
///       N      N
/// ηₖ =  Σ      Σ    ln(|Xₖ - Xⱼ|)
///      k=0  j=0,j≠k
///
///      a ⋅ b                  k+N
/// λₖ = —————   with   a = (-1)
///       lf0
///
/// b = exp(m),  m = -ηk, and lf0 = 2ⁿ⁻¹/n
/// ```
///
/// or, if N > 700:
///
/// ```text
///      ⎛ a ⋅ b ⎞   ⎛  b  ⎞   ⎛  b  ⎞
/// λk = ⎜ ————— ⎟ ⋅ ⎜ ——— ⎟ ⋅ ⎜ ——— ⎟
///      ⎝  lf0  ⎠   ⎝ lf1 ⎠   ⎝ lf2 ⎠
///
/// b = exp(m/3)  and  lf0⋅lf1⋅lf2 = 2ⁿ⁻¹/n
/// ```
///
/// # Properties
///
/// The Lagrange polynomial `ℓᵢ` corresponding to node xᵢ has the property:
///
/// ```text
///          ⎧ 1  if i = j
/// ℓᵢ(xⱼ) = ⎨
///          ⎩ 0  if i ≠ j
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
/// 4. Berrut JP, Trefethen LN (2004) Barycentric Lagrange Interpolation,
///    SIAM Review Vol. 46, No. 3, pp. 501-517
#[derive(Clone, Debug)]
pub struct InterpLagrange {
    // general
    N: usize,   // degree: N = len(X)-1
    np1: usize, // number of points N + 1
    X: Vector,  // grid points: len(X) = P+1; generated in [-1, 1]
    U: Vector,  // function evaluated @ nodes: f(x_i)

    // barycentric
    Bary: bool,   // [default=true] use barycentric weights
    UseEta: bool, // [default=true] use ηk when computing D1
    Eta: Vector,  // sum of log of differences: ηk = Σ ln(|xk-xl|) (k≠l)
    Lam: Vector,  // normalized barycentric weights λk = pow(-1, k+N) ⋅ ηk / (2ⁿ⁻¹/n)

    // computed
    D1: Matrix, // (dℓj/dx)(xi)
    D2: Matrix, // (d²ℓj/dx²)(xi)
}

impl InterpLagrange {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    ///	* n -- degree
    /// * grid_type -- 1D grid type
    ///
    ///	**Note:** the grid will be generated in [-1, 1]
    pub fn new(N: usize, grid_type: GridType) -> Result<Self, StrError> {
        // check
        if N < 1 || N > 2048 {
            return Err("N must be in [1, 2048]");
        }

        // allocate
        let mut o = InterpLagrange {
            N,
            np1: N + 1,
            X: match grid_type {
                GridType::Uniform => Vector::linspace(-1.0, 1.0, N + 1).unwrap(),
                GridType::ChebyshevGauss => chebyshev_gauss_points(N),
                GridType::ChebyshevGaussLobatto => chebyshev_lobatto_points(N),
            },
            U: Vector::new(0),
            Bary: true,
            UseEta: true,
            Eta: Vector::new(N + 1),
            Lam: Vector::new(N + 1),
            D1: Matrix::new(0, 0),
            D2: Matrix::new(0, 0),
        };

        // compute eta
        //      N+1    N+1
        // ηk =  Σ      Σ    ln(|xk - xj|) ()
        //      k=0  j=0,j≠k
        for k in 0..o.np1 {
            for j in 0..o.np1 {
                if j != k {
                    o.Eta[k] += f64::ln(f64::abs(o.X[k] - o.X[j]));
                }
            }
        }

        // lambda factors
        let n = o.N as f64;
        let (lf0, lf1, lf2) = if o.N > 700 {
            (
                f64::powf(2.0, n / 3.0),
                f64::powf(2.0, n / 3.0),
                f64::powf(2.0, n / 3.0 - 1.0) / n,
            )
        } else {
            (f64::powf(2.0, n - 1.0) / n, 0.0, 0.0)
        };

        // compute lambda
        for k in 0..o.np1 {
            let a = neg_one_pow_n((k + o.N) as i32);
            let m = -o.Eta[k];
            if o.N > 700 {
                let b = f64::exp(m / 3.0);
                o.Lam[k] = a * b / lf0;
                o.Lam[k] *= b / lf1;
                o.Lam[k] *= b / lf2;
            } else {
                let b = f64::exp(m);
                o.Lam[k] = a * b / lf0;
            }
            assert!(o.Lam[k].is_finite());
        }
        Ok(o)
    }

    /// Computes the generating (nodal) polynomial associated with grid X
    ///
    /// The nodal polynomial is the unique polynomial of degree N+1 and
    /// leading coefficient whose zeros are the N+1 nodes of X.
    ///
    /// ```text
    ///	         N
    ///	 X      ━━━━
    ///	ω (x) = ┃  ┃ (x - X[i])
    ///	N+1     ┃  ┃
    ///	       i = 0
    /// ```
    pub fn Om(&self, x: f64) -> f64 {
        let mut ω = 1.0;
        for i in 0..self.np1 {
            ω *= x - self.X[i];
        }
        ω
    }

    /// Computes the i-th Lagrange cardinal polynomial associated with grid X
    ///
    /// Computes:
    ///
    /// ```text
    ///          N
    ///  X      ━━━━    x  -  X[j]
    /// ℓ (x) = ┃  ┃  —————————————
    ///  i      ┃  ┃   X[i] - X[j]
    ///        j = 0
    ///        j ≠ i
    ///
    /// 0 ≤ i ≤ N
    /// ```
    ///
    /// or (barycentric):
    ///
    /// ```text
    ///              λ[i]
    ///            ————————
    ///  X         x - x[i]
    /// ℓ (x) = ———————————————
    ///  i        N     λ[k]
    ///           Σ   ————————
    ///          k=0  x - x[k]
    /// ```
    ///
    /// # Input
    ///
    /// * `i` -- index of the Xᵢ point
    /// * `x` -- where to evaluate the polynomial
    ///
    /// # Output
    ///
    /// Returns `ℓ^X_i(x)`
    pub fn L(&self, i: usize, x: f64) -> f64 {
        if self.Bary {
            // barycentric formula
            if f64::abs(x - self.X[i]) < 10.0 * f64::EPSILON {
                return 1.0;
            }
            let mut sum = 0.0;
            for k in 0..self.np1 {
                sum += self.Lam[k] / (x - self.X[k]);
            }
            self.Lam[i] / (x - self.X[i]) / sum
        } else {
            // standard formula
            let mut res = 1.0;
            for j in 0..self.np1 {
                if i != j {
                    res *= (x - self.X[j]) / (self.X[i] - self.X[j]);
                }
            }
            res
        }
    }

    /// Evaluates the function f(x) over all nodes
    ///
    /// The function is `(i: usize, x: f64) -> f64`
    ///
    /// # Output
    ///
    /// The results are stores in the `U` variable
    pub fn CalcU<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, f64) -> f64,
    {
        if self.U.dim() != self.np1 {
            self.U = Vector::new(self.np1);
        }
        for i in 0..self.np1 {
            self.U[i] = f(i, self.X[i]);
        }
    }

    /// Performs the interpolation @ x
    ///
    /// Computes:
    ///
    /// ```text
    ///              N
    ///  X          ————          X
    /// I {f}(x) =  \     U[i] ⋅ ℓ (x)       with   U[i] = f(x[i])
    ///  N          /             i
    ///             ————
    ///             i = 0
    /// ```
    ///
    /// or (barycentric):
    ///
    /// ```text
    ///              N   λ[i] ⋅ f[i]
    ///              Σ   ———————————
    ///  X          i=0   x - x[i]
    /// I {f}(x) = ——————————————————
    ///  N            N     λ[i]
    ///               Σ   ————————
    ///              i=0  x - x[i]
    /// ```
    ///
    /// TODO: calculate U first automatically
    /// Maybe get U as an input argument instead
    ///
    /// NOTE: Uᵢ = f(xᵢ) must be calculated with o.CalcU or set first
    pub fn I(&self, x: f64) -> f64 {
        if self.Bary {
            // barycentric formula
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..self.np1 {
                let dx = x - self.X[i];
                if f64::abs(dx) < 10.0 * f64::EPSILON {
                    return self.U[i];
                }
                num += self.U[i] * self.Lam[i] / dx;
                den += self.Lam[i] / dx;
            }
            num / den
        } else {
            // standard formula
            let mut res = 0.0;
            for i in 0..self.np1 {
                res += self.U[i] * self.L(i, x);
            }
            res
        }
    }

    /// Computes the differentiation matrix D1 of the L function
    ///
    /// Computes:
    ///
    /// ```text
    /// dI{f}(x) │       N  dℓⱼ(x) │
    /// ———————— │    =  Σ  —————— │   ⋅ f(xⱼ)  =  D1ₖⱼ ⋅ f(xⱼ)
    ///    dx    │x=xₖ  j=0   dx   │x=xₖ          
    /// ```
    ///
    /// See Eq (3) of Reference #3
    pub fn CalcD1(&mut self) {
        // allocate matrix
        self.D1 = Matrix::new(self.np1, self.np1);

        if self.UseEta {
            // calculate D1 using ηk
            for k in 0..self.np1 {
                let mut sumRow = 0.0;
                for j in 0..self.np1 {
                    if k != j {
                        let r = neg_one_pow_n((k + j) as i32) * f64::exp(self.Eta[k] - self.Eta[j]);
                        let v = r / (self.X[k] - self.X[j]);
                        self.D1.set(k, j, v);
                        sumRow += v;
                    }
                }
                self.D1.set(k, k, -sumRow);
            }
        } else {
            // calculate D1 using λk
            for k in 0..self.np1 {
                let mut sumRow = 0.0;
                for j in 0..self.np1 {
                    if k != j {
                        let v = (self.Lam[j] / self.Lam[k]) / (self.X[k] - self.X[j]);
                        self.D1.set(k, j, v);
                        sumRow += v;
                    }
                }
                self.D1.set(k, k, -sumRow);
            }
        }
    }

    /// CalcD2 calculates the second derivative of the L function
    ///
    /// ```text
    /// d²I{f}(x) │       N  d²ℓⱼ(x) │    
    /// ————————— │    =  Σ  ——————— │   ⋅ f(xⱼ)  =  D2ₖⱼ f(xⱼ)
    ///    dx²    │x=xₖ  j=0   dx²   │x=xₖ
    /// ```
    ///
    /// See Eq (10) of Reference #3
    ///
    /// TODO:
    /// 1. Impl flag "already_calculated"
    /// 2. Handle the note below
    ///
    /// NOTE: this function will call CalcD1() because the D1 values required to compute D2
    pub fn CalcD2(&mut self) {
        // calculate D1
        self.CalcD1();

        // allocate matrix
        self.D2 = Matrix::new(self.np1, self.np1);

        // compute D2 from D1 values using Eqs. (9) and (13) of [3]
        for k in 0..self.np1 {
            let mut sumRow = 0.0;
            for j in 0..self.np1 {
                if k != j {
                    let v = 2.0 * self.D1.get(k, j) * (self.D1.get(k, k) - 1.0 / (self.X[k] - self.X[j]));
                    self.D2.set(k, j, v);
                    sumRow += v;
                }
            }
            self.D2.set(k, k, -sumRow);
        }
    }

    /// Computes the maximum error due to differentiation using the D1 matrix
    ///
    /// Computes the error @ `X[i]`
    ///
    /// # Input
    ///
    /// * `dfdxAna` -- function `(i: usize, x: f64) -> f64`
    ///
    /// NOTE: U and D1 matrix must be computed previously
    pub fn CalcErrorD1<F>(&self, mut dfdxAna: F) -> f64
    where
        F: FnMut(usize, f64) -> f64,
    {
        // derivative of interpolation @ x_i
        let mut v = Vector::new(self.np1);
        mat_vec_mul(&mut v, 1.0, &self.D1, &self.U).unwrap();

        // compute error
        let mut maxDiff = 0.0;
        for i in 0..self.np1 {
            let vana = dfdxAna(i, self.X[i]);
            let diff = f64::abs(v[i] - vana);
            if diff > maxDiff {
                maxDiff = diff;
            }
        }
        maxDiff
    }

    /// Computes the maximum error due to differentiation using the D2 matrix
    ///
    /// Computes the error @ `X[i]`
    ///
    /// # Input
    ///
    /// * `dfdxAna` -- function `(i: usize, x: f64) -> f64`
    ///
    /// NOTE: U and D2 matrix must be computed previously
    pub fn CalcErrorD2<F>(&self, mut d2fdx2Ana: F) -> f64
    where
        F: FnMut(usize, f64) -> f64,
    {
        // derivative of interpolation @ x_i
        let mut v = Vector::new(self.np1);
        mat_vec_mul(&mut v, 1.0, &self.D2, &self.U).unwrap();

        // compute error
        let mut maxDiff = 0.0;
        for i in 0..self.np1 {
            let vana = d2fdx2Ana(i, self.X[i]);
            let diff = f64::abs(v[i] - vana);
            if diff > maxDiff {
                maxDiff = diff;
            }
        }
        maxDiff
    }

    /// Estimates the Lebesgue constant ΛN
    ///
    /// The estimate is made with 10000 stations in `[-1, 1]`
    pub fn EstimateLebesgue(&self) -> f64 {
        let n_station = 10000; // generate several points along [-1,1]
        let mut ΛN = 0.0;
        for j in 0..n_station {
            let x = -1.0 + 2.0 * (j as f64) / ((n_station - 1) as f64);
            let mut sum = f64::abs(self.L(0, x));
            for i in 1..self.np1 {
                sum += f64::abs(self.L(i, x));
            }
            if sum > ΛN {
                ΛN = sum;
            }
        }
        ΛN
    }

    /// Estimates the maximum error
    ///
    /// Computes:
    ///
    /// ```text
    /// maxerr = max(|f(x) - I{f}(x)|)
    /// ```
    ///
    /// Estimates the maximum error using n_station in `[-1,1]`
    ///
    /// # Input
    ///
    /// * `n_station` -- ≥ 2; e.g. 10000
    /// * `f` -- function `(x: f64, i: usize) -> f64`
    ///
    /// # Output
    ///
    /// Returns `(max_err, i_max)` where `i_max` is the location of the max error
    pub fn EstimateMaxErr<F>(&self, n_station: usize, mut f: F) -> (f64, f64)
    where
        F: FnMut(f64, usize) -> f64,
    {
        assert!(n_station >= 2);
        let mut maxerr = 0.0;
        let mut xloc = 0.0;
        for i in 0..n_station {
            let x = -1.0 + 2.0 * (i as f64) / ((n_station - 1) as f64);
            let fx = f(x, i);
            let ix = self.I(x);
            let e = f64::abs(fx - ix);
            if e > maxerr {
                maxerr = e;
                xloc = x;
            }
        }
        (maxerr, xloc)
    }

    /// Executes a loop over the grid points
    ///
    /// Loops over `(i: usize, x: f64)`
    ///
    /// # Input
    ///
    /// * `callback` -- a function of `(i, x)` where `i` is the point number,
    ///   and `x` is the Cartesian coordinates of the point.
    pub fn loop_over_grid_points<F>(&self, mut callback: F)
    where
        F: FnMut(usize, f64),
    {
        for i in 0..self.np1 {
            callback(i, self.X[i]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{GridType, InterpLagrange};
    use crate::{approx_eq, deriv1_approx_eq, deriv2_approx_eq, Vector};

    // --- auxiliary: essential -----------------------------------------------------------------------

    fn check_lambda(N: usize, grid_type: GridType, tol: f64) {
        let interp = InterpLagrange::new(N, grid_type).unwrap();
        let m = f64::powf(2.0, (interp.N as f64) - 1.0) / (interp.N as f64);
        for i in 0..(interp.N + 1) {
            let mut d = 1.0;
            for j in 0..(interp.N + 1) {
                if i != j {
                    d *= interp.X[i] - interp.X[j]
                }
            }
            approx_eq(interp.Lam[i], 1.0 / d / m, tol);
        }
    }

    fn check_ell(N: usize, grid_type: GridType, tol_comparison: f64) {
        let mut interp = InterpLagrange::new(N, grid_type).unwrap();

        // check Kronecker property (standard)
        interp.Bary = false;
        for i in 0..(N + 1) {
            for j in 0..(N + 1) {
                let li = interp.L(i, interp.X[j]);
                let mut ana = 1.0;
                if i != j {
                    ana = 0.0;
                }
                approx_eq(li, ana, 1e-17);
            }
        }

        // check Kronecker property (barycentric)
        interp.Bary = true;
        for i in 0..(N + 1) {
            for j in 0..(N + 1) {
                let li = interp.L(i, interp.X[j]);
                let mut ana = 1.0;
                if i != j {
                    ana = 0.0;
                }
                approx_eq(li, ana, 1e-17);
            }
        }

        // compare standard and barycentric (L)
        let xx = Vector::linspace(-1.0, 1.0, 20).unwrap();
        for x in xx {
            for i in 0..(N + 1) {
                interp.Bary = true;
                let li1 = interp.L(i, x);
                interp.Bary = false;
                let li2 = interp.L(i, x);
                approx_eq(li1, li2, tol_comparison);
            }
        }
    }

    fn check_execute<F>(N: usize, grid_type: GridType, tol_comparison: f64, mut f: F)
    where
        F: Copy + FnMut(usize, f64) -> f64,
    {
        // calculate U
        let mut interp = InterpLagrange::new(N, grid_type).unwrap();
        interp.CalcU(f);

        // check interpolation (standard)
        interp.Bary = false;
        interp.loop_over_grid_points(|i, x| {
            approx_eq(interp.I(x), f(i, x), 1e-17);
        });

        // check interpolation (barycentric)
        interp.Bary = true;
        interp.loop_over_grid_points(|i, x| {
            approx_eq(interp.I(x), f(i, x), 1e-17);
        });

        // compare standard and barycentric (I)
        let xx = Vector::linspace(-1.0, 1.0, 20).unwrap();
        for x in xx {
            for _ in 0..(interp.N + 1) {
                interp.Bary = false;
                let i1 = interp.I(x);
                interp.Bary = true;
                let i2 = interp.I(x);
                approx_eq(i1, i2, tol_comparison);
            }
        }
    }

    // --- auxiliary: differentiation -----------------------------------------------------------------

    fn check_dd1_matrix(n: usize, grid_type: GridType, tol: f64) {
        let mut interp = InterpLagrange::new(n, grid_type).unwrap();
        interp.CalcD1();
        struct Args {}
        let args = &mut Args {};
        let np1 = n + 1;
        for i in 0..np1 {
            let xi = interp.X[i];
            for j in 0..np1 {
                deriv1_approx_eq(interp.D1.get(i, j), xi, args, tol, |x, _| Ok(interp.L(j, x)));
            }
        }
    }

    fn check_dd2_matrix(n: usize, grid_type: GridType, tol: f64) {
        let mut interp = InterpLagrange::new(n, grid_type).unwrap();
        interp.CalcD2();
        struct Args {}
        let args = &mut Args {};
        let np1 = n + 1;
        for i in 0..np1 {
            let xi = interp.X[i];
            for j in 0..np1 {
                deriv2_approx_eq(interp.D2.get(i, j), xi, args, tol, |x, _| Ok(interp.L(j, x)));
            }
        }
    }

    fn check_dd1_error<F, G>(nn: usize, grid_type: GridType, use_eta: bool, tol: f64, f: F, dfdx_ana: G)
    where
        F: FnMut(usize, f64) -> f64,
        G: FnMut(usize, f64) -> f64,
    {
        let mut interp = InterpLagrange::new(nn, grid_type).unwrap();
        interp.CalcU(f);
        interp.UseEta = use_eta;
        interp.CalcD1();
        let max_diff = interp.CalcErrorD1(dfdx_ana);
        if max_diff > tol {
            panic!("D1⋅U failed; max_diff = {:?}", max_diff);
        }
    }

    fn check_dd2_error<F, H>(nn: usize, grid_type: GridType, use_eta: bool, tol: f64, f: F, d2fdx2_ana: H)
    where
        F: FnMut(usize, f64) -> f64,
        H: FnMut(usize, f64) -> f64,
    {
        let mut interp = InterpLagrange::new(nn, grid_type).unwrap();
        interp.CalcU(f);
        interp.UseEta = use_eta;
        interp.CalcD2();
        let max_diff = interp.CalcErrorD2(d2fdx2_ana);
        if max_diff > tol {
            panic!("D2⋅U failed; max_diff = {:?}", max_diff);
        }
    }

    // --- tests --------------------------------------------------------------------------------------

    #[test]
    fn new_works() {
        let interp = InterpLagrange::new(2, GridType::Uniform).unwrap();
        assert_eq!(interp.N, 2);
        assert_eq!(interp.np1, 3);
        assert_eq!(interp.X.as_data(), &[-1.0, 0.0, 1.0]);
        assert_eq!(interp.U.dim(), 0);
        assert_eq!(interp.Bary, true);
        assert_eq!(interp.UseEta, true);
        assert_eq!(interp.Eta.dim(), 3);
        assert_eq!(interp.Lam.dim(), 3);
        assert_eq!(interp.D1.dims(), (0, 0));
        assert_eq!(interp.D2.dims(), (0, 0));
    }

    #[test]
    fn calc_u_and_i_works() {
        let f = |_, x| f64::cos(f64::exp(2.0 * x));
        let mut interp = InterpLagrange::new(5, GridType::Uniform).unwrap();
        interp.CalcU(f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.I(x), f(i, x)));
    }

    #[test]
    fn lambda_works() {
        for n in 1..20 {
            check_lambda(n, GridType::Uniform, 1e-12);
        }
        for n in 1..20 {
            check_lambda(n, GridType::ChebyshevGauss, 1e-14);
        }
        for n in 1..20 {
            check_lambda(n, GridType::ChebyshevGaussLobatto, 1e-14);
        }
    }

    #[test]
    fn ell_works() {
        for n in 1..20 {
            check_ell(n, GridType::Uniform, 1e-11);
        }
        for n in 1..20 {
            check_ell(n, GridType::ChebyshevGauss, 1e-14);
        }
        for n in 1..20 {
            check_ell(n, GridType::ChebyshevGaussLobatto, 1e-14);
        }
    }

    #[test]
    fn execute_works() {
        let f = |_, x| f64::cos(f64::exp(2.0 * x));
        for n in 1..20 {
            check_execute(n, GridType::Uniform, 1e-13, f);
        }
        for n in 1..20 {
            check_execute(n, GridType::ChebyshevGauss, 1e-14, f);
        }
        for n in 1..20 {
            check_execute(n, GridType::ChebyshevGaussLobatto, 1e-14, f);
        }
    }

    #[test]
    fn runge_equation_works() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut interp = InterpLagrange::new(8, GridType::Uniform).unwrap();
        interp.CalcU(f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.I(x), f(i, x)));
    }

    #[test]
    fn dd1_matrix_is_ok() {
        #[rustfmt::skip]
        let n_and_tols = [
            (2, 1e-12),
            (5, 1e-8),
            (10, 1e-8),
        ];
        for (n, tol) in n_and_tols {
            // println!("n = {:?}", n);
            check_dd1_matrix(n, GridType::ChebyshevGauss, tol);
        }
    }

    #[test]
    fn dd2_matrix_is_ok() {
        #[rustfmt::skip]
        let n_and_tols = [
            (2, 1e-9),
            (5, 1e-9),
            (10, 1e-9),
        ];
        for (n, tol) in n_and_tols {
            // println!("n = {:?}", n);
            check_dd2_matrix(n, GridType::ChebyshevGauss, tol);
        }
    }

    #[test]
    fn dd1_times_uu_is_ok() {
        let f = |_, x| f64::powf(x, 8.0);
        let g = |_, x| 8.0 * f64::powf(x, 7.0);
        for (nn, grid_type, use_eta, tol) in [
            (8, GridType::Uniform, false, 1e-13),
            (8, GridType::Uniform, true, 1e-13),
            (8, GridType::ChebyshevGauss, false, 1e-13),
            (8, GridType::ChebyshevGauss, true, 1e-14),
            (8, GridType::ChebyshevGaussLobatto, false, 1e-13),
            (8, GridType::ChebyshevGaussLobatto, true, 1e-13),
        ] {
            check_dd1_error(nn, grid_type, use_eta, tol, f, g);
        }
    }

    #[test]
    fn dd2_times_uu_is_ok() {
        let f = |_, x| f64::powf(x, 8.0);
        // let g = |_, x| 8.0 * f64::powf(x, 7.0);
        let h = |_, x| 56.0 * f64::powf(x, 6.0);
        for (nn, grid_type, use_eta, tol) in [
            (8, GridType::Uniform, false, 1e-11),
            (8, GridType::Uniform, true, 1e-11),
            (8, GridType::ChebyshevGauss, false, 1e-12),
            (8, GridType::ChebyshevGauss, true, 1e-12),
            (8, GridType::ChebyshevGaussLobatto, false, 1e-12),
            (8, GridType::ChebyshevGaussLobatto, true, 1e-12),
        ] {
            check_dd2_error(nn, grid_type, use_eta, tol, f, h);
        }
    }

    #[test]
    fn lebesgue_works_uniform() {
        let N = 5;
        let interp = InterpLagrange::new(N, GridType::Uniform).unwrap();
        approx_eq(interp.EstimateLebesgue(), 3.106301040275436e+00, 1e-15);
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut interp = InterpLagrange::new(8, GridType::ChebyshevGauss).unwrap();
        interp.CalcU(f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.I(x), f(i, x)));
        let n_and_lebesgue = [
            (4, 1.988854381999833e+00),
            (8, 2.361856787767076e+00),
            (24, 3.011792612349363e+00),
        ];
        for (n, lambda_times_n) in n_and_lebesgue {
            let interp = InterpLagrange::new(n, GridType::ChebyshevGauss).unwrap();
            approx_eq(interp.EstimateLebesgue(), lambda_times_n, 1e-14);
        }
    }

    #[test]
    fn lebesgue_works_chebyshev_gauss_lobatto() {
        // Runge equation
        let f = |_, x| 1.0 / (1.0 + 16.0 * x * x);
        let mut interp = InterpLagrange::new(8, GridType::ChebyshevGaussLobatto).unwrap();
        interp.CalcU(f);
        interp.loop_over_grid_points(|i, x| assert_eq!(interp.I(x), f(i, x)));
        let n_and_lebesgue = [
            (4, 1.798761778849085e+00),
            (8, 2.274730699116020e+00),
            (24, 2.984443326362511e+00),
        ];
        for (n, lambda_times_n) in n_and_lebesgue {
            let interp = InterpLagrange::new(n, GridType::ChebyshevGaussLobatto).unwrap();
            approx_eq(interp.EstimateLebesgue(), lambda_times_n, 1e-14);
        }
    }
}
