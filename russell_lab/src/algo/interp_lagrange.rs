#![allow(non_snake_case)]

use crate::math::{chebyshev_gauss_points, chebyshev_lobatto_points, neg_one_pow_n};
use crate::StrError;
use crate::{mat_vec_mul, Matrix, Vector};
use serde::{Deserialize, Serialize};

/// Defines the type of the interpolation grid in 1D
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum GridType {
    Uniform,
    ChebyshevGauss,
    ChebyshevGaussLobatto,
}

/// Implements Lagrange interpolators associated with a grid X
///
/// An interpolant I^X_N{f} (associated with a grid X; of degree N; with N+1 points)
/// is expressed in the Lagrange form as follows:
///
/// ```text
///              N
///  X          ————             X
/// I {f}(x) =  \     f(x[i]) ⋅ ℓ (x)
///  N          /                i
///             ————
///             i = 0
/// ```
///
/// where `ℓ^X_i(x)` is the i-th Lagrange cardinal polynomial associated with grid X and given by:
///
/// ```text
///          N
///  N      ━━━━    x  -  X[j]
/// ℓ (x) = ┃  ┃  —————————————
///  i      ┃  ┃   X[i] - X[j]
///        j = 0
///        j ≠ i
///
/// 0 ≤ i ≤ N
/// ```
///
/// or, barycentric form:
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
/// with:
///
/// ```text
///              λ[i]
///            ————————
///  N         x - x[i]
/// ℓ (x) = ———————————————
///  i        N     λ[k]
///           Σ   ————————
///          k=0  x - x[k]
/// ```
///
/// The barycentric weights `λk` are normalized and computed from `ηk` as follows:
///
/// ```text
/// ηk = Σ ln(|xk-xl|) (k≠l)
///
///       a ⋅ b             k+N
/// λk =  —————     a = (-1)
///        lf0
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
/// * Canuto C, Hussaini MY, Quarteroni A, Zang TA (2006) Spectral Methods: Fundamentals in
///   Single Domains. Springer. 563p
/// * Berrut JP, Trefethen LN (2004) Barycentric Lagrange Interpolation,
///   SIAM Review Vol. 46, No. 3, pp. 501-517
/// * Costa B, Don WS (2000) On the computation of high order pseudospectral derivatives,
///   Applied Numerical Mathematics, 33:151-159.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InterpLagrange {
    // general
    N: usize,    // degree: N = len(X)-1
    np1: usize,  // number of points N + 1
    X: Vec<f64>, // grid points: len(X) = P+1; generated in [-1, 1]
    U: Vector,   // function evaluated @ nodes: f(x_i)

    // barycentric
    Bary: bool,    // [default=true] use barycentric weights
    UseEta: bool,  // [default=true] use ηk when computing D1
    Eta: Vec<f64>, // sum of log of differences: ηk = Σ ln(|xk-xl|) (k≠l)
    Lam: Vec<f64>, // normalized barycentric weights λk = pow(-1, k+N) ⋅ ηk / (2ⁿ⁻¹/n)

    // computed
    D1: Matrix, // (dℓj/dx)(xi)
    D2: Matrix, // (d²ℓj/dx²)(xi)
}

// Holds interpolators; e.g. for 2D or 3D applications
pub type InterpLagrangeSet = Vec<InterpLagrange>;

// impl InterpLagrangeSet {
//     // Allocates a new instance
//     pub fn new(ndim: usize, degrees: &[usize], grid_types: &[GridType]) -> Self {
//         let lis = vec![InterpLagrange; ndim];
//         for i in 0..ndim {
//             lis[i] = InterpLagrange::new(degrees[i], grid_types[i]);
//         }
//         lis
//     }
// }

fn uniform_grid(N: usize) -> Vec<f64> {
    let mut res = vec![0.0; N + 1];
    let count = N + 1;
    let start = -1.0;
    let stop = 1.0;
    if count == 0 {
        return res;
    }
    res[0] = start;
    if count == 1 {
        return res;
    }
    res[count - 1] = stop;
    if count == 2 {
        return res;
    }
    let den = (count - 1) as f64;
    let step = (stop - start) / den;
    for i in 1..count {
        let p = i as f64;
        res[i] = start + p * step;
    }
    res
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
                GridType::Uniform => uniform_grid(N),
                GridType::ChebyshevGauss => chebyshev_gauss_points(N),
                GridType::ChebyshevGaussLobatto => chebyshev_lobatto_points(N),
            },
            U: Vector::new(0),
            Bary: true,
            UseEta: true,
            Eta: vec![0.0; N + 1],
            Lam: vec![0.0; N + 1],
            D1: Matrix::new(0, 0),
            D2: Matrix::new(0, 0),
        };

        // compute η
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

        // compute λk
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
    /// * `i` -- index of X[i] point
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

    /// Evaluates the function f(x[i]) over all nodes
    ///
    /// The function is `(x: f64, i: usize) -> f64`
    ///
    /// # Output
    ///
    /// The results are stores in the `U` variable
    pub fn CalcU<F>(&mut self, mut f: F)
    where
        F: FnMut(f64, usize) -> f64,
    {
        if self.U.dim() != self.np1 {
            self.U = Vector::new(self.np1);
        }
        for i in 0..self.np1 {
            self.U[i] = f(self.X[i], i);
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
    /// NOTE: U[i] = f(x[i]) must be calculated with o.CalcU or set first
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
    ///  d I{f}(x)  │         N
    /// ——————————— │      =  Σ   D1_kj ⋅ f(x_j)
    ///      dx     │x=x_k   j=0
    /// ```
    ///
    /// See: Berrut and Trefethen (2004)
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
    ///         d²ℓ_l  |
    /// D2_jl = —————— |
    ///          dx²   |x=x_j
    /// ```
    ///
    /// TODO:
    /// 1. Impl flag "already_calculated"
    /// 2. Handle the below
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
    /// * `dfdxAna` -- function `(x: f64, i: usize) -> f64`
    ///
    /// NOTE: U and D1 matrix must be computed previously
    pub fn CalcErrorD1<F>(&self, mut dfdxAna: F) -> f64
    where
        F: FnMut(f64, usize) -> f64,
    {
        // derivative of interpolation @ x_i
        let mut v = Vector::new(self.np1);
        mat_vec_mul(&mut v, 1.0, &self.D1, &self.U).unwrap();

        // compute error
        let mut maxDiff = 0.0;
        for i in 0..self.np1 {
            let vana = dfdxAna(self.X[i], i);
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
    /// * `dfdxAna` -- function `(x: f64, i: usize) -> f64`
    ///
    /// NOTE: U and D2 matrix must be computed previously
    pub fn CalcErrorD2<F>(&self, mut d2fdx2Ana: F) -> f64
    where
        F: FnMut(f64, usize) -> f64,
    {
        // derivative of interpolation @ x_i
        let mut v = Vector::new(self.np1);
        mat_vec_mul(&mut v, 1.0, &self.D2, &self.U).unwrap();

        // compute error
        let mut maxDiff = 0.0;
        for i in 0..self.np1 {
            let vana = d2fdx2Ana(self.X[i], i);
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{GridType, InterpLagrange};

    #[test]
    fn new_works() {
        let lag = InterpLagrange::new(2, GridType::Uniform).unwrap();
        assert_eq!(lag.N, 2);
        assert_eq!(lag.np1, 3);
        assert_eq!(lag.X, &[-1.0, 0.0, 1.0]);
        assert_eq!(lag.U.dim(), 0);
        assert_eq!(lag.Bary, true);
        assert_eq!(lag.UseEta, true);
        assert_eq!(lag.Eta.len(), 3);
        assert_eq!(lag.Lam.len(), 3);
        assert_eq!(lag.D1.dims(), (0, 0));
        assert_eq!(lag.D2.dims(), (0, 0));
    }
}
