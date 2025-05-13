use crate::{AsArray2D, StrError, Vector};

/// Implements the B-spline algorithms from the NURBS book
///
/// The i-th B-spline basis function Nᵢ,ₚ(u) is defined recursively using the De Boor's formula:
///
/// ```text
///           ⎧ 1  if uᵢ ≤ u < uᵢ₊₁
/// Nᵢ,₀(u) = ⎨
///           ⎩ 0  otherwise
///
///            u - uᵢ                uᵢ₊ₚ₊₁ - u
/// Nᵢ,ₚ(u) = ————————— Nᵢ,ₚ₋₁(u) + ————————————— Nᵢ₊₁,ₚ₋₁(u)
///           uᵢ₊ₚ - uᵢ             uᵢ₊ₚ₊₁ - uᵢ₊₁
/// ```
///
/// where `uᵢ` is the i-th knot value and `p` is the degree of the B-spline.
///
/// Notes:
///
/// * the knots vector U={u₀, ..., uₘ} is a non-decreasing sequence of real numbers
/// * uᵢ ≤ uᵢ₊₁ with i=0, ..., m-1
/// * the number of knots must be ≥ 2 * (p+1)
/// * m+1 is the number of knots
/// * p is the degree of the B-spline (p+1 is the order)
/// * n+1 is the number of basis functions where n = m - p - 1
/// * the interval closed-open interval [uᵢ, uᵢ₊₁) is called the i-th knot span
///
/// # Reference
///
/// 1. Piegl, L., & Tiller, W. (1997). The NURBS book (2nd ed.). Springer.
pub struct Bspline {
    /// Degree p
    p: usize,

    /// Knots vector U
    ///
    /// (m+1) knot values
    uu: Vec<f64>,

    /// Minimum knot value
    u_min: f64,

    /// Maximum knot value
    u_max: f64,

    /// Control points P
    ///
    /// (n+1) control points for n+1 basis functions
    /// Each inner Vec represents one control point's coordinates
    pp: Vec<Vec<f64>>,

    /// Non-vanishing basis functions N_{i,p} and knot differences
    ///
    /// * The diagonal and upper triangle contain basis functions Nᵢ,ₚ  
    /// * The lower triangle contains knot differences (uᵢ₊₁ - uᵢ)
    ///
    /// (p+1)×(p+1) nested vectors
    ndu: Vec<Vec<f64>>,

    /// All derivatives of the basis functions
    ///
    /// Vector of vectors where ders[k][j] is the kth derivative of Nᵢ₋ₚ₊ⱼ,ₚ
    ///
    /// (p+1)×(p+1)
    ders: Vec<Vec<f64>>,

    /// Auxiliary "left" array for Algorithm A2.3
    ///
    /// (p+1)
    left: Vec<f64>,

    /// Auxiliary "right" array for Algorithm A2.3
    ///
    /// (p+1)
    right: Vec<f64>,

    /// Auxiliary "a" matrix for Algorithm A2.3
    ///
    /// 2×(p+1) nested vectors
    a: Vec<Vec<f64>>,

    /// Current knot span index determined in `calc_basis` and `calc_basis_and_derivs`
    ///
    /// This variable is needed for subsequent calls to `get_basis` and `get_basis_deriv`.
    span: usize,

    /// Derivatives of the B-spline curve
    ///
    /// (p+1)×ndim nested vectors
    cc_ders: Vec<Vec<f64>>,

    /// Control points of all derivative curves
    ///
    /// (p+1)×npp×dim  where nb = number of control points = number of basis functions
    pp_ders: Vec<Vec<Vec<f64>>>,
}

impl Bspline {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `p` - Degree of the B-spline; it must be ≥ 1
    /// * `uu` - Knot vector U; it must have at least 2 * (p + 1) elements
    pub fn new(p: usize, uu: &[f64]) -> Result<Self, StrError> {
        // check input
        if p < 1 {
            return Err("the degree must be ≥ 1");
        }
        if uu.len() < 2 * (p + 1) {
            return Err("the number of knots must be ≥ 2 (p + 1)");
        }

        // find min and max knot values
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        for &u in uu {
            if u < u_min {
                u_min = u;
            }
            if u > u_max {
                u_max = u;
            }
        }

        // return instance
        let pp1 = p + 1;
        Ok(Bspline {
            p,
            uu: uu.to_vec(),
            u_min,
            u_max,
            pp: Vec::new(),
            ndu: vec![vec![0.0; pp1]; pp1],
            ders: vec![vec![0.0; pp1]; pp1],
            left: vec![0.0; pp1],
            right: vec![0.0; pp1],
            a: vec![vec![0.0; pp1]; 2],
            span: 0,
            cc_ders: Vec::new(),
            pp_ders: Vec::new(),
        })
    }

    /// Returns the number of basis functions (= number of control points)
    ///
    /// Note: the number of basis functions equals the number of control points
    pub fn num_basis(&self) -> usize {
        // m = len(U) - 1 (number of knots - 1)
        // n = m - p - 1  (number of basis functions - 1)
        // nb = n + 1 = m - p = len(U) - 1 - p
        self.uu.len() - 1 - self.p
    }

    /// Sets the control points
    ///
    /// Note: the number of control points must equal the number of basis functions.
    /// Use [Bspline::num_basis()] to check the number of basis functions.
    pub fn set_control_points<'a, T>(&mut self, pp: &'a T) -> Result<(), StrError>
    where
        T: AsArray2D<'a, f64>,
    {
        let nb = self.num_basis();
        let (npp, ndim) = pp.size();
        if npp != nb {
            return Err("the number of control points must equal the number of basis functions");
        }
        if ndim == 0 {
            return Err("the number of dimensions must be ≥ 1");
        }
        self.pp = Vec::with_capacity(nb);
        for i in 0..nb {
            let mut point = Vec::with_capacity(ndim);
            for j in 0..ndim {
                point.push(pp.at(i, j));
            }
            self.pp.push(point);
        }
        Ok(())
    }

    /// Calculates all non-zero basis functions at parameter u
    ///
    /// **Note**: use [Bspline::get_basis()] to get the value of a specific basis function
    pub fn calc_basis(&mut self, u: f64) -> Result<(), StrError> {
        if u < self.u_min || u > self.u_max {
            return Err("u is out of range");
        }
        self.span = self.find_span(u);
        self.basis_funs(u, self.span);
        Ok(())
    }

    /// Computes all non-zero basis functions and first order derivatives up to the order p
    ///
    /// Note: use [Bspline::get_basis()] and [Bspline::get_basis_deriv()] to get specific values
    pub fn calc_basis_and_derivs(&mut self, u: f64, upto: usize) -> Result<(), StrError> {
        if u < self.u_min || u > self.u_max {
            return Err("u is out of range");
        }
        if upto > self.p {
            return Err("upto must be in [0, p]");
        }
        self.span = self.find_span(u);
        self.ders_basis_funs(u, self.span, upto);
        Ok(())
    }

    /// Returns the value of the i-th basis function at the last calculated parameter u
    ///
    /// **Note**: this function must be called after [Bspline::calc_basis()]
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of range (the range is [0, num_basis])
    pub fn get_basis(&self, i: usize) -> f64 {
        if i + self.p >= self.span && i <= self.span {
            let j = i + self.p - self.span;
            self.ndu[j][self.p]
        } else {
            0.0
        }
    }

    /// Returns the derivative of a basis function w.r.t the parameter (dNi/du) at the last calculated parameter u
    ///
    /// **Note**: this function must be called after [Bspline::calc_basis_and_derivs()]
    ///
    /// # Panics
    ///
    /// This function panics if `i` is out of range (the range is [0, p])
    pub fn get_basis_deriv(&self, i: usize, degree: usize) -> f64 {
        if i + self.p >= self.span && i <= self.span {
            let j = i + self.p - self.span;
            self.ders[degree][j]
        } else {
            0.0
        }
    }

    /// Computes one particular basis function recursively (not efficient)
    ///
    /// Note: this function is not very efficient and is useful for testing/debugging purposes only.
    ///
    /// # Panics
    ///
    /// This function panics if `u` is outside the knot range
    pub fn recursive_basis(&self, u: f64, i: usize) -> f64 {
        if u < self.u_min || u > self.u_max {
            panic!("u is out of range");
        }
        self.recursive_nn(u, i, self.p)
    }

    /// Calculates and returns the x-y-z coordinates of a point on the B-spline
    ///
    /// # Arguments
    ///
    /// * `cc` - `C` vector to store the coordinates (must have the same dimension as the control points)
    /// * `u` - Parameter value
    /// * `recursive` - use recursive algorithm (not efficient) instead of Piegl & Tiller algorithm
    ///
    /// # Panics
    ///
    /// Panics if control points are not set
    pub fn calc_point(&mut self, cc: &mut Vector, u: f64, recursive: bool) -> Result<(), StrError> {
        let nb = self.num_basis();
        if self.pp.len() != nb {
            return Err("control points must be set before calling calc_point");
        }
        let ndim = self.pp[0].len(); // nb is always > 0
        if cc.dim() != ndim {
            return Err("cc must have the same dimension as control points");
        }
        if recursive {
            // recursive
            cc.fill(0.0);
            for idx in 0..nb {
                let basis = self.recursive_basis(u, idx);
                for j in 0..ndim {
                    cc[j] += basis * self.pp[idx][j];
                }
            }
        } else {
            // Piegl & Tiller: A3.1 p82
            let span = self.find_span(u);
            self.basis_funs(u, span);
            cc.fill(0.0);
            for i in 0..=self.p {
                let idx = span - self.p + i;
                for j in 0..ndim {
                    cc[j] += self.ndu[i][self.p] * self.pp[idx][j];
                }
            }
        }
        Ok(())
    }

    /// Returns the indices of nonzero spans (aka, elements in isogeometric analysis/IGA)
    ///
    /// `tolerance` is a small value such as `1e-14` corresponding to the minimum length of a span  
    pub fn get_elements(&self, tolerance: f64) -> Vec<[usize; 2]> {
        let mut spans = Vec::with_capacity(2 * self.uu.len());
        for i in 0..self.uu.len() - 1 {
            let l = self.uu[i + 1] - self.uu[i];
            if l.abs() > tolerance {
                spans.push([i, i + 1]);
            }
        }
        spans
    }

    /// Calculates the derivatives of the B-spline curve at parameter u
    ///
    /// ```text
    ///             →
    ///            dC⁽ᵏ⁾
    /// calculates ————— @ u
    ///              du
    /// ```
    ///
    /// # Arguments
    ///
    /// * `u` - Parameter value
    /// * `upto` - Order of the derivative (0 ≤ k ≤ p)
    /// * `use_control` - Use the method that considers the control points of all derivative curves
    ///
    /// # Output
    ///
    /// Use the [Bspline::get_curve_deriv()] method to get the results
    pub fn calc_curve_derivs(&mut self, u: f64, upto: usize, use_control: bool) {
        let upto = usize::min(upto, self.p);
        if use_control {
            self.curve_derivs_alg2(u, upto);
        } else {
            self.curve_derivs_alg1(u, upto);
        }
    }

    /// Returns the derivatives of the B-spline curve at parameter u calculated in `calc_derivs`
    ///
    /// Note: this function must be called after [Bspline::calc_curve_derivs()]
    ///
    /// # Arguments
    ///
    /// * `cck` - `dC⁽ᵏ⁾/du` vector with the derivatives (must have the same dimension as the control points)
    pub fn get_curve_deriv(&mut self, cck: &mut Vector, k: usize) {
        if k > self.p {
            cck.fill(0.0);
        } else {
            for l in 0..cck.dim() {
                cck[l] = self.cc_ders[k][l];
            }
        }
    }

    // private methods --------------------------------------------------------------------------------

    /// Computes basis functions using Cox-DeBoors recursive formula
    fn recursive_nn(&self, u: f64, i: usize, p: usize) -> f64 {
        // remove noise, e.g., 1.000000000000002
        let u = if f64::abs(u - self.u_max) < 1e-14 {
            self.u_max - 1e-14
        } else {
            u
        };

        // exit point
        if p == 0 {
            if u >= self.uu[i] && u < self.uu[i + 1] {
                return 1.0;
            } else {
                return 0.0;
            };
        }

        let d1 = self.uu[i + p] - self.uu[i];
        let d2 = self.uu[i + p + 1] - self.uu[i + 1];

        let (n1, d1) = if d1.abs() < 1e-14 {
            (0.0, 1.0)
        } else {
            (self.recursive_nn(u, i, p - 1), d1)
        };

        let (n2, d2) = if d2.abs() < 1e-14 {
            (0.0, 1.0)
        } else {
            (self.recursive_nn(u, i + 1, p - 1), d2)
        };

        (u - self.uu[i]) * n1 / d1 + (self.uu[i + p + 1] - u) * n2 / d2
    }

    /// Finds the knot span index for a given parameter value.
    ///
    /// Implements Algorithm A2.1 on page 68 of Reference 1
    fn find_span(&self, u: f64) -> usize {
        // constants
        let m = self.uu.len() - 1; // number of knots - 1
        let n = m - self.p - 1; // number of basis functions - 1

        // special case: if u is at or beyond the last knot
        if u >= self.uu[n + 1] {
            return n;
        }

        // special case: if u is at or before the first knot
        if u <= self.uu[self.p] {
            return self.p;
        }

        // binary search for the correct span
        let mut low = self.p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while u < self.uu[mid] || u >= self.uu[mid + 1] {
            if u < self.uu[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    /// Computes all non-vanishing basis functions and knot differences.
    ///
    /// Implements part of Algorithm A2.3 on page 72 of Reference 1
    ///
    /// Saves the results in the `ndu` (p+1)×(p+1) matrix
    fn basis_funs(&mut self, u: f64, span: usize) {
        // first basis function
        self.ndu[0][0] = 1.0;

        // loop over the degree p
        let mut saved;
        for j in 1..=self.p {
            self.left[j] = u - self.uu[span + 1 - j];
            self.right[j] = self.uu[span + j] - u;
            saved = 0.0;
            for r in 0..j {
                // lower triangle (knot differences)
                self.ndu[j][r] = self.right[r + 1] + self.left[j - r];

                // upper triangle and diagonal (basis functions)
                let temp = self.ndu[r][j - 1] / self.ndu[j][r];
                self.ndu[r][j] = saved + self.right[r + 1] * temp;
                saved = self.left[j - r] * temp;
            }
            self.ndu[j][j] = saved;
        }
    }

    /// Computes derivatives of B-spline basis functions up to a specified order.
    ///
    /// Implements part of Algorithm A2.3 on page 72 of Reference 1
    ///
    /// Saves the results in the `ndu` and `ders` (p+1)×(p+1) matrices
    fn ders_basis_funs(&mut self, u: f64, span: usize, upto: usize) {
        // compute basis functions and knot differences
        self.basis_funs(u, span);

        // load basis functions (0th derivatives)
        for j in 0..=self.p {
            self.ders[0][j] = self.ndu[j][self.p];
        }

        // auxiliary variables
        let mut s1: usize;
        let mut s2: usize;
        let mut pk: usize; // pk = p - k;   rk = r - k; (but cannot be directly computed in Rust because the result may be negative)
        let mut rkj: usize; // = rk+j = (r+j)-k; need to add (r+j) before subtracting k
        let mut d: f64;

        // compute the derivatives (Eq 2.9)
        for r in 0..=self.p {
            // alternate rows in array a
            s1 = 0;
            s2 = 1;
            self.a[0][0] = 1.0;

            // loop to compute k-th derivative
            for k in 1..=upto {
                d = 0.0;
                pk = self.p - k;

                // first step
                if r >= k {
                    self.a[s2][0] = self.a[s1][0] / self.ndu[pk + 1][r - k];
                    d = self.a[s2][0] * self.ndu[r - k][pk];
                }

                // second step (need to avoid subtraction yielding negative values)
                let j1 = if r >= k - 1 { 1 } else { k - r };
                let j2 = if r <= pk + 1 { k - 1 } else { self.p - r };
                for j in j1..=j2 {
                    rkj = (r + j) - k; // must add (r+j) before subtracting k
                    self.a[s2][j] = (self.a[s1][j] - self.a[s1][j - 1]) / self.ndu[pk + 1][rkj];
                    d += self.a[s2][j] * self.ndu[rkj][pk];
                }

                // third step
                if r <= pk {
                    self.a[s2][k] = -self.a[s1][k - 1] / self.ndu[pk + 1][r];
                    d += self.a[s2][k] * self.ndu[r][pk];
                }
                self.ders[k][r] = d;

                // switch rows
                std::mem::swap(&mut s1, &mut s2);
            }
        }

        // multiply through by the correct factors
        d = self.p as f64;
        for k in 1..=upto {
            for j in 0..=self.p {
                self.ders[k][j] *= d;
            }
            d *= (self.p - k) as f64;
        }
    }

    /// Computes all derivatives up to and including the upto-th at fixed u value
    ///
    /// Implements Algorithm A3.2 of Ref 1 (page 93)
    ///
    /// Saves the results in the `cc_ders` array, where `cc_ders[k]` is the k-th derivative with `0 ≤ k ≤ upto`.
    ///
    /// `upto` must be in [0, p]
    fn curve_derivs_alg1(&mut self, u: f64, upto: usize) {
        assert_eq!(self.pp.len(), self.num_basis());
        assert!(self.pp[0].len() > 0);
        assert!(upto <= self.p);
        let ndim = self.pp[0].len();
        if self.cc_ders.len() != self.p + 1 {
            self.cc_ders = vec![vec![0.0; ndim]; self.p + 1];
        }
        let span = self.find_span(u);
        self.ders_basis_funs(u, span, upto);
        for k in 0..=upto {
            self.cc_ders[k].fill(0.0);
            for j in 0..=self.p {
                let idx = span - self.p + j;
                for l in 0..ndim {
                    self.cc_ders[k][l] += self.ders[k][j] * self.pp[idx][l];
                }
            }
        }
    }

    /// Computes the control points of all derivative curves up to and including the upto-th derivative
    ///
    /// Implements Algorithm A3.3 of Ref 1 (page 98)
    ///
    /// Saves the results in the `pp_ders` matrix, where `pp_ders(k,i)` corresponds the control point the
    /// k-th derivative curve, where `0 ≤ k ≤ upto` and `r1 ≤ i ≤ r2-k`.
    ///
    /// If `r1=0` and `r2=n`, all control points are computed.
    ///
    /// `upto` must be in [0, p]
    fn curve_deriv_cpts(&mut self, upto: usize, r1: usize, r2: usize) {
        assert_eq!(self.pp.len(), self.num_basis());
        assert!(self.pp[0].len() > 0);
        assert!(upto <= self.p);
        assert!(r1 <= r2);
        let ndim = self.pp[0].len();
        let npp = self.pp.len(); // == nb
        if self.pp_ders.len() != self.p + 1 {
            self.pp_ders = vec![vec![vec![0.0; ndim]; npp]; self.p + 1];
        }
        let r = r2 - r1;
        for i in 0..=r {
            for l in 0..ndim {
                self.pp_ders[0][i][l] = self.pp[r1 + i][l];
            }
        }
        let mut tmp: f64;
        let mut num: f64;
        let mut den: f64;
        for k in 1..=upto {
            tmp = (self.p - k + 1) as f64;
            for i in 0..=(r - k) {
                for l in 0..ndim {
                    num = self.pp_ders[k - 1][i + 1][l] - self.pp_ders[k - 1][i][l];
                    den = self.uu[r1 + i + self.p + 1] - self.uu[r1 + i + k];
                    self.pp_ders[k][i][l] = tmp * num / den;
                }
            }
        }
    }

    /// Computes all derivatives up to and including the upto-th at fixed u value
    ///
    /// Implements Algorithm A3.4 of Ref 1 (page 99)
    ///
    /// Saves the results in the `cc_ders` array, where `cc_ders[k]` is the k-th derivative with `0 ≤ k ≤ upto`.
    ///
    /// `upto` must be in [0, p]
    fn curve_derivs_alg2(&mut self, u: f64, upto: usize) {
        assert_eq!(self.pp.len(), self.num_basis());
        assert!(self.pp[0].len() > 0);
        assert!(upto <= self.p);
        let ndim = self.pp[0].len();
        if self.cc_ders.len() != self.p + 1 {
            self.cc_ders = vec![vec![0.0; ndim]; self.p + 1];
        }
        let span = self.find_span(u);
        self.ders_basis_funs(u, span, upto);
        self.curve_deriv_cpts(upto, span - self.p, span);
        for k in 0..=upto {
            self.cc_ders[k].fill(0.0);
            for j in 0..=(self.p - k) {
                for l in 0..ndim {
                    self.cc_ders[k][l] += self.ndu[j][self.p - k] * self.pp_ders[k][j][l];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Bspline;
    use crate::{approx_eq, deriv1_central5, mat_approx_eq, vec_approx_eq};
    use crate::{deriv2_forward8, AsArray2D, Matrix, Vector};
    use plotpy::{linspace, Canvas, Curve, Plot, Text};

    const SAVE_FIGURE: bool = false;

    /// Returns a sample pair of degree and knot vector for testing
    fn get_sample(sample: usize) -> (usize, Vec<f64>) {
        if sample == 1 {
            //         p         m-p-1       m-p     m
            //         p           n       K-1-p     K-1
            //   0  1  2           3           4  5  6
            //   |  |  |-----------|-----------|  |  |
            // 0.0 0.0 0.0        0.5        1.0 1.0 1.0
            (2, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        } else {
            //         p                          m-p-1    m-p     m
            //         p                              n  K-1-p     K-1
            //   0  1  2      3      4      5      6  7      8  9  10
            //   |  |  |------|------|------|------|  |------|  |  |
            // 0.0 0.0 0.0   1.0    2.0    3.0   4.0  4.0  5.0 5.0 5.0
            (2, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0])
        }
    }

    /// Implements the analytical solution from page 54-55 of Ref 1
    fn sample2_basis(i: usize, u: f64) -> f64 {
        //         p                          m-p-1    m-p     m
        //         p                              n  K-1-p     K-1
        //   0  1  2      3      4      5      6  7      8  9  10
        //   |  |  |------|------|------|------|  |------|  |  |
        // 0.0 0.0 0.0   1.0    2.0    3.0   4.0  4.0  5.0 5.0 5.0
        if i == 0 {
            if u >= 0.0 && u < 1.0 {
                (1.0 - u) * (1.0 - u)
            } else {
                0.0
            }
        } else if i == 1 {
            if u >= 0.0 && u < 1.0 {
                2.0 * u - 1.5 * u * u
            } else if u >= 1.0 && u < 2.0 {
                0.5 * (2.0 - u) * (2.0 - u)
            } else {
                0.0
            }
        } else if i == 2 {
            if u >= 0.0 && u < 1.0 {
                0.5 * u * u
            } else if u >= 1.0 && u < 2.0 {
                -1.5 + 3.0 * u - u * u
            } else if u >= 2.0 && u < 3.0 {
                0.5 * (3.0 - u) * (3.0 - u)
            } else {
                0.0
            }
        } else if i == 3 {
            if u >= 1.0 && u < 2.0 {
                0.5 * (u - 1.0) * (u - 1.0)
            } else if u >= 2.0 && u < 3.0 {
                -5.5 + 5.0 * u - u * u
            } else if u >= 3.0 && u < 4.0 {
                0.5 * (4.0 - u) * (4.0 - u)
            } else {
                0.0
            }
        } else if i == 4 {
            if u >= 2.0 && u < 3.0 {
                0.5 * (u - 2.0) * (u - 2.0)
            } else if u >= 3.0 && u < 4.0 {
                -16.0 + 10.0 * u - 1.5 * u * u
            } else {
                0.0
            }
        } else if i == 5 {
            if u >= 3.0 && u < 4.0 {
                (u - 3.0) * (u - 3.0)
            } else if u >= 4.0 && u < 5.0 {
                (5.0 - u) * (5.0 - u)
            } else {
                0.0
            }
        } else if i == 6 {
            if u >= 4.0 && u < 5.0 {
                2.0 * (u - 4.0) * (5.0 - u)
            } else {
                0.0
            }
        } else if i == 7 {
            if u >= 4.0 && u <= 5.0 {
                // must consider u == 5.0
                (u - 4.0) * (u - 4.0)
            } else {
                0.0
            }
        } else {
            panic!("i must be in [0, 7]");
        }
    }

    fn sample2_ders1(i: usize, u: f64) -> f64 {
        if i == 5 && (4.0 <= u && u <= 5.0) {
            2.0 * (-5.0 + u)
        } else if i == 6 && (4.0 <= u && u <= 5.0) {
            -2.0 * (-5.0 + u) - 2.0 * (-4.0 + u)
        } else if i == 3 && (3.0 <= u && u < 4.0) {
            -4.0 + u
        } else if i > 6 && (4.0 <= u && u <= 5.0) {
            2.0 * (-4.0 + u)
        } else if i == 2 && (2.0 <= u && u < 3.0) {
            -3.0 + u
        } else if i == 5 && (3.0 <= u && u < 4.0) {
            2.0 * (-3.0 + u)
        } else if (i == 1 && (1.0 <= u && u < 2.0)) || (i == 4 && (2.0 <= u && u < 3.0)) {
            -2.0 + u
        } else if i == 3 && (1.0 <= u && u < 2.0) {
            -1.0 + u
        } else if i == 0 && (0.0 <= u && u < 1.0) {
            2.0 * (-1.0 + u)
        } else if i == 2 && (0.0 <= u && u < 1.0) {
            u
        } else if i == 1 && (0.0 <= u && u < 1.0) {
            (4.0 - 6.0 * u) / 2.0
        } else if i == 4 && (3.0 <= u && u < 4.0) {
            (20.0 - 6.0 * u) / 2.0
        } else if i == 2 && (1.0 <= u && u < 2.0) {
            (6.0 - 4.0 * u) / 2.0
        } else if i == 3 && (2.0 <= u && u < 3.0) {
            (10.0 - 4.0 * u) / 2.0
        } else {
            0.0
        }
    }

    fn sample2_ders2(i: usize, u: f64) -> f64 {
        if i == 5 && (4.0 <= u && u <= 5.0) {
            2.0
        } else if i == 6 && (4.0 <= u && u <= 5.0) {
            -4.0
        } else if i == 3 && (3.0 <= u && u < 4.0) {
            1.0
        } else if i > 6 && (4.0 <= u && u <= 5.0) {
            2.0
        } else if i == 2 && (2.0 <= u && u < 3.0) {
            1.0
        } else if i == 5 && (3.0 <= u && u < 4.0) {
            2.0
        } else if (i == 1 && (1.0 <= u && u < 2.0))
            || (i == 4 && (2.0 <= u && u < 3.0))
            || (i == 3 && (1.0 <= u && u < 2.0))
        {
            1.0
        } else if i == 0 && (0.0 <= u && u < 1.0) {
            2.0
        } else if i == 2 && (0.0 <= u && u < 1.0) {
            1.0
        } else if (i == 1 && (0.0 <= u && u < 1.0)) || (i == 4 && (3.0 <= u && u < 4.0)) {
            -3.0
        } else if (i == 2 && (1.0 <= u && u < 2.0)) || (i == 3 && (2.0 <= u && u < 3.0)) {
            -2.0
        } else {
            0.0
        }
    }

    /// Draws the B-spline curve and control points
    fn draw_curve<'a, T>(b: &mut Bspline, pp: &'a T, tangents_at_uu: Option<&[f64]>, tg_scale: f64) -> Plot
    where
        T: AsArray2D<'a, f64>,
    {
        // draw control points
        let mut curve_control = Curve::new();
        let mut text_control = Text::new();
        text_control
            .set_fontsize(12.0)
            .set_align_horizontal("center")
            .set_align_vertical("center")
            .set_bbox(true)
            .set_bbox_facecolor("white")
            .set_bbox_style("circle,pad=0.2");
        curve_control.points_begin();
        let (npp, ndim) = pp.size();
        for i in 0..npp {
            let (x, y) = (pp.at(i, 0), pp.at(i, 1));
            curve_control.points_add(x, y);
            text_control.draw(x, y, &format!("${{\\bf P}}_{{{}}}$", i));
        }
        curve_control.points_end();

        // draw B-spline curve
        let mut curve = Curve::new();
        let n_station = 201;
        let knots = linspace(0.0, 1.0, n_station);
        let mut xx = vec![0.0; knots.len()];
        let mut yy = vec![0.0; knots.len()];
        let recursive = false;
        let mut cc = Vector::new(ndim);
        for i in 0..n_station {
            b.calc_point(&mut cc, knots[i], recursive).unwrap();
            xx[i] = cc[0];
            yy[i] = cc[1];
        }
        curve.draw(&xx, &yy);

        // draw spans
        let mut curve_spans = Curve::new();
        curve_spans
            .set_line_style("None")
            .set_marker_style("o")
            .set_marker_color("black");
        let elements = b.get_elements(1e-14);
        for e in &elements {
            b.calc_point(&mut cc, b.uu[e[0]], false).unwrap();
            curve_spans.draw(&[cc[0]], &[cc[1]]);
        }

        // tangent vectors
        let mut tangent_vectors = Canvas::new();
        if let Some(uu_tg) = tangents_at_uu {
            tangent_vectors
                .set_edge_color("gray")
                .set_face_color("gray")
                .set_arrow_scale(15.0);
            for &u in uu_tg {
                b.calc_point(&mut cc, u, false).unwrap();
                b.calc_curve_derivs(u, 1, false);
                let (dx, dy) = (b.cc_ders[1][0], b.cc_ders[1][1]);
                let norm = f64::sqrt(dx * dx + dy * dy);
                if norm > 1e-14 {
                    let (xi, yi) = (cc[0], cc[1]);
                    let (xf, yf) = (xi + tg_scale * dx / norm, yi + tg_scale * dy / norm);
                    tangent_vectors.draw_arrow(xi, yi, xf, yf);
                }
            }
        }

        // generate figure
        let mut plot = Plot::new();
        plot.add(&curve_control)
            .add(&tangent_vectors)
            .add(&curve)
            .add(&curve_spans)
            .add(&text_control)
            .set_hide_axes(true)
            .set_equal_axes(true)
            .set_figure_size_points(600.0, 600.0);
        plot
    }

    #[test]
    #[should_panic(expected = "i must be in [0, 7]")]
    fn test_aux_function_panics() {
        sample2_basis(8, 0.0);
    }

    #[test]
    fn test_new_captures_errors() {
        assert_eq!(Bspline::new(0, &[0.0, 1.0]).err(), Some("the degree must be ≥ 1"));
        assert_eq!(
            Bspline::new(2, &[0.0, 0.0, 0.0, 1.0, 1.0]).err(),
            Some("the number of knots must be ≥ 2 (p + 1)")
        );
    }

    #[test]
    fn test_essential_1() {
        // get sample # 1
        let (p, uu) = get_sample(1);

        // constants
        let kk = 7; // K: number of knots
        let m = kk - 1;
        let n = m - p - 1;
        let nn = n + 1; // N: number of basis
        assert_eq!(nn, 4);

        // allocate B-spline
        let b = Bspline::new(p, &uu).unwrap();

        // check essential data
        assert_eq!(b.num_basis(), nn);
        assert_eq!(b.p, p);
        assert_eq!(b.uu.len(), kk);
        assert_eq!(b.u_min, 0.0);
        assert_eq!(b.u_max, 1.0);
        assert_eq!(b.pp.len(), 0);
        assert_eq!(b.ndu.len(), p + 1);
        assert_eq!(b.ders.len(), p + 1);
        assert_eq!(b.left.len(), p + 1);
        assert_eq!(b.right.len(), p + 1);
        assert_eq!(b.a.len(), 2);
        assert_eq!(b.span, 0);

        // check find_span
        let eps = 1e-6;
        let knots = &[-eps, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.0 + eps];
        let spans = &[p, p, p, p, n, n, n, n];
        for i in 0..knots.len() {
            assert_eq!(b.find_span(knots[i]), spans[i]);
        }
    }

    #[test]
    fn test_essential_2() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // constants
        let kk = 11; // K: number of knots
        let m = kk - 1;
        let n = m - p - 1;
        let nn = n + 1; // N: number of basis
        assert_eq!(nn, 8);

        // allocate B-spline
        let b = Bspline::new(p, &uu).unwrap();

        // check essential data
        assert_eq!(b.num_basis(), nn);
        assert_eq!(b.p, p);
        assert_eq!(b.uu.len(), kk);
        assert_eq!(b.u_min, 0.0);
        assert_eq!(b.u_max, 5.0);

        // check find_span
        let eps = 1e-6;
        let knots_and_spans = &[
            (-eps, 2),
            (0.0, 2),
            (0.5, 2),
            (1.0, 3),
            (1.5, 3),
            (2.0, 4),
            (2.5, 4),
            (3.0, 5),
            (3.5, 5),
            (4.0, 7),
            (4.5, 7),
            (5.0, 7),
            (5.0 + eps, 7),
        ];
        for &(u, span) in knots_and_spans {
            assert_eq!(b.find_span(u), span);
        }
    }

    #[test]
    fn test_basis_funs_and_get_basis() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // allocate B-spline
        let mut b = Bspline::new(p, &uu).unwrap();

        // calculate basis functions @ u = 5/2
        let u = 5.0 / 2.0;
        let span = b.find_span(u);
        assert_eq!(span, 4); // i = 4
        b.basis_funs(u, span);

        // reference 1, page 71, Table Ex2.4 (the last column is Nᵢ,ₚ = {N₂,₂, N₃,₂, N₄,₂})
        let ndu_correct = Matrix::from(&[
            [1.0, 1.0 / 2.0, 1.0 / 8.0], // N₄,₀,  N₃,₁,  N₂,₂
            [1.0, 1.0 / 2.0, 6.0 / 8.0], // u₅-u₄, N₄,₁,  N₃,₂
            [2.0, 2.0, 1.0 / 8.0],       // u₅-u₃, u₆-u₄, N₄,₂
        ]);
        mat_approx_eq(&ndu_correct, &b.ndu, 1e-15);
    }

    #[test]
    fn test_calc_basis_and_get_basis() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // allocate B-spline
        let mut b = Bspline::new(p, &uu).unwrap();

        // check calc_basis and get_basis
        let knots = &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        for &u in knots {
            b.calc_basis(u).unwrap();
            for i in 0..=7 {
                approx_eq(b.get_basis(i), sample2_basis(i, u), 1e-15);
            }
        }

        // check error catching
        assert_eq!(b.calc_basis(-1.0).err(), Some("u is out of range"));

        // plot basis functions
        if SAVE_FIGURE {
            let mut curve_num = Curve::new();
            let x = linspace(0.0, 5.0, 201);
            for i in 0..=7 {
                let y_num = x.iter().map(|&u| {
                    b.calc_basis(u).unwrap();
                    b.get_basis(i)
                });
                curve_num.draw(&x, &y_num.collect());
            }
            let mut text = Text::new();
            text.set_fontsize(12.0).set_align_horizontal("center");
            let i_x_y = [
                (0, 0.1, 1.04),
                (1, 0.7, 0.7),
                (2, 1.5, 0.78),
                (3, 2.5, 0.78),
                (4, 3.3, 0.7),
                (5, 4.0, 1.04),
                (6, 4.5, 0.54),
                (7, 5.0, 1.04),
            ];
            for (i, x, y) in i_x_y {
                text.draw(x, y, &format!("$N_{{{},{}}}$", i, p));
            }
            let mut plot = Plot::new();
            plot.add(&curve_num)
                .add(&text)
                .grid_and_labels("u", "Nᵢ,ₚ")
                .set_ymax(1.1)
                .set_figure_size_points(600.0, 350.0)
                .save("/tmp/russell_lab/test_bspline_calc_basis_and_get_basis.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_ders_basis_funs() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // allocate B-spline
        let mut b = Bspline::new(p, &uu).unwrap();

        // find span of u = 5/2
        let u = 5.0 / 2.0;
        let span = b.find_span(u);
        assert_eq!(span, 4); // i = 4

        //  j  i-p+j      Nᵢ₋ₚ₊ⱼ,ₚ⁽⁰⁾
        //  0  4-2+0 = 2  N₂,₂⁽⁰⁾=N₂,₂
        //  1  4-2+1 = 3  N₃,₂⁽⁰⁾=N₃,₂
        //  2  4-2+2 = 4  N₄,₂⁽⁰⁾=N₄,₂

        // check ders_basis_funs with upto = 1
        let upto = 1;
        b.ders_basis_funs(u, span, upto);
        let ders_correct = Matrix::from(&[
            [1.0 / 8.0, 6.0 / 8.0, 1.0 / 8.0], // N₂,₂,    N₃,₂,    N₄,₂ (last column of ndu)
            [-1.0 / 2.0, 0.0, 1.0 / 2.0],      // N₂,₂⁽¹⁾, N₃,₂⁽¹⁾, N₄,₂⁽¹⁾
            [0.0, 0.0, 0.0],                   // N₂,₂⁽²⁾, N₃,₂⁽²⁾, N₄,₂⁽²⁾
        ]);
        mat_approx_eq(&ders_correct, &b.ders, 1e-15);

        // check ders_basis_funs with upto = 2
        let upto = 2;
        b.ders_basis_funs(u, span, upto);
        let ders_correct = Matrix::from(&[
            [1.0 / 8.0, 6.0 / 8.0, 1.0 / 8.0], // N₂,₂,    N₃,₂,    N₄,₂ (last column of ndu)
            [-1.0 / 2.0, 0.0, 1.0 / 2.0],      // N₂,₂⁽¹⁾, N₃,₂⁽¹⁾, N₄,₂⁽¹⁾
            [1.0, -2.0, 1.0],                  // N₂,₂⁽²⁾, N₃,₂⁽²⁾, N₄,₂⁽²⁾
        ]);
        mat_approx_eq(&ders_correct, &b.ders, 1e-15);
    }

    #[test]
    fn test_calc_basis_and_derivs_and_get_basis_deriv() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // allocate B-spline
        let mut b = Bspline::new(p, &uu).unwrap();

        // check calc_basis_and_derivs, get_basis, and get_basis_deriv
        let knots = &[0.0, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.1, 4.5, 5.0];
        for &u in knots {
            b.calc_basis_and_derivs(u, 2).unwrap();
            for i in 0..=7 {
                approx_eq(b.get_basis(i), sample2_basis(i, u), 1e-14);
                approx_eq(b.get_basis_deriv(i, 0), sample2_basis(i, u), 1e-14);
                approx_eq(b.get_basis_deriv(i, 1), sample2_ders1(i, u), 1e-15);
                approx_eq(b.get_basis_deriv(i, 2), sample2_ders2(i, u), 1e-15);
            }
        }

        // check error catching
        assert_eq!(b.calc_basis_and_derivs(-1.0, 2).err(), Some("u is out of range"));
        assert_eq!(b.calc_basis_and_derivs(0.0, 3).err(), Some("upto must be in [0, p]"));
    }

    #[test]
    fn test_recursive_basis() {
        // get sample # 2
        let (p, uu) = get_sample(2);

        // allocate B-spline
        let b = Bspline::new(p, &uu).unwrap();

        // check recursive_basis
        let knots = &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        for &u in knots {
            for i in 0..=7 {
                approx_eq(b.recursive_basis(u, i), sample2_basis(i, u), 1e-13);
            }
        }
    }

    #[test]
    #[should_panic(expected = "u is out of range")]
    fn test_recursive_basis_panics_on_error() {
        let b = Bspline::new(2, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        b.recursive_basis(-1.0, 0);
    }

    #[test]
    fn test_get_elements() {
        // check get_elements
        //   0  1  2           3           4  5  6
        //   |  |  |-----------|-----------|  |  |
        // 0.0 0.0 0.0        0.5        1.0 1.0 1.0
        //               ^           ^
        let (p, uu) = get_sample(1);
        let b = Bspline::new(p, &uu).unwrap();
        let elements = b.get_elements(1e-14);
        assert_eq!(elements, &[[2, 3], [3, 4]]);

        // check get_elements
        //   0  1  2      3      4      5      6  7      8  9  10
        //   |  |  |------|------|------|------|  |------|  |  |
        // 0.0 0.0 0.0   1.0    2.0    3.0   4.0  4.0  5.0 5.0 5.0
        //             ^      ^      ^      ^         ^
        let (p, uu) = get_sample(2);
        let b = Bspline::new(p, &uu).unwrap();
        let elements = b.get_elements(1e-14);
        assert_eq!(elements, &[[2, 3], [3, 4], [4, 5], [5, 6], [7, 8]]);
    }

    #[test]
    fn test_set_control_points_and_calc_point_errors() {
        let mut b = Bspline::new(1, &[0.0, 0.0, 1.0, 1.0]).unwrap();
        let wrong_pp = &[[0.0, 0.0]];
        assert_eq!(
            b.set_control_points(wrong_pp).err(),
            Some("the number of control points must equal the number of basis functions")
        );

        let wrong_pp = &[[], []];
        assert_eq!(
            b.set_control_points(wrong_pp).err(),
            Some("the number of dimensions must be ≥ 1")
        );

        let mut cc = Vector::new(2);
        assert_eq!(
            b.calc_point(&mut cc, 0.0, false).err(),
            Some("control points must be set before calling calc_point")
        );

        let pp = &[[0.0, 0.0], [1.0, 0.0]];
        b.set_control_points(pp).unwrap();

        let mut wrong_cc = Vector::new(1);
        assert_eq!(
            b.calc_point(&mut wrong_cc, 0.0, false).err(),
            Some("cc must have the same dimension as control points")
        );
    }

    #[test]
    fn test_set_control_points_and_calc_point_1() {
        // allocate B-spline (Figure 3.2 on page 84 of Ref 1)
        let p = 3;
        let uu = &[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();
        assert_eq!(b.num_basis(), 7);

        // define control points
        let pp = &[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.1, 1.0],
            [2.5, -0.1],
            [1.7, -0.7],
            [1.0, -0.4],
        ];

        // check set_control_points
        b.set_control_points(pp).unwrap();
        assert_eq!(b.pp.len(), 7);

        // auxiliary variable `C` holding a point on the curve
        let mut cc = Vector::new(2);

        // check calc_point for the first control point
        b.calc_point(&mut cc, uu[0], false).unwrap();
        vec_approx_eq(&cc, &pp[0], 1e-15);

        // check calc_point for the last control point
        b.calc_point(&mut cc, uu[uu.len() - 1], false).unwrap();
        vec_approx_eq(&cc, &pp[pp.len() - 1], 1e-15);

        // compare recursive and non-recursive methods
        let mut cc_rec = Vector::new(2);
        let knots = &[0.0, 0.1, 0.3, 0.25, 0.5, 0.6, 0.75, 0.8, 1.0];
        for &u in knots {
            b.calc_point(&mut cc_rec, u, true).unwrap();
            b.calc_point(&mut cc, u, false).unwrap();
            vec_approx_eq(&cc_rec, &cc, 1e-13);
        }

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, None, 1.0);
            plot.save("/tmp/russell_lab/test_bspline_set_control_points_and_calc_point_1.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_set_control_points_and_calc_point_2() {
        // allocate B-spline (Figure 3.6 on page 87 of Ref 1)
        let p = 2;
        let uu = &[0.0, 0.0, 0.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();
        assert_eq!(b.num_basis(), 7);

        // define control points
        let slope = 2.0;
        let pp4 = [0.0, 0.0];
        let pp2 = [1.15, slope * 1.15];
        let pp3 = [(pp4[0] + pp2[0]) / 2.0, (pp4[1] + pp2[1]) / 2.0];
        let pp = &[
            [-0.56, 1.21], // 0
            [-0.95, 2.05], // 1
            pp2,           // 2
            pp3,           // 3
            pp4,           // 4
            [2.0, 0.0],    // 5
            [1.73, 1.0],   // 6
        ];

        // check set_control_points
        b.set_control_points(pp).unwrap();
        assert_eq!(b.pp.len(), 7);

        // auxiliary variable `C` holding a point on the curve
        let mut cc = Vector::new(2);

        // check calc_point for the first control point
        b.calc_point(&mut cc, uu[0], false).unwrap();
        vec_approx_eq(&cc, &pp[0], 1e-15);

        // check calc_point for the last control point
        b.calc_point(&mut cc, uu[uu.len() - 1], false).unwrap();
        vec_approx_eq(&cc, &pp[pp.len() - 1], 1e-15);

        // check points on strait parts of curve
        b.calc_point(&mut cc, 2.0 / 5.0, false).unwrap();
        approx_eq((cc[1] - pp4[1]) / (cc[0] - pp4[0]), slope, 1e-15);
        b.calc_point(&mut cc, 3.0 / 5.0, false).unwrap();
        approx_eq((cc[1] - pp4[1]) / (cc[0] - pp4[0]), slope, 1e-15);

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, None, 1.0);
            plot.save("/tmp/russell_lab/test_bspline_set_control_points_and_calc_point_2.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_curve_derivs_1() {
        let p = 3;
        let uu = &[0.0, 0.0, 0.0, 0.0, 1.0 / 4.0, 3.0 / 4.0, 1.0, 1.0, 1.0, 1.0];

        // digitized data
        let pp = &[
            [471.00368882, 557.99288541],
            [230.72621102, 1235.22245895],
            [689.6261915, 1582.64009801],
            [1214.65028031, 1587.37004824],
            [1678.18499528, 1248.70588194],
            [1455.87749291, 566.64758777],
        ];
        let a1 = &[[784.22511874, 1526.06441423], [1454.6544, 1679.09006588]];
        let a2 = &[[784.22511874, 1526.0644142], [687.90361436, 71.83909643]];
        let a3 = &[[784.22511874, 1526.06441423], [1907.1143811, 1537.23260824]];
        let mut cck_ref = Vector::new(2);
        let sf = 5.0 / 2.0; // scale factor to re-scale the tangent vector because the book scales down by 2/5

        // B-spline
        let mut b = Bspline::new(p, uu).unwrap();
        b.set_control_points(pp).unwrap();

        // calculate all curve derivatives
        let upto = 3;
        b.calc_curve_derivs(2.0 / 5.0, upto, false);

        // C⁽⁰⁾, i.e., coordinate @ u = 2/5
        let mut cc0 = Vector::new(2);
        b.get_curve_deriv(&mut cc0, 0);
        vec_approx_eq(&cc0, &a1[0], 1.32); // inaccurate data due to digitization

        // C⁽¹⁾
        let mut cc1 = Vector::new(2);
        b.get_curve_deriv(&mut cc1, 1);
        cck_ref[0] = sf * (a1[1][0] - a1[0][0]);
        cck_ref[1] = sf * (a1[1][1] - a1[0][1]);
        vec_approx_eq(&cc1, &cck_ref, 1.2); // inaccurate data due to digitization

        // C⁽²⁾
        let mut cc2 = Vector::new(2);
        b.get_curve_deriv(&mut cc2, 2);
        cck_ref[0] = sf * (a2[1][0] - a2[0][0]);
        cck_ref[1] = sf * (a2[1][1] - a2[0][1]);
        vec_approx_eq(&cc2, &cck_ref, 27.1); // inaccurate data due to digitization

        // C⁽³⁾
        let mut cc3 = Vector::new(2);
        b.get_curve_deriv(&mut cc3, 3);
        cck_ref[0] = sf * (a3[1][0] - a3[0][0]);
        cck_ref[1] = sf * (a3[1][1] - a3[0][1]);
        vec_approx_eq(&cc3, &cck_ref, 70.7); // inaccurate data due to digitization

        // drawing
        if SAVE_FIGURE {
            let mut plot = draw_curve(&mut b, pp, None, 1.0);
            let mut canvas = Canvas::new();
            let mut text = Text::new();
            canvas
                .set_arrow_scale(10.0)
                .set_face_color("grey")
                .set_edge_color("grey");
            text.set_fontsize(12.0)
                .set_align_horizontal("left")
                .set_align_vertical("center");
            let xi = cc0[0];
            let yi = cc0[1];
            let mut xf = cc0[0] + cc1[0] / sf;
            let mut yf = cc0[1] + cc1[1] / sf;
            canvas.draw_arrow(xi, yi, xf, yf);
            text.draw(xf, yf, "${\\bf C}^{(1)}$");
            xf = cc0[0] + cc2[0] / sf;
            yf = cc0[1] + cc2[1] / sf;
            canvas.draw_arrow(xi, yi, xf, yf);
            text.set_align_vertical("top").draw(xf, yf, "${\\bf C}^{(2)}$");
            xf = cc0[0] + cc3[0] / sf;
            yf = cc0[1] + cc3[1] / sf;
            canvas.draw_arrow(xi, yi, xf, yf);
            text.set_align_vertical("top").draw(xf, yf, "${\\bf C}^{(3)}$");
            plot.add(&canvas)
                .add(&text)
                .set_xmax(2000.0)
                .set_ymin(0.0)
                .set_ymax(1700.0)
                .set_equal_axes(true)
                .save("/tmp/russell_lab/test_bspline_curve_derivs_1.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_curve_derivs_alg1() {
        // allocate B-spline (Figure 3.15 on page 96 of Ref 1)
        let p = 2;
        let uu = &[0.0, 0.0, 0.0, 2.0 / 5.0, 3.0 / 5.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();

        // set control points
        let pp = &[
            [0.0, 0.0],  // P0
            [0.5, 1.0],  // P1
            [1.75, 1.0], // P2
            [2.0, 0.0],  // P3
            [2.5, 0.5],  // P4
        ];
        b.set_control_points(pp).unwrap();

        // check derivative @ u = 0.0; slope(P1-P0) = (1.0-0.0)/(0.5-0.0) = 2.0
        b.curve_derivs_alg1(0.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 2.0);

        // check derivative @ u = 2/5; slope(P2-P1) = (1.0-1.0)/(1.75-0.5) = 0.0
        b.curve_derivs_alg1(2.0 / 5.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 0.0);

        // check derivative @ u = 3/5; slope(P3-P2) = (0.0-1.0)/(2.0-1.75) = -4.0
        b.curve_derivs_alg1(3.0 / 5.0, 1);
        approx_eq(b.cc_ders[1][1] / b.cc_ders[1][0], -4.0, 1e-14);

        // check derivative @ u = 1; slope(P4-P3) = (0.5-0.0)/(2.5-2.0) = 1.0
        b.curve_derivs_alg1(1.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 1.0);

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, Some(&[0.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 1.0]), 1.0);
            plot.save("/tmp/russell_lab/test_bspline_curve_derivs_alg1.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_curve_derivs_alg2() {
        // allocate B-spline (Figure 3.15 on page 96 of Ref 1)
        let p = 2;
        let uu = &[0.0, 0.0, 0.0, 2.0 / 5.0, 3.0 / 5.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();

        // set control points
        let pp = &[
            [0.0, 0.0],  // P0
            [0.5, 1.0],  // P1
            [1.75, 1.0], // P2
            [2.0, 0.0],  // P3
            [2.5, 0.5],  // P4
        ];
        b.set_control_points(pp).unwrap();

        // check derivative @ u = 0.0; slope(P1-P0) = (1.0-0.0)/(0.5-0.0) = 2.0
        b.curve_derivs_alg2(0.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 2.0);

        // check derivative @ u = 2/5; slope(P2-P1) = (1.0-1.0)/(1.75-0.5) = 0.0
        b.curve_derivs_alg2(2.0 / 5.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 0.0);

        // check derivative @ u = 3/5; slope(P3-P2) = (0.0-1.0)/(2.0-1.75) = -4.0
        b.curve_derivs_alg2(3.0 / 5.0, 1);
        approx_eq(b.cc_ders[1][1] / b.cc_ders[1][0], -4.0, 1e-14);

        // check derivative @ u = 1; slope(P4-P3) = (0.5-0.0)/(2.5-2.0) = 1.0
        b.curve_derivs_alg2(1.0, 1);
        assert_eq!(b.cc_ders[1][1] / b.cc_ders[1][0], 1.0);

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, Some(&[0.0, 1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 1.0]), 1.0);
            plot.save("/tmp/russell_lab/test_bspline_curve_derivs_alg2.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_calc_curve_derivs_and_get_curve_deriv_1() {
        // allocate B-spline
        let p = 3;
        let uu = &[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();

        // set control points
        let pp = &[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]];
        b.set_control_points(pp).unwrap();

        // get point on the curve
        let knot = 0.5;
        let mut cc = Vector::new(2);
        b.calc_point(&mut cc, knot, false).unwrap();

        // calculate derivatives
        let mut cck = Vector::new(2);
        b.calc_curve_derivs(knot, 10, false); // will be capped at 3

        // check derivative: 0
        b.get_curve_deriv(&mut cck, 0); // the zero-th derivative equals the point on the curve
        vec_approx_eq(&cck, &cc, 1e-15);

        // check derivative: 1
        b.get_curve_deriv(&mut cck, 1);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 1.5);

        // check derivative: 2
        b.get_curve_deriv(&mut cck, 2);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 0.0);

        // check derivative: 3
        b.get_curve_deriv(&mut cck, 3);
        assert_eq!(cck[0], -48.0); // need find a reference to this result
        assert_eq!(cck[1], -12.0);

        // check derivative: 4
        b.get_curve_deriv(&mut cck, 4);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 0.0);

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, Some(&[0.5]), 1.0);
            plot.save("/tmp/russell_lab/test_calc_bspline_curve_derivs_and_get_curve_deriv_1.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_calc_curve_derivs_and_get_curve_deriv_2() {
        // allocate B-spline
        let p = 3;
        let uu = &[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();

        // set control points
        let pp = &[[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]];
        b.set_control_points(pp).unwrap();

        // get point on the curve
        let knot = 0.5;
        let mut cc = Vector::new(2);
        b.calc_point(&mut cc, knot, false).unwrap();

        // calculate derivatives
        let mut cck = Vector::new(2);
        b.calc_curve_derivs(knot, 10, true); // will be capped at 3

        // check derivative: 0
        b.get_curve_deriv(&mut cck, 0); // the zero-th derivative equals the point on the curve
        vec_approx_eq(&cck, &cc, 1e-15);

        // check derivative: 1
        b.get_curve_deriv(&mut cck, 1);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 1.5);

        // check derivative: 2
        b.get_curve_deriv(&mut cck, 2);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 0.0);

        // check derivative: 3
        b.get_curve_deriv(&mut cck, 3);
        assert_eq!(cck[0], -48.0); // need find a reference to this result
        assert_eq!(cck[1], -12.0);

        // check derivative: 4
        b.get_curve_deriv(&mut cck, 4);
        assert_eq!(cck[0], 0.0);
        assert_eq!(cck[1], 0.0);

        // drawing
        if SAVE_FIGURE {
            let plot = draw_curve(&mut b, pp, Some(&[0.5]), 1.0);
            plot.save("/tmp/russell_lab/test_bspline_calc_curve_derivs_and_get_curve_deriv_2.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_curve_derivs_ana_num() {
        // function returning the x-component of the curve point
        let fx = |u: f64, b: &mut Bspline| {
            let mut cc = Vector::new(2);
            b.calc_point(&mut cc, u, false).unwrap();
            Ok(cc[0])
        };

        // function returning the y-component of the curve point
        let fy = |u: f64, b: &mut Bspline| {
            let mut cc = Vector::new(2);
            b.calc_point(&mut cc, u, false).unwrap();
            Ok(cc[1])
        };

        // allocate B-splines (Figure 3.15 on page 96 of Ref 1)
        let p = 2;
        let uu = &[0.0, 0.0, 0.0, 2.0 / 5.0, 3.0 / 5.0, 1.0, 1.0, 1.0];
        let mut b = Bspline::new(p, uu).unwrap();

        // set control points
        let pp = &[
            [0.0, 0.0],  // P0
            [0.5, 1.0],  // P1
            [1.75, 1.0], // P2
            [2.0, 0.0],  // P3
            [2.5, 0.5],  // P4
        ];
        b.set_control_points(pp).unwrap();

        // stations to check the derivatives
        let knots_tols = [
            (0.0, 1e-15, 1e-9),
            (0.1, 1e-12, 1e-8),
            (0.2, 1e-13, 1e-7),
            (0.3, 1e-12, 1e-8),
            (0.4, 1.2e-6, 1e-8),
            (0.5, 1.2e-12, 1e-8),
            (0.6, 1.4e-6, 1e-7),
            (0.7, 1.2e-12, 1e-7),
            (1.0, 1e-12, 1e-7),
        ];

        // output header
        println!(
            "{:>3} {:>15} {:>15} {:>15} | {:>15} {:>15} {:>15} | {:>15} {:>15} {:>15} | {:>15} {:>15} {:>15}",
            "u",
            "alg1: dfx_du",
            "alg2: dfx_du",
            "num: dfx_du",
            "alg1: dfy_du",
            "alg2: dfy_du",
            "num: dfy_du",
            "alg1: d2fx_du2",
            "alg2: d2fx_du2",
            "num: d2fx_du2",
            "alg1: d2fy_du2",
            "alg2: d2fy_du2",
            "num: d2fy_du2"
        );

        // check derivatives
        let upto = 2;
        let mut cck = Vector::new(2);
        for (u, tol1, tol2) in knots_tols.iter().copied() {
            // alg1
            b.calc_curve_derivs(u, upto, false);
            b.get_curve_deriv(&mut cck, 1);
            let dfx_du_alg1 = cck[0];
            let dfy_du_alg1 = cck[1];
            b.get_curve_deriv(&mut cck, 2);
            let d2fx_du2_alg1 = cck[0];
            let d2fy_du2_alg1 = cck[1];

            // alg2 (control points)
            b.calc_curve_derivs(u, upto, true);
            b.get_curve_deriv(&mut cck, 1);
            let dfx_du_alg2 = cck[0];
            let dfy_du_alg2 = cck[1];
            b.get_curve_deriv(&mut cck, 2);
            let d2fx_du2_alg2 = cck[0];
            let d2fy_du2_alg2 = cck[1];

            // numerical
            let dfx_du_num = deriv1_central5(u, &mut b, fx).unwrap();
            let dfy_du_num = deriv1_central5(u, &mut b, fy).unwrap();
            let d2fx_du2_num = deriv2_forward8(u, &mut b, fx).unwrap();
            let d2fy_du2_num = deriv2_forward8(u, &mut b, fy).unwrap();

            // output
            println!(
                "{:>3} {:>15.10} {:>15.10} {:>15.10} | {:>15.10} {:>15.10} {:>15.10} | {:>15.10} {:>15.10} {:>15.10} | {:>15.10} {:>15.10} {:>15.10}",
                u,
                dfx_du_alg1,
                dfx_du_alg2,
                dfx_du_num,
                dfy_du_alg1,
                dfy_du_alg2,
                dfy_du_num,
                d2fx_du2_alg1,
                d2fx_du2_alg2,
                d2fx_du2_num,
                d2fy_du2_alg1,
                d2fy_du2_alg2,
                d2fy_du2_num
            );

            // compare alg1 and alg2
            approx_eq(dfx_du_alg2, dfx_du_alg1, 1e-15);
            approx_eq(dfy_du_alg2, dfy_du_alg1, 1e-15);
            approx_eq(d2fx_du2_alg2, d2fx_du2_alg1, 1e-14);
            approx_eq(d2fy_du2_alg2, d2fy_du2_alg1, 1e-14);

            // compare alg1 and numerical
            approx_eq(dfx_du_alg1, dfx_du_num, tol1);
            approx_eq(dfy_du_alg1, dfy_du_num, tol1);
            approx_eq(d2fx_du2_alg1, d2fx_du2_num, tol2);
            approx_eq(d2fy_du2_alg1, d2fy_du2_num, tol2);
        }
    }
}
