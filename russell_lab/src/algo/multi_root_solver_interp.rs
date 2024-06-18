#![allow(unused)]

use crate::{mat_gen_eigen, Matrix, Vector};
use crate::{vec_norm, Norm, StrError};

/// Tolerance to avoid division by zero and discard near-zero beta (the denominator of eigenvalue) values
const TOL_BETA: f64 = 100.0 * f64::EPSILON;

/// Tolerance to discard root values outside the [xa, xb] range
const TOL_X_RANGE: f64 = 100.0 * f64::EPSILON;

pub struct MultiRootSolverInterp {
    /// Number of grid points (= N + 1)
    npoint: usize,

    /// Weights (aka, lambda)
    w: Vector,

    /// Companion matrix A
    aa: Matrix,

    /// Companion matrix B
    bb: Matrix,

    /// Balancing coefficients
    balance: Vector,

    /// Real part of the denominator to calculate the eigenvalues
    alpha_real: Vector,

    /// Imaginary part of the denominator to calculate the eigenvalues
    alpha_imag: Vector,

    /// Denominators to calculate the eigenvalues
    beta: Vector,

    /// The eigenvectors (as columns)
    v: Matrix,

    /// Possible roots
    roots: Vector,
}

impl MultiRootSolverInterp {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `yy` -- holds the coordinates of the interpolant in `[-1, 1]` (e.g., Chebyshev-Gauss-Lobatto points)
    ///
    /// # Notes
    ///
    /// 1. The number of points must be ≥ 2; i.e., `yy.len() ≥ 2`
    /// 2. The interpolant's degree `N` is equal to `npoint - 1`
    pub fn new(yy: &Vector) -> Result<Self, StrError> {
        if yy.dim() < 2 {
            return Err("at least 2 grid points are required");
        }
        let npoint = yy.dim();
        let nc = 1 + npoint; // companion matrix' dimension
        let mut w = Vector::new(npoint);
        let mut aa = Matrix::new(nc, nc);
        let mut bb = Matrix::new(nc, nc);
        for j in 0..npoint {
            let mut prod = 1.0;
            for k in 0..npoint {
                if k != j {
                    if yy[j] == yy[k] {
                        return Err("grid points must not coincide");
                    }
                    prod *= yy[j] - yy[k];
                }
            }
            w[j] = 1.0 / prod;
            aa.set(1 + j, 1 + j, yy[j]);
            bb.set(1 + j, 1 + j, 1.0);
        }
        Ok(MultiRootSolverInterp {
            npoint,
            w,
            aa,
            bb,
            balance: Vector::new(nc),
            alpha_real: Vector::new(nc),
            alpha_imag: Vector::new(nc),
            beta: Vector::new(nc),
            v: Matrix::new(nc, nc),
            roots: Vector::new(nc),
        })
    }

    /// Finds multiple roots using the Lagrange interpolation method
    ///
    /// The problem coordinates are `x ∈ [a, b]` and the grid coordinates are `y ∈ [-1, 1]`
    /// Thus, consider the mapping:
    ///
    /// ```text
    ///        2 x - xb - xa
    /// y(x) = —————————————
    ///           xb - xa
    /// ```
    ///
    /// And
    ///
    /// ```text
    ///        xb + xa + (xb - xa) y
    /// x(y) = —————————————————————
    ///                 2
    /// ```
    ///
    /// The interpolated values are:
    ///
    /// ```text
    /// Uⱼ = f(Xⱼ(Yⱼ))
    ///
    /// where xa ≤ Xⱼ ≤ xb
    /// and   -1 ≤ Yⱼ ≤ 1
    /// ```
    ///
    /// # Input
    ///
    /// * `uu` -- holds the function evaluations at the grid points
    /// * `xa` -- the lower bound (must `be < xb`)
    /// * `xb` -- the upper bound (must `be > xa`)
    ///
    /// # Output
    ///
    /// Returns the roots, sorted in ascending order
    pub fn find(&mut self, uu: &Vector, xa: f64, xb: f64) -> Result<&[f64], StrError> {
        // check
        if uu.dim() != self.npoint {
            return Err("U vector must have the same dimension as the grid points vector");
        }
        if xb <= xa {
            return Err("xb must be greater than xa");
        }

        // balancing coefficients
        let nc = 1 + self.npoint;
        self.balance[0] = 1.0;
        for i in 0..self.npoint {
            if uu[i] == 0.0 {
                self.balance[1 + i] = 1.0;
            } else {
                self.balance[1 + i] = f64::sqrt(f64::abs(self.w[i]) / f64::abs(uu[i]));
            }
        }

        // scaling coefficients
        let mut row0 = Vector::new(nc);
        let mut col0 = Vector::new(nc);
        for j in 0..self.npoint {
            let s = self.balance[1 + j];
            let t = 1.0 / s;
            row0[1 + j] = -uu[j] * s;
            col0[1 + j] = t * self.w[j];
        }
        let sl = vec_norm(&row0, Norm::Euc);
        let sr = vec_norm(&col0, Norm::Euc);

        // (balanced) companion matrix
        for k in 1..nc {
            self.aa.set(0, k, sl * row0[k]);
            self.aa.set(k, 0, col0[k] * sr);
        }

        // generalized eigenvalues
        mat_gen_eigen(
            &mut self.alpha_real,
            &mut self.alpha_imag,
            &mut self.beta,
            &mut self.v,
            &mut self.aa,
            &mut self.bb,
        )?;

        // roots = real eigenvalues
        let mut nroot = 0;
        for k in 0..nc {
            let imaginary = f64::abs(self.alpha_imag[k]) > 0.0;
            let infinite = f64::abs(self.beta[k]) < TOL_BETA;
            if !imaginary && !infinite {
                let y_root = self.alpha_real[k] / self.beta[k];
                let x_root = (xb + xa + (xb - xa) * y_root) / 2.0;
                println!(
                    "alpha = ({}, {}), beta = {:.e}, lambda = {}, x_root = {}",
                    self.alpha_real[k], self.alpha_imag[k], self.beta[k], y_root, x_root,
                );
                if x_root >= xa - TOL_X_RANGE && x_root <= xb + TOL_X_RANGE {
                    self.roots[nroot] = x_root;
                    nroot += 1;
                }
            }
        }
        for i in nroot..nc {
            self.roots[i] = f64::MAX;
        }
        self.roots.as_mut_data().sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(&self.roots.as_data()[..nroot])
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::MultiRootSolverInterp;
    use crate::algo::NoArgs;
    use crate::math::chebyshev_lobatto_points;
    use crate::StrError;
    use crate::{array_approx_eq, Vector};
    use plotpy::{Curve, Plot};

    const SAVE_FIGURE: bool = true;

    fn graph<F, A>(name: &str, xa: f64, xb: f64, roots: &[f64], args: &mut A, mut f: F)
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let xx = Vector::linspace(xa, xb, 101).unwrap();
        let yy = xx.get_mapped(|x| f(x, args).unwrap());
        let mut curve = Curve::new();
        let mut zeros = Curve::new();
        zeros
            .set_marker_style("o")
            .set_marker_color("red")
            .set_marker_void(true)
            .set_line_style("None");
        for root in roots {
            zeros.draw(&[*root], &[f(*root, args).unwrap()]);
        }
        curve.draw(xx.as_data(), yy.as_data());
        let mut plot = Plot::new();
        plot.add(&curve)
            .add(&zeros)
            .set_cross(0.0, 0.0, "gray", "-", 1.0)
            .grid_and_labels("x", "f(x)")
            .save(&format!("/tmp/russell/{}.svg", name));
    }

    #[test]
    fn multi_root_solver_interp_works_simple() {
        // function
        let f = |x, _: &mut NoArgs| -> Result<f64, StrError> { Ok(x * x - 1.0) };
        let (xa, xb) = (-4.0, 4.0);

        // grid points
        let nn = 44;
        // for nn in 2..128 {
        let yy = chebyshev_lobatto_points(nn);

        // evaluate the data over grid points
        let npoint = nn + 1;
        let mut uu = Vector::new(npoint);
        let args = &mut 0;
        for i in 0..npoint {
            let x = (xb + xa + (xb - xa) * yy[i]) / 2.0;
            uu[i] = f(x, args).unwrap();
        }

        // solver
        let mut solver = MultiRootSolverInterp::new(&yy).unwrap();

        // find roots
        let roots = solver.find(&uu, xa, xb).unwrap();
        println!("N = {}, roots = {:?}", nn, roots);
        // array_approx_eq(roots, &[-1.0, 1.0], 1e-14);
        // }

        if SAVE_FIGURE {
            graph("test_multi_root_solver_interp_simple", xa, xb, roots, args, f);
        }
    }
}
