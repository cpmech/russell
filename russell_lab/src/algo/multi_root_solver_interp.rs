use crate::StrError;
use crate::{mat_gen_eigen, Matrix, Vector};

pub struct MultiRootSolverInterp {
    /// Number of grid points (= N + 1)
    npoint: usize,

    /// Companion matrix A
    aa: Matrix,

    /// Companion matrix B
    bb: Matrix,

    // For the generalized eigenvalues
    alpha_real: Vector,
    alpha_imag: Vector,
    beta: Vector,
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
            let wj = 1.0 / prod;
            aa.set(1 + j, 0, wj);
            aa.set(1 + j, 1 + j, yy[j]);
            bb.set(1 + j, 1 + j, 1.0);
        }
        Ok(MultiRootSolverInterp {
            npoint,
            aa: Matrix::new(nc, nc),
            bb: Matrix::new(nc, nc),
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
    ///
    /// # Output
    ///
    /// Returns the roots
    pub fn find(&mut self, uu: &Vector, xa: f64, xb: f64) -> Result<&[f64], StrError> {
        // check
        if uu.dim() != self.npoint {
            return Err("U vector must have the same dimension as the grid points vector");
        }

        // companion matrix
        for j in 0..self.npoint {
            self.aa.set(0, 1 + j, -uu[j]);
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
        let nc = 1 + self.npoint;
        let mut nroot = 0;
        for i in 0..nc {
            let imaginary = f64::abs(self.alpha_imag[i]) > f64::EPSILON;
            let infinite = f64::abs(self.beta[i]) < 10.0 * f64::EPSILON;
            if !imaginary && !infinite {
                let y_root = self.alpha_real[i] / self.beta[i];
                self.roots[nroot] = (xb + xa + (xb - xa) * y_root) / 2.0;
                nroot += 1;
            }
        }
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

    #[test]
    fn multi_root_solver_works_simple() {
        // function
        let f = |x, _: &mut NoArgs| -> Result<f64, StrError> { Ok(x * x - 1.0) };
        let (xa, xb) = (-4.0, 4.0);

        // grid points
        let nn = 2;
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
        array_approx_eq(roots, &[-1.0, 1.0], 1e-15);
    }
}
