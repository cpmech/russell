#![allow(unused)]

use crate::math::PI;
use crate::{cpx, mat_eigen, mat_vec_mul, vec_mat_mul, StrError};
use crate::{Matrix, Vector};
use num_complex::{Complex64, ComplexFloat};

const TOL_TAU: f64 = 1.0e-8; // discard roots with abs(Im(root)) > tau
const TOL_SIGMA: f64 = 1.0e-6; // discard roots such that abs(Re(root)) > (1 + sigma)

pub struct MultiRootSolverCheby {
    /// Degree N
    nn: usize,

    /// Chebyshev-Gauss-Lobatto coordinates
    yy: Vector,

    /// Interpolation matrix P
    pp: Matrix,

    /// Companion matrix A
    aa: Matrix,

    /// Function evaluations at the grid points (Chebyshev-Gauss-Lobatto)
    u: Vector,

    /// Coefficients of interpolation: c = P u
    c: Vector,

    /// Possible roots
    roots: Vector,
}

impl MultiRootSolverCheby {
    /// Allocates a new instance
    pub fn new(nn: usize) -> Result<Self, StrError> {
        // check
        if nn < 2 {
            return Err("the degree N must be ≥ 2");
        }

        // Chebyshev-Gauss-Lobatto coordinates
        let yy = standard_chebyshev_lobatto_points(nn);

        // println!("yy =\n{}", yy);

        // interpolation matrix (Boyd's phia)
        let nf = nn as f64;
        let np = nn + 1;
        let mut pp = Matrix::new(np, np);
        for j in 0..np {
            let jf = j as f64;
            let qj = if j == 0 || j == nn { 2.0 } else { 1.0 };
            for k in 0..np {
                let kf = k as f64;
                let qk = if k == 0 || k == nn { 2.0 } else { 1.0 };
                pp.set(j, k, 2.0 / (qj * qk * nf) * f64::cos(PI * jf * kf / nf));
            }
        }
        println!("P =\n{:.4}", pp);

        // companion matrix (except last row)
        let mut aa = Matrix::new(nn, nn);
        aa.set(0, 1, 1.0);
        for r in 1..(nn - 1) {
            aa.set(r, r + 1, 0.5); // upper diagonal
            aa.set(r, r - 1, 0.5); // lower diagonal
        }
        println!("A =\n{:.4}", aa);

        // done
        Ok(MultiRootSolverCheby {
            nn,
            yy,
            pp,
            aa,
            u: Vector::new(np),
            c: Vector::new(np),
            roots: Vector::new(nn),
        })
    }

    // pub fn interp<F, A>(&mut self, x: f64, args: &mut A, mut f: F) -> f64
    // where
    //     F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    // {
    //     // function evaluations at the grid points
    //     let nf = self.nn as f64;
    //     let np = self.nn + 1;
    //     for k in 0..np {
    //         let y = f64::cos(PI * (k as f64) / nf); // Chebyshev-Gauss-Lobatto
    //         let x = (xb + xa + (xb - xa) * y) / 2.0;
    //         self.u[k] = f(x, args).unwrap();
    //     }
    //     Err("STOP")
    // }

    pub fn find_given_data(&mut self, xa: f64, xb: f64, uu: &Vector) -> Result<&[f64], StrError> {
        // expansion coefficients
        let nn = self.nn;
        let np = nn + 1;
        let mut gamma = Vector::new(np);
        vec_mat_mul(&mut gamma, 1.0, &uu, &self.pp).unwrap();
        println!("gamma =\n{}", gamma);
        let gamma_n = gamma[nn];
        if f64::abs(gamma_n) < 10.0 * f64::EPSILON {
            println!("leading expansion coefficient vanishes; try smaller degree N");
            // return Err("leading expansion coefficient vanishes; try smaller degree N");
        }
        for k in 0..np {
            gamma[k] = -0.5 * gamma[k] / gamma_n;
        }
        println!("gamma (normalized) =\n{}", gamma);

        // nonstandard companion matrix
        let mut cc = Matrix::new(nn, nn);
        for i in 0..(nn - 1) {
            cc.set(i, i + 1, 0.5);
            cc.set(i + 1, i, 0.5);
        }
        cc.set(0, 1, 1.0);
        for i in 0..nn {
            cc.add(nn - 1, i, gamma[i]); // last row
        }

        // last row of the companion matrix
        for i in 0..nn {
            self.aa.set(nn - 1, i, gamma[i]); // last row
        }
        self.aa.add(nn - 1, nn - 2, 0.5);
        println!("A =\n{:.4}", self.aa);

        // eigenvalues
        let mut l_real = Vector::new(nn);
        let mut l_imag = Vector::new(nn);
        let mut v_real = Matrix::new(nn, nn);
        let mut v_imag = Matrix::new(nn, nn);
        mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut self.aa).unwrap();

        println!("l_real =\n{}", l_real);
        println!("l_imag =\n{}", l_imag);

        // roots = real eigenvalues within the interval
        let mut nroot = 0;
        for i in 0..nn {
            if f64::abs(l_imag[i]) < TOL_TAU * f64::abs(l_real[i]) {
                if f64::abs(l_real[i]) <= (1.0 + TOL_SIGMA) {
                    self.roots[nroot] = (xb + xa + (xb - xa) * l_real[i]) / 2.0;
                    nroot += 1;
                }
            }
        }

        // sort roots
        for i in nroot..self.nn {
            self.roots[i] = f64::MAX;
        }
        self.roots.as_mut_data().sort_by(|a, b| a.partial_cmp(b).unwrap());

        // results
        Ok(&self.roots.as_data()[..nroot])
    }
}

/// Returns the standard (from 1 to -1) Chebyshev-Gauss-Lobatto coordinates
fn standard_chebyshev_lobatto_points(nn: usize) -> Vector {
    let mut yy = Vector::new(nn + 1);
    yy[0] = 1.0;
    yy[nn] = -1.0;
    if nn < 3 {
        return yy;
    }
    let nf = nn as f64;
    let d = 2.0 * nf;
    let l = if (nn & 1) == 0 {
        // even number of segments
        nn / 2
    } else {
        // odd number of segments
        (nn + 3) / 2 - 1
    };
    for i in 1..l {
        yy[nn - i] = -f64::sin(PI * (nf - 2.0 * (i as f64)) / d);
        yy[i] = -yy[nn - i];
    }
    yy
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::MultiRootSolverCheby;
    use crate::algo::NoArgs;
    use crate::array_approx_eq;
    use crate::math::PI;
    use crate::StrError;
    use crate::Vector;
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
    fn multi_root_solver_cheby_simple() {
        // function
        let f = |x, _: &mut NoArgs| -> Result<f64, StrError> { Ok(x * x - 1.0) };
        let (xa, xb) = (-4.0, 4.0);

        // degree
        let nn = 2; // 7 causes Inf and NaN which causes segmentation fault in LAPACK

        // solver
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();

        // data
        let np = nn + 1;
        let mut uu = Vector::new(np);
        let args = &mut 0;
        for i in 0..np {
            let x = (xb + xa + (xb - xa) * solver.yy[i]) / 2.0;
            // println!("x = {}", x);
            uu[i] = f(x, args).unwrap();
        }
        println!("U =\n{}", uu);

        // find roots
        let roots = solver.find_given_data(xa, xb, &uu).unwrap();
        println!("N = {}, roots = {:?}", nn, roots);
        // array_approx_eq(roots, &[-1.0, 1.0], 1e-14);

        if SAVE_FIGURE {
            graph("test_multi_root_solver_cheby_simple", xa, xb, roots, args, f);
        }
    }

    #[test]
    fn multi_root_solver_cheby_day_romero_paper() {
        let nn = 20;
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();

        let np = nn + 1;
        let mut uu = Vector::new(np);
        for i in 0..np {
            let w = 3.0 * solver.yy[i] + 4.0;
            uu[i] = f64::cos(PI * w) - 1.0 / f64::cosh(PI * w);
        }

        let mut solver = MultiRootSolverCheby::new(nn).unwrap();
        let roots = solver.find_given_data(-1.0, 1.0, &uu).unwrap();
        println!("N = {}, roots = {:?}", nn, roots);

        if SAVE_FIGURE {
            let f = |x, _: &mut NoArgs| -> Result<f64, StrError> {
                let w = 3.0 * x + 4.0;
                Ok(f64::cos(PI * w) - 1.0 / f64::cosh(PI * w))
            };
            let (xa, xb) = (-1.0, 1.0);
            let args = &mut 0;
            graph("test_multi_root_solver_cheby_day_romero_paper", xa, xb, roots, args, f);
        }
    }
}
