#![allow(unused)]

use crate::math::{chebyshev_tn, PI};
use crate::{approx_eq, array_approx_eq, mat_eigen, mat_vec_mul, Stopwatch};
use crate::{vec_approx_eq, StrError};
use crate::{InterpGrid, InterpLagrange, InterpParams, RootSolverBrent};
use crate::{Matrix, Vector};

/// Tolerance to avoid division by zero on the trailing Chebyshev coefficient
const TOL_EPS: f64 = 1.0e-13;

/// Tolerance to discard roots with abs(Im(root)) > tau
const TOL_TAU: f64 = 1.0e-8;

/// Tolerance to discard roots such that abs(Re(root)) > (1 + sigma)
const TOL_SIGMA: f64 = 1.0e-6;

pub struct MultiRootSolverCheby {
    /// Degree N
    nn: usize,

    /// Number of grid points (= N + 1)
    np: usize,

    /// Chebyshev-Gauss-Lobatto coordinates
    yy: Vector,

    /// Interpolation matrix P
    pp: Matrix,

    /// Companion matrix A
    aa: Matrix,

    /// Function evaluations at the (standard) Chebyshev-Gauss-Lobatto grid points
    u: Vector,

    /// Coefficients of interpolation: c = P u
    c: Vector,

    /// Possible roots
    roots: Vector,

    /// Indicates whether the data vector (u) has been set or not
    all_data_set: bool,

    /// Lower bound
    xa: f64,

    /// Upper bound
    xb: f64,
}

// Chebyshev-Gauss-Lobatto
fn chebyshev_coefficients<F, A>(
    workspace_uu: &mut [f64],
    workspace_cc: &mut [f64],
    nn: usize,
    xa: f64,
    xb: f64,
    args: &mut A,
    f: &mut F,
) where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    // check
    let np = nn + 1;
    assert!(nn > 0);
    assert!(xb > xa);
    assert!(workspace_uu.len() >= np);
    assert!(workspace_cc.len() >= np);

    // data vector
    let nf = nn as f64;
    let uu = &mut workspace_uu[0..np];
    for k in 0..np {
        let kf = k as f64;
        let x = (xb + xa + (xb - xa) * f64::cos(PI * kf / nf)) / 2.0;
        uu[k] = (*f)(x, args).unwrap();
    }

    // coefficients
    let cc = &mut workspace_cc[0..np];
    for j in 0..np {
        let jf = j as f64;
        let qj = if j == 0 || j == nn { 2.0 } else { 1.0 };
        cc[j] = 0.0;
        for k in 0..np {
            let kf = k as f64;
            let qk = if k == 0 || k == nn { 2.0 } else { 1.0 };
            cc[j] += uu[k] * 2.0 * f64::cos(PI * jf * kf / nf) / (qj * qk * nf);
        }
    }
}

fn chebyshev_interpolation(x: f64, xa: f64, xb: f64, cc: &Vector) -> f64 {
    assert!(xb > xa);
    let y = (2.0 * x - xb - xa) / (xb - xa);
    let mut sum = 0.0;
    for k in 0..cc.dim() {
        sum += cc[k] * chebyshev_tn(k, y);
    }
    sum
}

pub fn adaptive_interpolation<F, A>(
    nn_max: usize,
    tolerance: f64,
    xa: f64,
    xb: f64,
    args: &mut A,
    mut f: F,
) -> Result<usize, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    if nn_max > 2048 {
        return Err("max N must be ≤ 2048");
    }
    let np_max = nn_max + 1;
    let mut workspace_uu = vec![0.0; np_max];
    let mut workspace_cc = vec![0.0; np_max];
    let nn = 1;
    chebyshev_coefficients(&mut workspace_uu, &mut workspace_cc, nn, xa, xb, args, &mut f);
    let mut cn_prev = workspace_cc[nn];
    println!("N = {}, cn = {}", nn, cn_prev);
    for nn in 2..nn_max {
        chebyshev_coefficients(&mut workspace_uu, &mut workspace_cc, nn, xa, xb, args, &mut f);
        let cn = workspace_cc[nn];
        println!("N = {}, cn = {}", nn, cn);
        if f64::abs(cn_prev) < tolerance && f64::abs(cn) < tolerance {
            return Ok(nn - 2);
        }
        cn_prev = cn;
    }
    Err("adaptive interpolation did not converge")
}

impl MultiRootSolverCheby {
    /// Allocates a new instance
    pub fn new(nn: usize) -> Result<Self, StrError> {
        // check
        if nn < 2 {
            return Err("the degree N must be ≥ 2");
        }

        // standard Chebyshev-Gauss-Lobatto coordinates
        // let yy = standard_chebyshev_lobatto_points(nn);
        let yy = Vector::new(nn + 1);

        // interpolation matrix
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
        // println!("P = \n{:.5}", pp);

        // companion matrix (except last row)
        let mut aa = Matrix::new(nn, nn);
        aa.set(0, 1, 1.0);
        for r in 1..(nn - 1) {
            aa.set(r, r + 1, 0.5); // upper diagonal
            aa.set(r, r - 1, 0.5); // lower diagonal
        }

        // done
        Ok(MultiRootSolverCheby {
            nn,
            np,
            yy,
            pp,
            aa,
            u: Vector::new(np),
            c: Vector::new(np),
            roots: Vector::new(nn),
            all_data_set: false,
            xa: -1.0,
            xb: 1.0,
        })
    }

    /// Sets the data vector (u) from the function evaluated at the standard Chebyshev-Gauss-Lobatto points
    pub fn set_data_from_function<F, A>(&mut self, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<(), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // check
        if xb <= xa + TOL_EPS {
            return Err("xb must be greater than xa + ϵ");
        }

        // set data vector
        for k in 0..self.np {
            let x = (xb + xa + (xb - xa) * self.yy[k]) / 2.0;
            self.u[k] = f(x, args).unwrap();
        }

        // calculate the Chebyshev coefficients
        mat_vec_mul(&mut self.c, 1.0, &self.pp, &self.u).unwrap();
        // println!("c = \n{:.5}", self.c);

        // done
        self.all_data_set = true;
        self.xa = xa;
        self.xb = xb;
        Ok(())
    }

    /// Computes the interpolated function
    ///
    /// **Warning:** The data vector (u) must be set first.
    pub fn interp(&self, x: f64) -> Result<f64, StrError> {
        if !self.all_data_set {
            return Err("The data vector (u) must be set first");
        }
        if x < self.xa || x > self.xb {
            return Err("x must be in [xa, xb]");
        }
        let y = (2.0 * x - self.xb - self.xa) / (self.xb - self.xa);
        let mut sum = 0.0;
        for k in 0..self.np {
            sum += self.c[k] * chebyshev_tn(k, y);
        }
        Ok(sum)
    }

    /// Find all roots in the interval
    ///
    /// **Warning:** The data vector (u) must be set first.
    pub fn find(&mut self) -> Result<&[f64], StrError> {
        // check
        if !self.all_data_set {
            return Err("The data vector (u) must be set first");
        }

        // expansion coefficients
        let nn = self.nn;
        let cn = self.c[nn];
        if f64::abs(cn) < TOL_EPS {
            return Err("trailing Chebyshev coefficient vanishes; try another degree N");
        }

        // last row of the companion matrix
        for k in 0..nn {
            self.aa.set(nn - 1, k, -0.5 * self.c[k] / cn);
        }
        self.aa.add(nn - 1, nn - 2, 0.5);
        // println!("A =\n{:.4}", self.aa);

        // eigenvalues
        let mut l_real = Vector::new(nn);
        let mut l_imag = Vector::new(nn);
        let mut v_real = Matrix::new(nn, nn);
        let mut v_imag = Matrix::new(nn, nn);
        mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut self.aa).unwrap();

        // println!("l_real =\n{}", l_real);
        // println!("l_imag =\n{}", l_imag);

        // roots = real eigenvalues within the interval
        let mut nroot = 0;
        for i in 0..nn {
            if f64::abs(l_imag[i]) < TOL_TAU * f64::abs(l_real[i]) {
                if f64::abs(l_real[i]) <= (1.0 + TOL_SIGMA) {
                    self.roots[nroot] = (self.xb + self.xa + (self.xb - self.xa) * l_real[i]) / 2.0;
                    nroot += 1;
                }
            }
        }

        // sort roots
        for i in nroot..nn {
            self.roots[i] = f64::MAX;
        }
        self.roots.as_mut_data().sort_by(|a, b| a.partial_cmp(b).unwrap());

        // results
        Ok(&self.roots.as_data()[..nroot])
    }
}

/// Polishes the root using Brent's method
pub fn polish_roots<F, A>(
    roots_out: &mut [f64],
    roots_in: &[f64],
    xa: f64,
    xb: f64,
    args: &mut A,
    mut f: F,
) -> Result<(), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let nr = roots_in.len();
    if nr < 2 {
        return Err("this function works with at least two roots");
    }
    let solver = RootSolverBrent::new();
    let l = nr - 1;
    for i in 0..nr {
        let xr = roots_in[i];
        if xr < xa || xr > xb {
            return Err("a root is outside [xa, xb]");
        }
        let a = if i == 0 {
            xa
        } else {
            (roots_in[i - 1] + roots_in[i]) / 2.0
        };
        let b = if i == l {
            xb
        } else {
            (roots_in[i] + roots_in[i + 1]) / 2.0
        };
        let fa = f(a, args)?;
        let fb = f(b, args)?;
        if fa * fb < 0.0 {
            let (xo, _) = solver.find(a, b, args, &mut f)?;
            roots_out[i] = xo;
        } else {
            roots_out[i] = roots_in[i];
        }
    }
    Ok(())
}

/// Returns the standard (from 1 to -1) Chebyshev-Gauss-Lobatto coordinates
fn standard_chebyshev_lobatto_points(yy: &mut [f64], nn: usize) {
    yy[0] = 1.0;
    yy[nn] = -1.0;
    if nn < 3 {
        return;
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::adaptive_interpolation;
    use super::{chebyshev_coefficients, MultiRootSolverCheby};
    use crate::algo::NoArgs;
    use crate::array_approx_eq;
    use crate::get_test_functions;
    use crate::math::PI;
    use crate::polish_roots;
    use crate::StrError;
    use crate::Vector;
    use crate::{InterpGrid, InterpLagrange, InterpParams};
    use plotpy::Legend;
    use plotpy::{Curve, Plot};

    const SAVE_FIGURE: bool = true;

    fn graph<F, A>(
        name: &str,
        xa: f64,
        xb: f64,
        solver: &MultiRootSolverCheby,
        roots_unpolished: &[f64],
        roots_polished: &[f64],
        args: &mut A,
        mut f: F,
    ) where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let xx = Vector::linspace(xa, xb, 101).unwrap();
        let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
        let yy_int = xx.get_mapped(|x| solver.interp(x).unwrap());
        let mut curve_ana = Curve::new();
        let mut curve_int = Curve::new();
        let mut zeros_unpolished = Curve::new();
        let mut zeros_polished = Curve::new();
        curve_ana.set_label("analytical");
        curve_int
            .set_label("interpolated")
            .set_line_style("--")
            .set_marker_style(".")
            .set_marker_every(5);
        zeros_unpolished
            .set_marker_style("o")
            .set_marker_void(true)
            .set_marker_line_color("#00760F")
            .set_line_style("None");
        zeros_polished
            .set_marker_style("s")
            .set_marker_size(10.0)
            .set_marker_void(true)
            .set_marker_line_color("#00760F")
            .set_line_style("None");
        for root in roots_unpolished {
            zeros_unpolished.draw(&[*root], &[solver.interp(*root).unwrap()]);
        }
        for root in roots_polished {
            zeros_polished.draw(&[*root], &[f(*root, args).unwrap()]);
        }
        curve_int.draw(xx.as_data(), yy_int.as_data());
        curve_ana.draw(xx.as_data(), yy_ana.as_data());
        let mut plot = Plot::new();
        let mut legend = Legend::new();
        legend.set_num_col(2);
        legend.set_outside(true);
        legend.draw();
        plot.add(&curve_ana)
            .add(&curve_int)
            .add(&zeros_unpolished)
            .add(&zeros_polished)
            .add(&legend)
            .set_cross(0.0, 0.0, "gray", "-", 1.5)
            .grid_and_labels("x", "f(x)")
            .save(&format!("/tmp/russell/{}.svg", name))
            .unwrap();
    }

    #[test]
    fn todo_works() {
        let mut f = |x: f64, _: &mut NoArgs| -> Result<f64, StrError> { Ok(x * x - 1.0) };
        let (xa, xb) = (-1.0, 1.0);
        let mut workspace_uu = vec![0.0; 2000];
        let mut workspace_cc = vec![0.0; 2000];
        let args = &mut 0;
        for nn in 1000..1001 {
            chebyshev_coefficients(&mut workspace_uu, &mut workspace_cc, nn, xa, xb, args, &mut f);
        }
    }

    #[test]
    fn adaptive_interpolation_works() {
        let mut f = |x: f64, _: &mut NoArgs| -> Result<f64, StrError> {
            // Ok(0.0)
            // Ok(x - 0.5)
            // Ok(x * x - 1.0)
            // Ok(x * x * x - 0.5)
            // Ok(x * x * x * x - 0.5)
            // Ok(x * x * x * x * x - 0.5)
            // Ok(f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x))
            // Ok(0.092834 * f64::sin(77.0001 + 19.87 * x))
            Ok(f64::ln(2.0 * f64::cos(x / 2.0)))
        };
        // let (xa, xb) = (-1.0, 1.0);
        // let (xa, xb) = (-2.34567, 12.34567);
        let (xa, xb) = (-0.995 * PI, 0.995 * PI);

        let nn_max = 400;
        let tol = 1e-8;
        let args = &mut 0;
        let nn = adaptive_interpolation(nn_max, tol, xa, xb, args, f).unwrap();
        println!("N = {}", nn);

        if SAVE_FIGURE {
            let np = nn + 1;
            let interp = InterpLagrange::new(nn, None).unwrap();
            let mut uu = Vector::new(np);
            for (i, y) in interp.get_points().into_iter().enumerate() {
                let x = (xb + xa + (xb - xa) * y) / 2.0;
                uu[i] = f(x, args).unwrap();
            }
            let xx = Vector::linspace(xa, xb, 201).unwrap();
            let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
            let yy_int = xx.get_mapped(|x| {
                let y = (2.0 * x - xb - xa) / (xb - xa);
                let yy = f64::max(-1.0, f64::min(1.0, y));
                interp.eval(yy, &uu).unwrap()
            });
            let mut curve_ana = Curve::new();
            let mut curve_int = Curve::new();
            curve_ana.set_label("analytical");
            curve_int
                .set_label("interpolated")
                .set_line_style("--")
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
                .save("/tmp/russell/test_adaptive_interpolation.svg")
                .unwrap();
        }
    }

    #[test]
    fn multi_root_solver_cheby_simple() {
        // function
        let f = |x, _: &mut NoArgs| -> Result<f64, StrError> {
            // Ok(x * x - 1.0)
            Ok(0.092834 * f64::sin(77.0001 + 19.87 * x))
            // Ok(f64::ln(2.0 * f64::cos(x / 2.0)))
        };
        // let (xa, xb) = (-4.0, 4.0);
        let (xa, xb) = (-2.34567, 12.34567);
        // let (xa, xb) = (-0.995 * PI, 0.995 * PI);

        // degree
        let nn = 146;

        // solver
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();

        // data
        let args = &mut 0;
        solver.set_data_from_function(xa, xb, args, f).unwrap();
        // println!("U =\n{}", solver.u);

        // find roots
        let roots_unpolished = Vec::from(solver.find().unwrap());
        let mut roots_polished = vec![0.0; roots_unpolished.len()];
        polish_roots(&mut roots_polished, &roots_unpolished, xa, xb, args, f).unwrap();
        println!("n_roots = {}", roots_polished.len());
        println!("roots_unpolished = {:?}", roots_unpolished);
        println!("roots_polished = {:?}", roots_polished);

        // figure
        if SAVE_FIGURE {
            graph(
                "test_multi_root_solver_cheby_simple",
                xa,
                xb,
                &solver,
                &roots_unpolished,
                &roots_polished,
                args,
                f,
            );
        }

        // check
        // array_approx_eq(&roots_polished, &[-1.0, 1.0], 1e-12);
    }

    #[test]
    fn multi_root_solver_cheby_works() {
        let tests = get_test_functions();
        let id = 4;
        let test = &tests[id];
        if test.root1.is_some() || test.root2.is_some() || test.root3.is_some() {
            println!("\n===================================================================");
            println!("\n{}", test.name);
            let (xa, xb) = test.range;
            let nn = 20;
            let mut solver = MultiRootSolverCheby::new(nn).unwrap();
            let args = &mut 0;
            solver.set_data_from_function(xa, xb, args, test.f).unwrap();
            let roots_unpolished = Vec::from(solver.find().unwrap());
            // let mut roots_polished = vec![0.0; roots_unpolished.len()];
            // polish_roots(&mut roots_polished, &roots_unpolished, xa, xb, args, test.f).unwrap();
            let roots_polished = roots_unpolished.clone();
            if SAVE_FIGURE {
                graph(
                    &format!("test_multi_root_solver_cheby_{:0>3}", id),
                    xa,
                    xb,
                    &solver,
                    &roots_unpolished,
                    &roots_polished,
                    args,
                    test.f,
                );
            }
        }
    }
}
