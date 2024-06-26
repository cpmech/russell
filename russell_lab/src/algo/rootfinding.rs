use crate::StrError;
use crate::{mat_eigenvalues, InterpChebyshev, TOL_RANGE};
use crate::{Matrix, Vector};

/// Implements a root finding solver using Chebyshev interpolation
///
/// This struct depends on [InterpChebyshev], an interpolant using
/// Chebyshev-Gauss-Lobatto points.
///
/// It is essential that the interpolant best approximates the
/// data/function; otherwise, not all roots can be found.
///
/// The roots are the eigenvalues of the companion matrix.
///
/// # References
///
/// 1. Boyd JP (2002) Computing zeros on a real interval through Chebyshev expansion
///    and polynomial rootfinding, SIAM Journal of Numerical Analysis, 40(5):1666-1682
/// 2. Boyd JP (2013) Finding the zeros of a univariate equation: proxy rootfinders,
///    Chebyshev interpolation, and the companion matrix, SIAM Journal of Numerical
///    Analysis, 55(2):375-396.
/// 3. Boyd JP (2014) Solving Transcendental Equations: The Chebyshev Polynomial Proxy
///    and Other Numerical Rootfinders, Perturbation Series, and Oracles, SIAM, pp460
pub struct RootFinding {
    /// Holds the tolerance to avoid division by zero with the trailing Chebyshev coefficient
    ///
    /// Default = 1e-13
    pub tol_zero_an: f64,

    /// Holds the tolerance to discard roots with imaginary part
    ///
    /// Accepts only roots such that `abs(Im(root)) < tol_rel_imag * abs(Re(root))`
    ///
    /// Default = 1e-8
    pub tol_rel_imag: f64,

    /// Holds the tolerance to discard roots outside the boundaries
    ///
    /// Accepts only roots such that `abs(Re(root)) <= 1 + tol_abs_range`
    ///
    /// Default = [TOL_RANGE] / 10.0
    ///
    /// The root will then be moved back to the lower or upper bound
    pub tol_abs_boundary: f64,

    /// Holds the tolerance to stop Newton's iterations when dx ~ 0
    ///
    /// Default = 1e-13
    pub newton_tol_zero_dx: f64,

    /// Holds the tolerance to stop Newton's iterations when f(x) ~ 0
    ///
    /// Default = 1e-13
    pub newton_tol_zero_fx: f64,

    /// Holds the maximum number of iterations for the Newton polishing
    ///
    /// Default = 15
    pub newton_max_iterations: usize,

    /// Stepsize for one-sided differences
    h_osd: f64,

    /// Stepsize for central differences
    h_cen: f64,
}

impl RootFinding {
    /// Allocates a new instance
    pub fn new() -> Self {
        RootFinding {
            tol_zero_an: 1e-13,
            tol_rel_imag: 1.0e-8,
            tol_abs_boundary: TOL_RANGE / 10.0,
            newton_tol_zero_dx: 1e-13,
            newton_tol_zero_fx: 1e-13,
            newton_max_iterations: 15,
            h_osd: f64::powf(f64::EPSILON, 1.0 / 2.0),
            h_cen: f64::powf(f64::EPSILON, 1.0 / 3.0),
        }
    }

    /// Find all roots in the interval
    ///
    /// # Input
    ///
    /// * `interp` -- The Chebyshev-Gauss-Lobatto interpolant with the data vector U
    ///    already computed. The interpolant must have the same degree N as this struct.
    ///
    /// **Warning:** It is essential that the interpolant best approximates the
    /// data/function; otherwise, not all roots can be found.
    ///
    /// # Output
    ///
    /// Returns a sorted list (from xa to xb) with the roots.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
    ///     let (xa, xb) = (-2.0, 2.0);
    ///     let args = &mut 0;
    ///
    ///     // interpolant
    ///     let nn = 2;
    ///     let mut interp = InterpChebyshev::new(nn, xa, xb)?;
    ///     interp.set_function(nn, args, f)?;
    ///
    ///     // find all roots in the interval
    ///     let mut solver = RootFinding::new();
    ///     let roots = solver.find(&interp)?;
    ///     array_approx_eq(&roots, &[-1.0, 1.0], 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn find(&self, interp: &InterpChebyshev) -> Result<Vec<f64>, StrError> {
        // check
        if !interp.is_ready() {
            return Err("the interpolant must initialized first");
        }

        // handle constant function
        let nn = interp.get_degree();
        if nn == 0 {
            return Ok(Vec::new());
        }

        // expansion coefficients
        let a = interp.get_coefficients();
        let an = a[nn];
        if f64::abs(an) < self.tol_zero_an {
            return Err("the trailing Chebyshev coefficient vanishes; try a smaller degree N");
        }

        // handle linear function
        let (xa, xb, dx) = interp.get_range();
        if nn == 1 {
            let z = -a[0] / a[1];
            if f64::abs(z) <= 1.0 + self.tol_abs_boundary {
                let root = (xb + xa + dx * z) / 2.0;
                return Ok(vec![root]);
            } else {
                return Ok(Vec::new());
            }
        }

        // companion matrix
        let mut aa = Matrix::new(nn, nn);
        aa.set(0, 1, 1.0);
        for r in 1..(nn - 1) {
            aa.set(r, r + 1, 0.5); // upper diagonal
            aa.set(r, r - 1, 0.5); // lower diagonal
        }
        for k in 0..nn {
            aa.set(nn - 1, k, -0.5 * a[k] / an);
        }
        aa.add(nn - 1, nn - 2, 0.5);

        // eigenvalues
        let mut l_real = Vector::new(nn);
        let mut l_imag = Vector::new(nn);
        mat_eigenvalues(&mut l_real, &mut l_imag, &mut aa).unwrap();

        // roots = real eigenvalues within the interval
        let mut roots = Vec::new();
        for i in 0..nn {
            if f64::abs(l_imag[i]) < self.tol_rel_imag * f64::abs(l_real[i]) {
                if f64::abs(l_real[i]) <= 1.0 + self.tol_abs_boundary {
                    let x = (xb + xa + dx * l_real[i]) / 2.0;
                    roots.push(f64::max(xa, f64::min(xb, x)));
                }
            }
        }

        // sort roots
        if roots.len() > 0 {
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        Ok(roots)
    }

    /// Polishes the roots using Newton's method
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(x * x * x * x - 1.0);
    ///     let (xa, xb) = (-2.0, 2.0);
    ///     let args = &mut 0;
    ///
    ///     // interpolant
    ///     let nn = 2;
    ///     let mut interp = InterpChebyshev::new(nn, xa, xb)?;
    ///     interp.set_function(nn, args, f)?;
    ///
    ///     // find all roots in the interval
    ///     let mut solver = RootFinding::new();
    ///     let mut roots = solver.find(&interp)?;
    ///     array_approx_eq(&roots, &[-0.5, 0.5], 1e-15); // inaccurate
    ///
    ///     // polish the roots
    ///     solver.polish_roots_newton(&mut roots, xa, xb, args, f)?;
    ///     array_approx_eq(&roots, &[-1.0, 1.0], 1e-15); // accurate
    ///     Ok(())
    /// }
    /// ```
    pub fn polish_roots_newton<F, A>(
        &self,
        roots: &mut [f64],
        xa: f64,
        xb: f64,
        args: &mut A,
        mut f: F,
    ) -> Result<(), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // check
        let nr = roots.len();
        if nr < 1 {
            return Err("at least one root is required");
        }

        // Newton's method with approximate Jacobian
        let h_cen_2 = self.h_cen * 2.0;
        for r in 0..nr {
            let mut x = roots[r];
            let mut converged = false;
            for _ in 0..self.newton_max_iterations {
                // check convergence on f(x)
                let fx = f(x, args)?;
                if f64::abs(fx) < self.newton_tol_zero_fx {
                    converged = true;
                    break;
                }

                // calculate Jacobian
                let dfdx = if x - self.h_cen <= xa {
                    // forward difference
                    (f(x + self.h_osd, args)? - f(x, args)?) / self.h_osd
                } else if x + self.h_cen >= xb {
                    // backward difference
                    (f(x, args)? - f(x - self.h_osd, args)?) / self.h_osd
                } else {
                    // central difference
                    (f(x + self.h_cen, args)? - f(x - self.h_cen, args)?) / h_cen_2
                };

                // skip zero Jacobian
                if f64::abs(dfdx) < self.newton_tol_zero_fx {
                    converged = true;
                    break;
                }

                // update x
                let dx = -f(x, args)? / dfdx;
                if f64::abs(dx) < self.newton_tol_zero_dx {
                    converged = true;
                    break;
                }
                x += dx;
            }
            if !converged {
                return Err("Newton's method did not converge");
            }
            roots[r] = x;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::RootFinding;
    use crate::algo::NoArgs;
    use crate::InterpChebyshev;
    use crate::{approx_eq, array_approx_eq, get_test_functions};

    #[allow(unused)]
    use crate::{StrError, Vector};

    #[allow(unused)]
    use plotpy::{Curve, Legend, Plot};

    /*
    fn graph<F, A>(
        name: &str,
        interp: &InterpChebyshev,
        roots_unpolished: &[f64],
        roots_polished: &[f64],
        args: &mut A,
        mut f: F,
        nstation: usize,
        fig_width: f64,
    ) where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let (xa, xb, _) = interp.get_range();
        let xx = Vector::linspace(xa, xb, nstation).unwrap();
        let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
        let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
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
            zeros_unpolished.draw(&[*root], &[interp.eval(*root).unwrap()]);
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
            .set_figure_size_points(fig_width, 500.0)
            .save(&format!("/tmp/russell_lab/{}.svg", name))
            .unwrap();
    }
    */

    #[test]
    fn find_captures_errors() {
        let (xa, xb) = (-4.0, 4.0);
        let nn = 2;
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        let solver = RootFinding::new();
        assert_eq!(
            solver.find(&interp).err(),
            Some("the interpolant must initialized first")
        );
    }

    #[test]
    fn find_captures_trailing_zero_error() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);
        let nn = 3;
        let args = &mut 0;
        let mut interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        interp.set_function(nn, args, f).unwrap();
        let solver = RootFinding::new();
        assert_eq!(
            solver.find(&interp).err(),
            Some("the trailing Chebyshev coefficient vanishes; try a smaller degree N")
        );
    }

    #[test]
    fn find_works_simple() {
        // function
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);

        // interpolant
        let nn = 2;
        let args = &mut 0;
        let mut interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        interp.set_function(nn, args, f).unwrap();

        // find roots
        let solver = RootFinding::new();
        let roots_unpolished = solver.find(&interp).unwrap();
        let mut roots_polished = roots_unpolished.clone();
        solver
            .polish_roots_newton(&mut roots_polished, xa, xb, args, f)
            .unwrap();
        println!("n_roots = {}", roots_polished.len());
        println!("roots_unpolished = {:?}", roots_unpolished);
        println!("roots_polished = {:?}", roots_polished);

        // figure
        /*
        graph(
            "test_multi_root_solver_cheby_simple",
            &interp,
            &roots_unpolished,
            &roots_polished,
            args,
            f,
            101,
            600.0,
        );
        */

        // check
        array_approx_eq(&roots_polished, &[-1.0, 1.0], 1e-14);
    }

    #[test]
    fn polish_roots_newton_captures_errors() {
        let f = |_, _: &mut NoArgs| Ok(0.0);
        let args = &mut 0;
        let _ = f(0.0, args);
        let (xa, xb) = (-1.0, 1.0);
        let mut solver = RootFinding::new();
        let mut roots = Vec::new();
        assert_eq!(
            solver.polish_roots_newton(&mut roots, xa, xb, args, f).err(),
            Some("at least one root is required")
        );
        let mut roots = [0.0];
        solver.newton_max_iterations = 0;
        assert_eq!(
            solver.polish_roots_newton(&mut roots, xa, xb, args, f).err(),
            Some("Newton's method did not converge")
        );
    }

    #[test]
    fn polish_roots_newton_works() {
        // function
        let f = |x, _: &mut NoArgs| Ok(x * x * x * x - 1.0);
        let (xa, xb) = (-2.0, 2.0);

        // interpolant
        let nn = 2;
        let args = &mut 0;
        let mut interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        interp.set_function(nn, args, f).unwrap();

        // find roots
        let solver = RootFinding::new();
        let roots_unpolished = solver.find(&interp).unwrap();
        let mut roots_polished = roots_unpolished.clone();
        solver
            .polish_roots_newton(&mut roots_polished, xa, xb, args, f)
            .unwrap();
        println!("n_roots = {}", roots_polished.len());
        println!("roots_unpolished = {:?}", roots_unpolished);
        println!("roots_polished = {:?}", roots_polished);

        // figure
        /*
        graph(
            "test_polish_roots_newton",
            &interp,
            &roots_unpolished,
            &roots_polished,
            args,
            f,
            101,
            600.0,
        );
        */

        // check
        array_approx_eq(&roots_polished, &[-1.0, 1.0], 1e-14);
    }

    #[test]
    fn find_works_with_test_functions() {
        let nn_max = 200;
        let tol = 1e-8;
        let args = &mut 0;
        let tests = get_test_functions();
        for id in &[2, 3, 4, 5, 6, 7, 8, 9, 10, 13] {
            let test = &tests[*id];
            println!("\n===================================================================");
            println!("\n{}", test.name);
            let (xa, xb) = test.range;
            let mut interp = InterpChebyshev::new(nn_max, xa, xb).unwrap();
            interp.adapt_function(tol, args, test.f).unwrap();
            let solver = RootFinding::new();
            let roots_unpolished = solver.find(&interp).unwrap();
            let mut roots_polished = roots_unpolished.clone();
            solver
                .polish_roots_newton(&mut roots_polished, xa, xb, args, test.f)
                .unwrap();
            for xr in &roots_polished {
                let fx = (test.f)(*xr, args).unwrap();
                println!("x = {}, f(x) = {:.2e}", xr, fx);
                assert!(fx < 1e-10);
                if let Some(bracket) = test.root1 {
                    if *xr >= bracket.a && *xr <= bracket.b {
                        approx_eq(*xr, bracket.xo, 1e-14);
                    }
                }
                if let Some(bracket) = test.root2 {
                    if *xr >= bracket.a && *xr <= bracket.b {
                        approx_eq(*xr, bracket.xo, 1e-14);
                    }
                }
                if let Some(bracket) = test.root3 {
                    if *xr >= bracket.a && *xr <= bracket.b {
                        approx_eq(*xr, bracket.xo, 1e-14);
                    }
                }
            }
            if *id == 9 {
                assert_eq!(roots_unpolished.len(), 93);
            }
            // figure
            /*
            let (nstation, fig_width) = if *id == 9 { (1001, 2048.0) } else { (101, 600.0) };
            graph(
                &format!("test_multi_root_solver_cheby_{:0>3}", id),
                &interp,
                &roots_unpolished,
                &roots_polished,
                args,
                test.f,
                nstation,
                fig_width,
            );
            */
        }
    }

    #[test]
    fn constant_function_works() {
        // data
        let (xa, xb) = (0.0, 1.0);
        let uu = &[0.5];

        // interpolant
        let nn_max = 10;
        let mut interp = InterpChebyshev::new(nn_max, xa, xb).unwrap();
        interp.set_data(uu).unwrap();

        // find all roots in the interval
        let solver = RootFinding::new();
        let roots = &solver.find(&interp).unwrap();
        let nroot = roots.len();
        assert_eq!(nroot, 0)
    }

    #[test]
    fn linear_function_no_roots_works() {
        // data
        let (xa, xb) = (0.0, 1.0);
        let uu = &[0.5, 3.0];

        // interpolant
        let nn_max = 10;
        let tol = 1e-8;
        let mut interp = InterpChebyshev::new(nn_max, xa, xb).unwrap();
        interp.adapt_data(tol, uu).unwrap();

        // find all roots in the interval
        let solver = RootFinding::new();
        let roots = &solver.find(&interp).unwrap();
        let nroot = roots.len();
        assert_eq!(nroot, 0)
    }

    #[test]
    fn linear_function_works() {
        // data
        let (xa, xb) = (0.0, 1.0);
        let dx = xb - xa;
        let uu = &[-7.0, -4.5, 0.5, 3.0];
        let np = uu.len(); // number of points
        let nn = np - 1; // degree
        let mut xx_dat = Vector::new(np);
        let zz = InterpChebyshev::points(nn);
        for i in 0..np {
            xx_dat[i] = (xb + xa + dx * zz[i]) / 2.0;
        }

        // interpolant
        let nn_max = 100;
        let tol = 1e-8;
        let mut interp = InterpChebyshev::new(nn_max, xa, xb).unwrap();
        interp.adapt_data(tol, uu).unwrap();

        // find all roots in the interval
        let solver = RootFinding::new();
        let roots = solver.find(&interp).unwrap();
        let nroot = roots.len();
        assert_eq!(nroot, 1);
        approx_eq(roots[0], 0.7, 1e-15);
        approx_eq(interp.eval(roots[0]).unwrap(), 0.0, 1e-15);

        // plot
        /*
        let xx = Vector::linspace(xa, xb, 201).unwrap();
        let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
        let mut curve_dat = Curve::new();
        let mut curve_int = Curve::new();
        let mut curve_xr = Curve::new();
        curve_dat.set_label("data").set_line_style("None").set_marker_style(".");
        curve_int
            .set_label(&format!("interpolated,N={}", nn))
            .set_marker_every(5);
        curve_xr
            .set_label("root")
            .set_line_style("None")
            .set_marker_style("o")
            .set_marker_void(true);
        curve_dat.draw(xx_dat.as_data(), uu.as_data());
        curve_int.draw(xx.as_data(), yy_int.as_data());
        curve_xr.draw(&roots, &vec![0.0]);
        let mut plot = Plot::new();
        let mut legend = Legend::new();
        legend.set_num_col(4);
        legend.set_outside(true);
        legend.draw();
        plot.add(&curve_int)
            .add(&curve_dat)
            .add(&curve_xr)
            .add(&legend)
            .set_cross(0.0, 0.0, "gray", "-", 1.5)
            .grid_and_labels("x", "f(x)")
            .save("/tmp/russell_lab/test_multi_root_solver_cheby_linear_function.svg")
            .unwrap();
        */
    }
}
