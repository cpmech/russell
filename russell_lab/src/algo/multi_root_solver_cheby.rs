use crate::StrError;
use crate::{mat_eigen, InterpChebyshev, RootSolverBrent};
use crate::{Matrix, Vector};

/// Tolerance to avoid division by zero on the trailing Chebyshev coefficient
const TOL_EPS: f64 = 1.0e-13;

/// Tolerance to discard roots with abs(Im(root)) > tau
const TOL_TAU: f64 = 1.0e-8;

/// Tolerance to discard roots such that abs(Re(root)) > (1 + sigma)
const TOL_SIGMA: f64 = 1.0e-6;

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
pub struct MultiRootSolverCheby {
    /// Holds the polynomial degree N
    nn: usize,

    /// Holds the companion matrix A
    aa: Matrix,

    /// Holds the real part of the eigenvalues
    l_real: Vector,

    /// Holds the imaginary part of the eigenvalues
    l_imag: Vector,

    /// Holds all possible roots (dim == N)
    roots: Vector,
}

impl MultiRootSolverCheby {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `nn` -- polynomial degree N (must be ≥ 2)
    pub fn new(nn: usize) -> Result<Self, StrError> {
        // check
        if nn < 2 {
            return Err("the degree N must be ≥ 2");
        }

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
            aa,
            l_real: Vector::new(nn),
            l_imag: Vector::new(nn),
            roots: Vector::new(nn),
        })
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
    pub fn find(&mut self, interp: &InterpChebyshev) -> Result<&[f64], StrError> {
        // check
        let nn = interp.get_degree();
        if nn != self.nn {
            return Err("the interpolant must have the same degree N as the solver");
        }
        if !interp.is_ready() {
            return Err("the interpolant must have the U vector already computed");
        }

        // last expansion coefficient
        let a = interp.get_coefficients();
        let an = a[nn];
        if f64::abs(an) < TOL_EPS {
            return Err("the trailing Chebyshev coefficient vanishes; try a smaller degree N");
        }

        // last row of the companion matrix
        for k in 0..nn {
            self.aa.set(nn - 1, k, -0.5 * a[k] / an);
        }
        self.aa.add(nn - 1, nn - 2, 0.5);

        // eigenvalues
        let mut v_real = Matrix::new(nn, nn);
        let mut v_imag = Matrix::new(nn, nn);
        mat_eigen(
            &mut self.l_real,
            &mut self.l_imag,
            &mut v_real,
            &mut v_imag,
            &mut self.aa,
        )
        .unwrap();

        // roots = real eigenvalues within the interval
        let (xa, xb, dx) = interp.get_range();
        let mut nroot = 0;
        for i in 0..nn {
            if f64::abs(self.l_imag[i]) < TOL_TAU * f64::abs(self.l_real[i]) {
                if f64::abs(self.l_real[i]) <= (1.0 + TOL_SIGMA) {
                    self.roots[nroot] = (xb + xa + dx * self.l_real[i]) / 2.0;
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

/// Polishes the roots using Brent's method
pub fn polish_roots_brent<F, A>(
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
    // check
    let nr = roots_in.len();
    if nr < 1 {
        return Err("this function works with at least one root");
    }
    if roots_out.len() != roots_in.len() {
        return Err("root_in and root_out must have the same lengths");
    }

    // handle single root
    let solver = RootSolverBrent::new();
    if nr == 1 {
        let xr = roots_in[0];
        if xr < xa || xr > xb {
            return Err("a root is outside [xa, xb]");
        }
        let fa = f(xa, args)?;
        let fb = f(xb, args)?;
        if fa * fb < 0.0 {
            let (xo, _) = solver.find(xa, xb, args, &mut f)?;
            roots_out[0] = xo;
        } else {
            roots_out[0] = xr;
        }
        return Ok(());
    }

    // handle multiple roots
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
            roots_out[i] = xr;
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{polish_roots_brent, MultiRootSolverCheby};
    use crate::algo::NoArgs;
    use crate::{array_approx_eq, get_test_functions};
    use crate::{mat_approx_eq, Matrix, StrError};
    use crate::{InterpChebyshev, Vector};
    use plotpy::{Curve, Legend, Plot};

    const SAVE_FIGURE: bool = false;

    fn graph<F, A>(
        name: &str,
        interp: &InterpChebyshev,
        roots_unpolished: &[f64],
        roots_polished: &[f64],
        args: &mut A,
        mut f: F,
    ) where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let (xa, xb, _) = interp.get_range();
        let xx = Vector::linspace(xa, xb, 101).unwrap();
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
            .save(&format!("/tmp/russell_lab/{}.svg", name))
            .unwrap();
    }

    #[test]
    fn new_captures_errors() {
        let nn = 1;
        assert_eq!(MultiRootSolverCheby::new(nn).err(), Some("the degree N must be ≥ 2"));
    }

    #[test]
    fn new_works() {
        let nn = 2;
        let solver = MultiRootSolverCheby::new(nn).unwrap();
        let aa_correct = Matrix::from(&[[0.0, 1.0000], [0.0, 0.0]]);
        mat_approx_eq(&solver.aa, &aa_correct, 1e-15);
    }

    #[test]
    fn find_captures_errors() {
        let (xa, xb) = (-4.0, 4.0);
        let nn = 2;
        let interp = InterpChebyshev::new(nn, xa, xb).unwrap();
        let nn_wrong = 3;
        let mut solver = MultiRootSolverCheby::new(nn_wrong).unwrap();
        assert_eq!(
            solver.find(&interp).err(),
            Some("the interpolant must have the same degree N as the solver")
        );
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();
        assert_eq!(
            solver.find(&interp).err(),
            Some("the interpolant must have the U vector already computed")
        );
    }

    #[test]
    fn find_captures_trailing_zero_error() {
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);
        let nn = 3;
        let args = &mut 0;
        let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();
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
        let interp = InterpChebyshev::new_with_f(nn, xa, xb, args, f).unwrap();

        // find roots
        let mut solver = MultiRootSolverCheby::new(nn).unwrap();
        let roots_unpolished = Vec::from(solver.find(&interp).unwrap());
        let mut roots_polished = vec![0.0; roots_unpolished.len()];
        polish_roots_brent(&mut roots_polished, &roots_unpolished, xa, xb, args, f).unwrap();
        println!("n_roots = {}", roots_polished.len());
        println!("roots_unpolished = {:?}", roots_unpolished);
        println!("roots_polished = {:?}", roots_polished);

        // figure
        if SAVE_FIGURE {
            graph(
                "test_multi_root_solver_cheby_simple",
                &interp,
                &roots_unpolished,
                &roots_polished,
                args,
                f,
            );
        }

        // check
        array_approx_eq(&roots_polished, &[-1.0, 1.0], 1e-12);
    }

    #[test]
    fn find_works_1() {
        let nn_max = 200;
        let tol = 1e-8;
        let args = &mut 0;
        let tests = get_test_functions();
        for id in &[2, 3, 4, 5, 8] {
            let test = &tests[*id];
            if test.root1.is_some() || test.root2.is_some() || test.root3.is_some() {
                println!("\n===================================================================");
                println!("\n{}", test.name);
                let (xa, xb) = test.range;
                let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, test.f).unwrap();
                let nn = interp.get_degree();
                let mut solver = MultiRootSolverCheby::new(nn).unwrap();
                let roots_unpolished = Vec::from(solver.find(&interp).unwrap());
                let mut roots_polished = vec![0.0; roots_unpolished.len()];
                polish_roots_brent(&mut roots_polished, &roots_unpolished, xa, xb, args, test.f).unwrap();
                for xr in &roots_polished {
                    assert!((test.f)(*xr, args).unwrap() < 1e-10);
                }
                if SAVE_FIGURE {
                    graph(
                        &format!("test_multi_root_solver_cheby_{:0>3}", id),
                        &interp,
                        &roots_unpolished,
                        &roots_polished,
                        args,
                        test.f,
                    );
                }
            }
        }
    }
}
