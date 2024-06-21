#![allow(unused)]

use crate::math::{chebyshev_tn, PI};
use crate::StrError;

/// Defines the tolerance to make sure that the range [xa, xb] is not zero
const TOL_RANGE: f64 = 1.0e-8;

/// Implements the Chebyshev interpolant and associated functions
///
/// **Note:** Only Chebyshev-Gauss-Lobatto points are considered here.
///
/// # References
///
/// 1. Canuto C, Hussaini MY, Quarteroni A, Zang TA (2006) Spectral Methods: Fundamentals in
///    Single Domains. Springer. 563p
///
/// # Warning
///
/// The grid points are sorted from +1 to -1; See the note below from Reference # 1 (page 86):
///
/// "Note that the Chebyshev quadrature points as just defined are ordered
/// from right to left. This violates our general convention that quadrature points
/// are ordered from left to right (see Sect. 2.2.3). Virtually all of the classical
/// literature on Chebyshev spectral methods uses this reversed order. Therefore,
/// in the special case of the Chebyshev quadrature points we shall adhere to the
/// ordering convention that is widely used in the literature (and implemented
/// in the available software). We realize that our resolution of this dilemma
/// imposes upon the reader the task of mentally reversing the ordering of the
/// Chebyshev nodes whenever they are used in general formulas for orthogonal
/// polynomials." (Canuto, Hussaini, Quarteroni, Zang)
pub struct InterpChebyshev {
    nn: usize,
    np: usize,
    xa: f64,
    xb: f64,
    dx: f64,
    coef: Vec<f64>,
    data: Vec<f64>,
    constant_fx: f64,
}

impl InterpChebyshev {
    pub fn new(nn: usize, xa: f64, xb: f64) -> Result<Self, StrError> {
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np = nn + 1;
        Ok(InterpChebyshev {
            nn,
            np,
            xa,
            xb,
            dx: xb - xa,
            coef: vec![0.0; np],
            data: vec![0.0; np],
            constant_fx: 0.0,
        })
    }

    pub fn calc_data<F, A>(&mut self, args: &mut A, mut f: F) -> Result<(), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if self.nn == 0 {
            self.constant_fx = f((self.xa + self.xb) / 2.0, args)?;
        } else {
            chebyshev_coefficients(&mut self.coef, &mut self.data, self.nn, self.xa, self.xb, args, &mut f);
        }
        Ok(())
    }

    pub fn new_adapt<F, A>(nn_max: usize, tol: f64, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<Self, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if nn_max > 2048 {
            return Err("the maximum degree N must be ≤ 2048");
        }
        let np_max = nn_max + 1;
        let mut work_coef = vec![0.0; np_max];
        let mut work_data = vec![0.0; np_max];
        let mut an_prev = 0.0;
        for nn in 1..=nn_max {
            chebyshev_coefficients(&mut work_coef, &mut work_data, nn, xa, xb, args, &mut f);
            let an = work_coef[nn];
            println!("N = {}, an = {}", nn, an);
            if nn > 1 && f64::abs(an_prev) < tol && f64::abs(an) < tol {
                let nn_final = nn - 2; // -2 because the last two coefficients are zero
                let mut interp = InterpChebyshev::new(nn_final, xa, xb).unwrap();
                interp.calc_data(args, f)?;
                return Ok(interp);
            }
            an_prev = an;
        }
        Err("adaptive interpolation did not converge")
    }

    pub fn eval(&self, x: f64) -> f64 {
        if self.nn == 0 {
            return self.constant_fx;
        }
        let mut sum = 0.0;
        for k in 0..self.np {
            let y = f64::max(-1.0, f64::min(1.0, (2.0 * x - self.xb - self.xa) / self.dx));
            sum += self.coef[k] * chebyshev_tn(k, y);
        }
        sum
    }
}

/// Computes the Chebyshev-Gauss-Lobatto coefficients
pub(crate) fn chebyshev_coefficients<F, A>(
    work_coef: &mut [f64],
    work_data: &mut [f64],
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
    assert!(work_coef.len() >= np);
    assert!(work_data.len() >= np);

    // data vector
    let nf = nn as f64;
    let data = &mut work_data[0..np];
    for k in 0..np {
        let kf = k as f64;
        let x = (xb + xa + (xb - xa) * f64::cos(PI * kf / nf)) / 2.0;
        data[k] = (*f)(x, args).unwrap();
    }

    // coefficients
    let coef = &mut work_coef[0..np];
    for j in 0..np {
        let jf = j as f64;
        let qj = if j == 0 || j == nn { 2.0 } else { 1.0 };
        coef[j] = 0.0;
        for k in 0..np {
            let kf = k as f64;
            let qk = if k == 0 || k == nn { 2.0 } else { 1.0 };
            coef[j] += data[k] * 2.0 * f64::cos(PI * jf * kf / nf) / (qj * qk * nf);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::InterpChebyshev;
    use crate::math::PI;
    use crate::StrError;
    use crate::{NoArgs, Vector};
    use plotpy::{Curve, Legend, Plot};

    const SAVE_FIGURE: bool = true;

    #[test]
    fn adaptive_interpolation_works() {
        let mut f = |x: f64, _: &mut NoArgs| -> Result<f64, StrError> {
            // Ok(2.0)
            // Ok(x - 0.5)
            // Ok(x * x - 1.0)
            // Ok(x * x * x - 0.5)
            // Ok(x * x * x * x - 0.5)
            // Ok(x * x * x * x * x - 0.5)
            Ok(f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x))
            // Ok(0.092834 * f64::sin(77.0001 + 19.87 * x))
            // Ok(f64::ln(2.0 * f64::cos(x / 2.0)))
        };
        let (xa, xb) = (-1.0, 1.0);
        // let (xa, xb) = (-2.34567, 12.34567);
        // let (xa, xb) = (-0.995 * PI, 0.995 * PI);

        let nn_max = 400;
        let tol = 1e-7;
        let args = &mut 0;
        let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f).unwrap();
        println!("N = {}", interp.nn);

        if SAVE_FIGURE {
            let xx = Vector::linspace(xa, xb, 201).unwrap();
            let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
            let yy_int = xx.get_mapped(|x| interp.eval(x));
            let mut curve_ana = Curve::new();
            let mut curve_int = Curve::new();
            curve_ana.set_label("analytical");
            curve_int
                .set_label("interpolated")
                .set_line_style(":")
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
                .save("/tmp/russell/test_interp_chebyshev_new_adapt.svg")
                .unwrap();
        }
    }
}
