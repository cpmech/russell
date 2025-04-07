use super::{System, NoArgs};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Sym};

/// Holds a collection of nonlinear problems
pub struct Samples {}

impl Samples {
    /// Simple two-equation system
    ///
    /// Returns `(system, u_trial, u_reference, args)`
    pub fn simple_two_equations<'a>() -> (System<'a, NoArgs>, Vector, Vector, NoArgs) {
        // system
        let ndim = 2;
        let mut system = System::new(ndim, |gg: &mut Vector, _l: f64, u: &Vector, _args: &mut NoArgs| {
            gg[0] = u[0].powf(3.0) + u[1] - 1.0;
            gg[1] = -u[0] + u[1].powf(3.0) + 1.0;
            Ok(())
        })
        .unwrap();

        // function to compute Gu
        let nnz = 4;
        system
            .set_calc_ggu(
                Some(nnz),
                Sym::No,
                |ggu: &mut CooMatrix, _l: f64, u: &Vector, _args: &mut NoArgs| {
                    ggu.reset();
                    ggu.put(0, 0, 3.0 * u[0] * u[0]).unwrap();
                    ggu.put(0, 1, 1.0).unwrap();
                    ggu.put(1, 0, -1.0).unwrap();
                    ggu.put(1, 1, 3.0 * u[1] * u[1]).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // trial u vector for Newton's method
        let u_trial = Vector::from(&[0.5, 0.5]);

        // reference solution
        let u_reference = Vector::from(&[1.0, 0.0]);

        // done
        let args = 0;
        (system, u_trial, u_reference, args)
    }
}
