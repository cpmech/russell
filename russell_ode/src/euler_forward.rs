use crate::StrError;
use crate::{OdeSolverTrait, System, Workspace};
use russell_lab::{vec_add, vec_copy, Vector};
use russell_sparse::CooMatrix;

pub(crate) struct EulerForward<'a, F, J, A>
where
    F: Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// ODE system
    system: &'a System<F, J, A>,

    /// Vector holding the function evaluation
    ///
    /// k := f(x, y)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,
}

impl<'a, F, J, A> EulerForward<'a, F, J, A>
where
    F: Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(system: &'a System<F, J, A>) -> Self {
        let ndim = system.ndim;
        EulerForward {
            system,
            k: Vector::new(ndim),
            w: Vector::new(ndim),
        }
    }
}

impl<'a, F, J, A> OdeSolverTrait<A> for EulerForward<'a, F, J, A>
where
    F: Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Enables dense output
    fn enable_dense_output(&mut self) -> Result<(), StrError> {
        Err("dense output is not available for the FwEuler method")
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        work.bench.n_function += 1;
        (self.system.function)(&mut self.k, x, y, args)?; // k := f(x, y)
        vec_add(&mut self.w, 1.0, &y, h, &self.k).unwrap(); // w := y + h * f(x, y)
        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        _work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        _args: &mut A,
    ) -> Result<(), StrError> {
        *x += h;
        vec_copy(y, &self.w).unwrap();
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {}

    /// Computes the dense output with x-h ≤ x_out ≤ x
    fn dense_output(&self, _y_out: &mut Vector, _x_out: f64, _x: f64, _y: &Vector, _h: f64) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EulerForward;
    use crate::{no_jacobian, HasJacobian, Method, NoArgs, OdeSolverTrait, Samples, System, Workspace};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn euler_forward_works() {
        // This test relates to Table 21.2 of Kreyszig's book, page 904

        // problem
        let (system, data, mut args) = Samples::kreyszig_eq6_page902();
        let yfx = data.y_analytical.unwrap();
        let ndim = system.ndim;

        // allocate structs
        let mut solver = EulerForward::new(&system);
        let mut work = Workspace::new(Method::FwEuler);

        // check dense output availability
        assert_eq!(
            solver.enable_dense_output().err(),
            Some("dense output is not available for the FwEuler method")
        );

        // numerical approximation
        let h = 0.2;
        let mut x = data.x0;
        let mut y = data.y0.clone();
        let mut y_ana = Vector::new(ndim);
        yfx(&mut y_ana, x);
        let mut xx = vec![x];
        let mut yy_num = vec![y[0]];
        let mut yy_ana = vec![y_ana[0]];
        let mut errors = vec![f64::abs(yy_num[0] - yy_ana[0])];
        for n in 0..5 {
            solver.step(&mut work, x, &y, h, &mut args).unwrap();
            assert_eq!(work.bench.n_function, n + 1);

            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();
            xx.push(x);
            yy_num.push(y[0]);

            yfx(&mut y_ana, x);
            yy_ana.push(y_ana[0]);
            errors.push(f64::abs(yy_num.last().unwrap() - yy_ana.last().unwrap()));
        }

        // Mathematica code:
        //
        // FwEulerSingleEq[f_, x0_, y0_, x1_, h_] := Module[{x, y, nstep},
        //    x[1] = x0;
        //    y[1] = y0;
        //    nstep = IntegerPart[(x1 - x0)/h] + 1;
        //    Do[
        //     x[i + 1] = x[i] + h;
        //     y[i + 1] = y[i] + h f[x[i], y[i]];
        //     , {i, 1, nstep}];
        //    Table[{x[i], y[i]}, {i, 1, nstep}]
        //  ];
        //
        // f[x_, y_] := x + y;
        // x0 = 0;  y0 = 0;  x1 = 1;  h = 0.2;
        // xy = FwEulerSingleEq[f, x0, y0, x1, h];
        // err = Abs[#[[2]] - (Exp[#[[1]]] - #[[1]] - 1)] & /@ xy;
        //
        // Print["x = ", NumberForm[xy[[All, 1]], 20]]
        // Print["y = ", NumberForm[xy[[All, 2]], 20]]
        // Print["err = ", NumberForm[err, 20]]

        // compare with Mathematica results
        let xx_correct = &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let yy_correct = &[0.0, 0.0, 0.04000000000000001, 0.128, 0.2736000000000001, 0.48832];
        let errors_correct = &[
            0.0,
            0.0214027581601699,
            0.05182469764127042,
            0.094118800390509,
            0.1519409284924678,
            0.229961828459045,
        ];
        vec_approx_eq(&xx, xx_correct, 1e-15);
        vec_approx_eq(&yy_num, yy_correct, 1e-15);
        vec_approx_eq(&errors, errors_correct, 1e-15);
    }

    #[test]
    fn euler_forward_handles_errors() {
        let system = System::new(
            1,
            |f, _, _, _: &mut NoArgs| {
                f[0] = 1.0;
                Err("stop")
            },
            no_jacobian,
            HasJacobian::No,
            None,
            None,
        );
        let mut solver = EulerForward::new(&system);
        let mut work = Workspace::new(Method::FwEuler);
        let x = 0.0;
        let y = Vector::from(&[0.0]);
        let h = 0.1;
        let mut args = 0;
        assert_eq!(solver.step(&mut work, x, &y, h, &mut args).err(), Some("stop"));
        // call other functions just to make sure all is well
        let mut y_out = Vector::new(1);
        let x_out = 0.1;
        solver.reject(&mut work, h);
        solver.dense_output(&mut y_out, x_out, x, &y, h);
    }
}
