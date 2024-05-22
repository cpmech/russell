use crate::StrError;
use crate::{OdeSolverTrait, Params, System, Workspace};
use russell_lab::{vec_copy, vec_rms_scaled, vec_update, Vector};
use russell_sparse::{numerical_jacobian, LinSolver, SparseMatrix};

/// Implements the backward Euler (implicit) solver (implicit, order 1, unconditionally stable)
pub(crate) struct EulerBackward<'a, A> {
    /// Holds the parameters
    params: Params,

    /// ODE system
    system: &'a System<'a, A>,

    /// Vector holding the function evaluation
    ///
    /// k := f(x_new, y_new)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,

    /// Residual vector (right-hand side vector)
    r: Vector,

    /// Unknowns vector (the solution of the linear system)
    dy: Vector,

    /// Coefficient matrix K = h J - I
    kk: SparseMatrix,

    /// Linear solver
    solver: LinSolver<'a>,
}

impl<'a, A> EulerBackward<'a, A> {
    /// Allocates a new instance
    pub fn new(params: Params, system: &'a System<'a, A>) -> Self {
        let ndim = system.ndim;
        let jac_nnz = if params.newton.use_numerical_jacobian {
            ndim * ndim
        } else {
            system.jac_nnz
        };
        let nnz = jac_nnz + ndim; // +ndim corresponds to the diagonal I matrix
        EulerBackward {
            params,
            system,
            k: Vector::new(ndim),
            w: Vector::new(ndim),
            r: Vector::new(ndim),
            dy: Vector::new(ndim),
            kk: SparseMatrix::new_coo(ndim, ndim, nnz, system.symmetric).unwrap(),
            solver: LinSolver::new(params.newton.genie).unwrap(),
        }
    }
}

impl<'a, A> OdeSolverTrait<A> for EulerBackward<'a, A> {
    /// Enables dense output
    fn enable_dense_output(&mut self) -> Result<(), StrError> {
        Err("dense output is not available for the BwEuler method")
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // auxiliary
        let traditional_newton = !self.params.bweuler.use_modified_newton;
        let ndim = self.system.ndim;

        // trial update
        let x_new = x + h;
        let y_new = &mut self.w;
        vec_copy(y_new, &y).unwrap();

        // perform iterations
        let mut success = false;
        work.stats.n_iterations = 0;
        for _ in 0..self.params.newton.n_iteration_max {
            // stats
            work.stats.n_iterations += 1;

            // calculate k_new
            work.stats.n_function += 1;
            (self.system.function)(&mut self.k, x_new, y_new, args)?; // k := f(x_new, y_new)

            // calculate the residual and its norm
            for i in 0..ndim {
                self.r[i] = y_new[i] - y[i] - h * self.k[i];
            }
            let r_norm = vec_rms_scaled(&self.r, y, self.params.tol.abs, self.params.tol.rel);

            // check convergence
            if r_norm < self.params.tol.newton {
                success = true;
                break;
            }

            // compute K matrix (augmented Jacobian)
            if traditional_newton || work.stats.n_accepted == 0 {
                // stats
                work.stats.sw_jacobian.reset();
                work.stats.n_jacobian += 1;

                // calculate J_new := h J
                let kk = self.kk.get_coo_mut().unwrap();
                if self.params.newton.use_numerical_jacobian || self.system.jacobian.is_none() {
                    work.stats.n_function += ndim;
                    let w1 = &mut self.k; // workspace
                    let w2 = &mut self.dy; // workspace
                    numerical_jacobian(kk, h, x_new, y_new, w1, w2, args, &self.system.function)?;
                } else {
                    (self.system.jacobian.as_ref().unwrap())(kk, h, x_new, y_new, args)?;
                }

                // add diagonal entries => calculate K = h J_new - I
                for i in 0..self.system.ndim {
                    kk.put(i, i, -1.0).unwrap();
                }

                // stats
                work.stats.stop_sw_jacobian();

                // perform factorization
                work.stats.sw_factor.reset();
                work.stats.n_factor += 1;
                self.solver
                    .actual
                    .factorize(&mut self.kk, self.params.newton.lin_sol_params)?;
                work.stats.stop_sw_factor();
            }

            // solve the linear system
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            self.solver.actual.solve(&mut self.dy, &self.kk, &self.r, false)?;
            work.stats.stop_sw_lin_sol();

            // update y
            vec_update(y_new, 1.0, &self.dy).unwrap(); // y := y + δy
        }

        // check
        work.stats.update_n_iterations_max();
        if !success {
            return Err("Newton-Raphson method did not complete successfully");
        }
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

    /// Update the parameters (e.g., for sensitive analyses)
    fn update_params(&mut self, params: Params) {
        self.params = params;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EulerBackward;
    use crate::{Method, OdeSolverTrait, Params, Samples, System, Workspace};
    use russell_lab::{array_approx_eq, Vector};
    use russell_sparse::Sym;

    // Mathematica code:
    //
    // (* The code below works with 2 equations only *)
    // BwEulerTwoEqs[f_, x0_, y0_, x1_, h_] := Module[{x, y, nstep, sol},
    //    x[1] = x0;
    //    y[1] = y0;
    //    nstep = IntegerPart[(x1 - x0)/h] + 1;
    //    Do[
    //     x[i + 1] = x[i] + h;
    //     sol = NSolve[{Y1, Y2} - y[i] - h f[x[i + 1], {Y1, Y2}] == 0, {Y1, Y2}][[1]];
    //     y[i + 1] = {Y1, Y2} /. sol;
    //     , {i, 1, nstep}];
    //    Table[{x[i], y[i]}, {i, 1, nstep}]
    // ];
    //
    // f[x_, y_] := {y[[2]], -10 y[[1]] - 11 y[[2]] + 10 x + 11};
    // x0 = 0; x1 = 2.0;
    // y0 = {2, -10};
    // h = 0.4;
    //
    // xyBE = BwEulerTwoEqs[f, x0, y0, x1, h];
    // Print["x = ", NumberForm[xyBE[[All, 1]], 20]]
    // Print["y1 = ", NumberForm[xyBE[[All, 2]][[All, 1]], 20]]
    // Print["y2 = ", NumberForm[xyBE[[All, 2]][[All, 2]], 20]]
    //
    // ana = {y1 -> Function[{x}, Exp[-x] + Exp[-10 x] + x], y2 -> Function[{x}, -Exp[-x] - 10 Exp[-10 x] + 1]};
    //
    // yBE = xyBE[[All, 2]];
    // yAna = {(y1 /. ana)[#[[1]]], (y2 /. ana)[#[[1]]]} & /@ xyBE;
    // errY1 = Abs[yBE - yAna][[All, 1]];
    // errY2 = Abs[yBE - yAna][[All, 2]];
    // Print["errY1 = ", NumberForm[errY1, 20]]
    // Print["errY2 = ", NumberForm[errY2, 20]]

    const XX_MATH: [f64; 6] = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
    const YY0_MATH: [f64; 6] = [
        2.0,
        1.314285714285715,
        1.350204081632653,
        1.572431486880467,
        1.861908204914619,
        2.186254432081871,
    ];
    const YY1_MATH: [f64; 6] = [
        -10.0,
        -1.714285714285714,
        0.0897959183673469,
        0.5555685131195336,
        0.723691795085381,
        0.810865567918129,
    ];
    const ERR_Y0_MATH: [f64; 6] = [
        0.0,
        0.2256500293613408,
        0.100539654887529,
        0.07123113075591125,
        0.06001157438478888,
        0.05091914678410436,
    ];
    const ERR_Y1_MATH: [f64; 6] = [
        0.0,
        1.860809279362733,
        0.4575204912364065,
        0.1431758328447311,
        0.07441056156821646,
        0.05379912823372179,
    ];

    #[test]
    fn euler_backward_works() {
        // This test relates to Table 21.13 of Kreyszig's book, page 921

        // problem
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_ex4_page920();
        let ndim = system.ndim;

        // allocate structs
        let params = Params::new(Method::BwEuler);
        let mut solver = EulerBackward::new(params, &system);
        let mut work = Workspace::new(Method::BwEuler);

        // check dense output availability
        assert_eq!(
            solver.enable_dense_output().err(),
            Some("dense output is not available for the BwEuler method")
        );

        // numerical approximation
        let h = 0.4;
        let mut x = x0;
        let mut y = y0.clone();
        let mut xx = vec![x];
        let mut yy0_num = vec![y[0]];
        let mut yy1_num = vec![y[1]];
        let mut y_ana = Vector::new(ndim);
        y_fn_x(&mut y_ana, x, &mut args);
        let mut err_y0 = vec![f64::abs(yy0_num[0] - y_ana[0])];
        let mut err_y1 = vec![f64::abs(yy1_num[0] - y_ana[1])];
        for n in 0..5 {
            solver.step(&mut work, x, &y, h, &mut args).unwrap();
            assert_eq!(work.stats.n_iterations, 2);
            assert_eq!(work.stats.n_function, (n + 1) * 2);
            assert_eq!(work.stats.n_jacobian, (n + 1)); // already converged before calling Jacobian again

            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();
            xx.push(x);
            yy0_num.push(y[0]);
            yy1_num.push(y[1]);

            y_fn_x(&mut y_ana, x, &mut args);
            err_y0.push(f64::abs(yy0_num.last().unwrap() - y_ana[0]));
            err_y1.push(f64::abs(yy1_num.last().unwrap() - y_ana[1]));
        }

        // compare with Mathematica results
        array_approx_eq(&xx, &XX_MATH, 1e-15);
        array_approx_eq(&yy0_num, &YY0_MATH, 1e-15);
        array_approx_eq(&yy1_num, &YY1_MATH, 1e-14);
        array_approx_eq(&err_y0, &ERR_Y0_MATH, 1e-15);
        array_approx_eq(&err_y1, &ERR_Y1_MATH, 1e-14);
    }

    #[test]
    fn euler_backward_works_num_jacobian() {
        // This test relates to Table 21.13 of Kreyszig's book, page 921

        // problem
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_ex4_page920();
        let ndim = system.ndim;

        // allocate structs
        let mut params = Params::new(Method::BwEuler);
        params.newton.use_numerical_jacobian = true;
        let mut solver = EulerBackward::new(params, &system);
        let mut work = Workspace::new(Method::BwEuler);

        // numerical approximation
        let h = 0.4;
        let mut x = x0;
        let mut y = y0.clone();
        let mut xx = vec![x];
        let mut yy0_num = vec![y[0]];
        let mut yy1_num = vec![y[1]];
        let mut y_ana = Vector::new(ndim);
        y_fn_x(&mut y_ana, x, &mut args);
        let mut err_y0 = vec![f64::abs(yy0_num[0] - y_ana[0])];
        let mut err_y1 = vec![f64::abs(yy1_num[0] - y_ana[1])];
        for n in 0..5 {
            solver.step(&mut work, x, &y, h, &mut args).unwrap();
            assert_eq!(work.stats.n_iterations, 2);
            assert_eq!(work.stats.n_function, (n + 1) * 2 * ndim);
            assert_eq!(work.stats.n_jacobian, (n + 1)); // already converged before calling Jacobian again

            work.stats.n_accepted += 1; // important (must precede accept)
            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();
            xx.push(x);
            yy0_num.push(y[0]);
            yy1_num.push(y[1]);

            y_fn_x(&mut y_ana, x, &mut args);
            err_y0.push(f64::abs(yy0_num.last().unwrap() - y_ana[0]));
            err_y1.push(f64::abs(yy1_num.last().unwrap() - y_ana[1]));
        }

        // compare with Mathematica results
        array_approx_eq(&xx, &XX_MATH, 1e-15);
        array_approx_eq(&yy0_num, &YY0_MATH, 1e-7);
        array_approx_eq(&yy1_num, &YY1_MATH, 1e-6);
        array_approx_eq(&err_y0, &ERR_Y0_MATH, 1e-7);
        array_approx_eq(&err_y1, &ERR_Y1_MATH, 1e-6);
    }

    #[test]
    fn euler_backward_captures_failed_iterations() {
        let mut params = Params::new(Method::BwEuler);
        let (system, x0, y0, mut args, _) = Samples::kreyszig_ex4_page920();
        params.newton.n_iteration_max = 0;
        let mut solver = EulerBackward::new(params, &system);
        let mut work = Workspace::new(Method::BwEuler);
        assert_eq!(
            solver.step(&mut work, x0, &y0, 0.1, &mut args).err(),
            Some("Newton-Raphson method did not complete successfully")
        );
    }

    #[test]
    fn euler_backward_handles_errors() {
        struct Args {
            count_f: usize,
        }
        let mut system = System::new(1, |f, _, _, args: &mut Args| {
            f[0] = 1.0;
            args.count_f += 1;
            if args.count_f == 1 {
                Err("f: stop (count = 1)")
            } else if args.count_f == 4 {
                Err("f: stop (count = 4; num-jacobian)")
            } else {
                Ok(())
            }
        });
        system
            .set_jacobian(None, Sym::No, |jj, alpha, _x, _y, _args: &mut Args| {
                jj.reset();
                jj.put(0, 0, alpha * (0.0)).unwrap();
                Err("jj: stop")
            })
            .unwrap();
        let params = Params::new(Method::BwEuler);
        let mut solver = EulerBackward::new(params, &system);
        let mut work = Workspace::new(Method::BwEuler);
        let x = 0.0;
        let y = Vector::from(&[0.0]);
        let h = 0.1;
        let mut args = Args { count_f: 0 };
        assert_eq!(
            solver.step(&mut work, x, &y, h, &mut args).err(),
            Some("f: stop (count = 1)")
        );
        assert_eq!(solver.step(&mut work, x, &y, h, &mut args).err(), Some("jj: stop"));
        solver.params.newton.use_numerical_jacobian = true;
        assert_eq!(
            solver.step(&mut work, x, &y, h, &mut args).err(),
            Some("f: stop (count = 4; num-jacobian)")
        );
        // call other functions just to make sure all is well
        let mut y_out = Vector::new(1);
        let x_out = 0.1;
        solver.reject(&mut work, h);
        solver.dense_output(&mut y_out, x_out, x, &y, h);

        solver.update_params(params);
    }
}
