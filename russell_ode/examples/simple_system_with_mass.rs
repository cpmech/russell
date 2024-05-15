use russell_lab::{vec_approx_eq, StrError, Vector};
use russell_ode::prelude::*;
use russell_sparse::CooMatrix;

fn main() -> Result<(), StrError> {
    // ODE system
    let ndim = 3;
    let jac_nnz = 4;
    let mut system = System::new(
        ndim,
        |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = -y[0] + y[1];
            f[1] = y[0] + y[1];
            f[2] = 1.0 / (1.0 + x);
            Ok(())
        },
        move |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
            jj.reset();
            jj.put(0, 0, alpha * (-1.0)).unwrap();
            jj.put(0, 1, alpha * (1.0)).unwrap();
            jj.put(1, 0, alpha * (1.0)).unwrap();
            jj.put(1, 1, alpha * (1.0)).unwrap();
            Ok(())
        },
        HasJacobian::Yes,
        Some(jac_nnz),
        None,
    );

    // mass matrix
    let mass_nnz = 5;
    system.init_mass_matrix(mass_nnz).unwrap();
    system.mass_put(0, 0, 1.0).unwrap();
    system.mass_put(0, 1, 1.0).unwrap();
    system.mass_put(1, 0, 1.0).unwrap();
    system.mass_put(1, 1, -1.0).unwrap();
    system.mass_put(2, 2, 1.0).unwrap();

    // solver
    let params = Params::new(Method::Radau5);
    let mut solver = OdeSolver::new(params, &system)?;

    // initial values
    let x = 0.0;
    let mut y = Vector::from(&[1.0, 0.0, 0.0]);

    // solve from x = 0 to x = 20
    let x1 = 20.0;
    let mut args = 0;
    solver.solve(&mut y, x, x1, None, &mut args)?;
    println!("y =\n{}", y);

    // check the results
    let y_ana = Vector::from(&[f64::cos(x1), -f64::sin(x1), f64::ln(1.0 + x1)]);
    vec_approx_eq(&y, &y_ana, 1e-3);

    // print stats
    println!("{}", solver.stats());
    Ok(())
}
