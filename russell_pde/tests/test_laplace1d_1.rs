use russell_lab::vec_approx_eq;
use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d};

#[test]
fn test_laplace1d_1() -> Result<(), String> {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //   ∂²ϕ
    // - ——— = x
    //   ∂x²
    //
    // on a unit interval with homogeneous boundary conditions

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, 1.0, 5)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous();

    // allocate the Laplacian operator
    // (note that we have to use negative kx)
    let kx = 1.0;
    let fdm = Fdm1d::new(grid, ebcs, -kx)?;

    // solve the problem
    let a = fdm.solve(|x| x)?;

    // analytical solution
    let analytical = |x| (x - f64::powi(x, 3)) / 6.0;
    fdm.for_each_coord(|m, x| {
        println!("{}: 128 ϕ = {} ({})", m, 128.0 * a[m], 128.0 * analytical(x));
    });

    // check
    let correct = [0.0, 5.0 / 128.0, 8.0 / 128.0, 7.0 / 128.0, 0.0];
    vec_approx_eq(&a, &correct, 1e-15);
    Ok(())
}
