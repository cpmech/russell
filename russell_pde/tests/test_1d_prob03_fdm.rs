use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d, NaturalBcs1d, Side};

// This problem simulates the heat conduction-confection of a 1D rod.
//
// (With flux prescribed at the right-hand side)
//
// The rod has a length of lx = 1.0 m and the conductivity coefficient
// is 1.0 W/m/°C. The surrounding environment has a temperature
// of 0°C and the convection coefficient is 1 W/m/°C. The left-hand
// side of the rod is kept at a constant temperature of 2°C, while
// the right-hand side has a flux input of 3 W. The rod has also a heat
// (source) generation equal to x². The goal is to find the temperature
// distribution along the rod.
//
// The Model is:
//
//     ∂²ϕ
// -kx ——— + (ϕ - ϕ∞) β = x²
//     ∂x²
//
// where ϕ∞ = 0°C is the temperature of the surrounding environment and
// β = 1 W/m/°C is the convection coefficient. The essential boundary
// condition is:
//
// ϕ(0) = ϕₐ = 2°C
//
// The natural boundary condition is:
//
// -kx ∂ϕ/∂x |_(x=1) = q̄ = -3 W
//
// The closed-form solution is:
//
//        sinh(x)
// ϕ(x) = ——————— + x² + 2
//        cosh(1)

const SAVE_FIGURE: bool = true;

struct Model {
    pub lx: f64,
    pub kx: f64,
    pub beta: f64,
    pub phi_a: f64,
    pub phi_inf: f64,
}

impl Model {
    fn analytical(&self, x: f64) -> f64 {
        let d = f64::cosh(1.0);
        f64::sinh(x) / d + x * x + 2.0
    }
}

#[test]
fn test_1d_prob03_fdm_sps() -> Result<(), String> {
    // constants
    let model = Model {
        lx: 1.0,
        kx: 1.0,
        beta: 1.0,
        phi_a: 2.0,
        phi_inf: 0.0,
    };
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, model.lx, nx)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(Side::Xmin, |_| model.phi_a);

    // natural boundary conditions
    let mut nbcs = NaturalBcs1d::new();
    nbcs.set_flux(Side::Xmax, |_| -3.0); // negative => inflow

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, model.kx)?;

    // solve the problem
    let a = fdm.solve_ext(model.beta, model.phi_inf, |x| x * x)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], model.analytical(x), 0.000282);
    });

    // plot
    if SAVE_FIGURE {
        let mut curve_ana = Curve::new();
        let mut curve_num = Curve::new();
        curve_ana.set_label("Analytical");
        curve_num
            .set_label("Numerical")
            .set_marker_style("o")
            .set_line_style("None");
        let xx_ana = linspace(0.0, model.lx, 101);
        let uu_ana = xx_ana.iter().map(|&x| model.analytical(x)).collect::<Vec<_>>();
        let mut xx_num = vec![0.0; nx];
        fdm.for_each_coord(|i, x| {
            xx_num[i] = x;
        });
        let uu_num = a.as_data();
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_1d_prob03_fdm_sps.svg")?;
    }
    Ok(())
}

#[test]
fn test_1d_prob03_fdm_lmm() -> Result<(), String> {
    // constants
    let model = Model {
        lx: 1.0,
        kx: 1.0,
        beta: 1.0,
        phi_a: 2.0,
        phi_inf: 0.0,
    };
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, model.lx, nx)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(Side::Xmin, |_| model.phi_a);

    // natural boundary conditions
    let mut nbcs = NaturalBcs1d::new();
    nbcs.set_flux(Side::Xmax, |_| -3.0); // negative => inflow

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, model.kx)?;

    // solve the problem
    let a = fdm.solve_ext_lmm(model.beta, model.phi_inf, |x| x * x)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        approx_eq(a[m], model.analytical(x), 0.000282);
    });
    Ok(())
}
