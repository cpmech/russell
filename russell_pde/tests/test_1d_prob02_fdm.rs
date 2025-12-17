use plotpy::{linspace, Curve, Plot};
use russell_lab::{approx_eq, math::PI};
use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d, NaturalBcs1d, Side, StrError};

// This problem simulates the heat conduction-confection of a 1D rod.
//
// The rod has a length of lx = 0.05 m and the conductivity coefficient
// is 0.01571 W/m/°C. The surrounding environment has a temperature
// of 20°C and the convection coefficient is 2 π W/m/°C. The left-hand
// side of the rod is kept at a constant temperature of 320°C, while
// the right-hand side is insulated. The goal is to find the temperature
// distribution along the rod.
//
// The Model is:
//
//      ∂²ϕ
// - kx ——— + (ϕ - ϕ∞) β = 0
//      ∂x²
//
// where ϕ∞ = 20°C is the temperature of the surrounding environment and
// β = 2 π W/m/°C is the convection coefficient. The essential boundary
// condition is:
//
// ϕ(0) = ϕₐ = 320°C
//
// The closed-form solution is:
//
//                       cosh(m (lx - x))
// ϕ(x) = ϕ∞ + (ϕₐ - ϕ∞) —————————————————
//                           cosh(m lx)
//
// where m = √(β / kx)

const SAVE_FIGURE: bool = false;

struct Model {
    pub lx: f64,
    pub kx: f64,
    pub beta: f64,
    pub phi_a: f64,
    pub phi_inf: f64,
}

impl Model {
    fn analytical(&self, x: f64) -> f64 {
        let m = f64::sqrt(self.beta / self.kx);
        self.phi_inf + (self.phi_a - self.phi_inf) * f64::cosh(m * (self.lx - x)) / f64::cosh(m * self.lx)
    }
}

#[test]
fn test_1d_prob02_fdm_sps() -> Result<(), StrError> {
    // constants
    let model = Model {
        lx: 0.05,
        kx: 0.01571,
        beta: 2.0 * PI,
        phi_a: 320.0,
        phi_inf: 20.0,
    };
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, model.lx, nx)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(Side::Xmin, |_| model.phi_a);

    // natural boundary conditions
    let nbcs = NaturalBcs1d::new();

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, model.kx)?;

    // solve the problem
    let a = fdm.solve_ext(model.beta, model.phi_inf, |_| 0.0)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        // println!("{}: ϕ = {} ({})", m, a[m], model.analytical(x));
        approx_eq(a[m], model.analytical(x), 0.0155);
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
            .save("/tmp/russell_pde/test_1d_prob02_fdm_sps.svg")?;
    }
    Ok(())
}

#[test]
fn test_1d_prob02_fdm_lmm() -> Result<(), StrError> {
    // constants
    let model = Model {
        lx: 0.05,
        kx: 0.01571,
        beta: 2.0 * PI,
        phi_a: 320.0,
        phi_inf: 20.0,
    };
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(0.0, model.lx, nx)?;

    // essential boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set(Side::Xmin, |_| model.phi_a);

    // natural boundary conditions
    let nbcs = NaturalBcs1d::new();

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, model.kx)?;

    // solve the problem
    let a = fdm.solve_ext_lmm(model.beta, model.phi_inf, |_| 0.0)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        // println!("{}: ϕ = {} ({})", m, a[m], model.analytical(x));
        approx_eq(a[m], model.analytical(x), 0.0155);
    });
    Ok(())
}
