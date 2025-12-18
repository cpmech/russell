use crate::{EssentialBcs1d, EssentialBcs2d, NaturalBcs1d, NaturalBcs2d, Side};
use russell_lab::math::PI;

pub struct ProblemSamples;

impl ProblemSamples {
    /// 1D Problem # 01
    ///
    /// Returns `(xmin, xmax, kx, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin` and `xmax` are the domain limits
    /// * `kx` is the diffusion coefficient
    /// * `source` is the source function `f(x)`
    /// * `analytical` is the analytical solution function `ϕ(x)`
    /// * `ebcs` are the essential boundary conditions
    /// * `nbcs` are the natural boundary conditions
    ///
    /// The problem is:
    ///
    /// ```text
    ///   ∂²ϕ
    /// - ——— = x
    ///   ∂x²
    /// ```
    ///
    /// on a unit interval with homogeneous boundary conditions
    ///
    /// The analytical solution is:
    ///
    /// ```text
    ///        x - x³
    /// ϕ(x) = ——————
    ///          6
    /// ```
    pub fn d1_problem_01<'a>() -> (
        f64,
        f64,
        f64,
        EssentialBcs1d<'a>,
        NaturalBcs1d<'a>,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
    ) {
        let xmin = 0.0;
        let xmax = 1.0;
        let kx = 1.0;
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs1d::new();
        let source = Box::new(|x: f64| x);
        let analytical = Box::new(|x: f64| (x - x.powi(3)) / 6.0);
        (xmin, xmax, kx, ebcs, nbcs, source, analytical)
    }

    /// 1D Problem # 02
    ///
    /// Returns `(xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin` and `xmax` are the domain limits
    /// * `kx` is the diffusion coefficient
    /// * `beta` is the convection coefficient
    /// * `phi_inf` is the temperature of the surrounding environment
    /// * `source` is the source function `f(x)`
    /// * `analytical` is the analytical solution function `ϕ(x)`
    /// * `ebcs` are the essential boundary conditions
    /// * `nbcs` are the natural boundary conditions
    ///
    /// This problem simulates the heat conduction-confection of a 1D rod.
    ///
    /// The rod has a length of lx = 0.05 m and the conductivity coefficient
    /// is 0.01571 W/m/°C. The surrounding environment has a temperature
    /// of 20°C and the convection coefficient is 2 π W/m/°C. The left-hand
    /// side of the rod is kept at a constant temperature of 320°C, while
    /// the right-hand side is insulated. The goal is to find the temperature
    /// distribution along the rod.
    ///
    /// The Model is:
    ///
    /// ```text
    ///      ∂²ϕ
    /// - kx ——— + (ϕ - ϕ∞) β = 0
    ///      ∂x²
    /// ```
    ///
    /// where ϕ∞ = 20°C is the temperature of the surrounding environment and
    /// β = 2 π W/m/°C is the convection coefficient. The essential boundary
    /// condition is:
    ///
    /// ```text
    /// ϕ(0) = ϕₐ = 320°C
    /// ```
    ///
    /// The closed-form solution is:
    ///
    /// ```text
    ///                       cosh(m (lx - x))
    /// ϕ(x) = ϕ∞ + (ϕₐ - ϕ∞) —————————————————
    ///                           cosh(m lx)
    /// ```
    ///
    /// where m = √(β / kx)
    pub fn d1_problem_02<'a>() -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        EssentialBcs1d<'a>,
        NaturalBcs1d<'a>,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
    ) {
        let lx = 0.05;
        let kx = 0.01571;
        let beta = 2.0 * PI;
        let phi_a = 320.0;
        let phi_inf = 20.0;
        let xmin = 0.0;
        let xmax = lx;
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set(Side::Xmin, move |_| phi_a);
        let nbcs = NaturalBcs1d::new();
        let source = Box::new(|_| 0.0);
        let analytical = Box::new(move |x: f64| {
            let m = f64::sqrt(beta / kx);
            phi_inf + (phi_a - phi_inf) * f64::cosh(m * (lx - x)) / f64::cosh(m * lx)
        });
        (xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical)
    }

    /// 1D Problem # 03
    ///
    /// Returns `(xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin` and `xmax` are the domain limits
    /// * `kx` is the diffusion coefficient
    /// * `beta` is the convection coefficient
    /// * `phi_inf` is the temperature of the surrounding environment
    /// * `source` is the source function `f(x)`
    /// * `analytical` is the analytical solution function `ϕ(x)`
    /// * `ebcs` are the essential boundary conditions
    /// * `nbcs` are the natural boundary conditions
    ///
    /// This problem simulates the heat conduction-confection of a 1D rod.
    ///
    /// (With flux prescribed at the right-hand side)
    ///
    /// The rod has a length of lx = 1.0 m and the conductivity coefficient
    /// is 1.0 W/m/°C. The surrounding environment has a temperature
    /// of 0°C and the convection coefficient is 1 W/m/°C. The left-hand
    /// side of the rod is kept at a constant temperature of 2°C, while
    /// the right-hand side has a flux input of 3 W. The rod has also a heat
    /// (source) generation equal to x². The goal is to find the temperature
    /// distribution along the rod.
    ///
    /// The Model is:
    ///
    /// ```text
    ///     ∂²ϕ
    /// -kx ——— + (ϕ - ϕ∞) β = x²
    ///     ∂x²
    /// ```
    ///
    /// where ϕ∞ = 0°C is the temperature of the surrounding environment and
    /// β = 1 W/m/°C is the convection coefficient. The essential boundary
    /// condition is:
    ///
    /// ```text
    /// ϕ(0) = ϕₐ = 2°C
    /// ```
    ///
    /// The natural boundary condition is:
    ///
    /// ```text
    /// -kx ∂ϕ/∂x |_(x=1) = q̄ = -3 W
    /// ```
    ///
    /// The closed-form solution is:
    ///
    /// ```text
    ///        sinh(x)
    /// ϕ(x) = ——————— + x² + 2
    ///        cosh(1)
    /// ```
    pub fn d1_problem_03<'a>() -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        EssentialBcs1d<'a>,
        NaturalBcs1d<'a>,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
    ) {
        let lx = 1.0;
        let kx = 1.0;
        let beta = 1.0;
        let phi_a = 2.0;
        let phi_inf = 0.0;
        let xmin = 0.0;
        let xmax = lx;
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set(Side::Xmin, move |_| phi_a);
        let mut nbcs = NaturalBcs1d::new();
        nbcs.set_flux(Side::Xmax, |_| -3.0);
        let source = Box::new(|x: f64| x * x);
        let analytical = Box::new(|x: f64| {
            let d = f64::cosh(1.0);
            f64::sinh(x) / d + x * x + 2.0
        });
        (xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical)
    }

    /// 2D Problem # 01
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `u(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the Poisson equation:
    ///
    /// ```text
    /// ∂²u     ∂²u
    /// ———  +  ——— = s(x, y)
    /// ∂x²     ∂y²
    /// ```
    ///
    /// on a (1.0 × 1.0) square with homogeneous essential boundary conditions
    ///
    /// The source term is given by (for a manufactured solution):
    ///
    /// ```text
    /// s(x, y) = 2 x (y - 1) (y - 2 x + x y + 2) exp(x - y)
    /// ```
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// u(x, y) = x y (x - 1) (y - 1) exp(x - y)
    /// ```
    pub fn d2_problem_01() -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        EssentialBcs2d<'static>,
        NaturalBcs2d<'static>,
        Box<dyn Fn(f64, f64) -> f64>,
        Box<dyn Fn(f64, f64) -> f64>,
    ) {
        let (xmin, xmax, ymin, ymax) = (0.0, 1.0, 0.0, 1.0);
        let (kx, ky) = (-1.0, -1.0);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let source = Box::new(|x, y| 2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * f64::exp(x - y));
        let analytical = Box::new(|x, y| x * y * (x - 1.0) * (y - 1.0) * f64::exp(x - y));
        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }

    /// 2D Problem # 02
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `u(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the Poisson equation:
    ///
    /// ```text
    /// ∂²u     ∂²u
    /// ———  +  ——— = s(x, y)
    /// ∂x²     ∂y²
    /// ```
    ///
    /// on a (1.0 × 1.0) square with the following essential boundary conditions:
    ///
    /// * Left: u = 0
    /// * Right: u = 0
    /// * Bottom: u = 0
    /// * Top: u = sin(π x)
    ///
    /// The source term is given by (for a manufactured solution):
    ///
    /// ```text
    /// s(x, y) = -π² y sin(π x)
    /// ```
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// u(x, y) = y sin(π x)
    /// ```
    pub fn d2_problem_02() -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        EssentialBcs2d<'static>,
        NaturalBcs2d<'static>,
        Box<dyn Fn(f64, f64) -> f64>,
        Box<dyn Fn(f64, f64) -> f64>,
    ) {
        let (xmin, xmax, ymin, ymax) = (0.0, 1.0, 0.0, 1.0);
        let (kx, ky) = (-1.0, -1.0);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        ebcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        ebcs.set(Side::Ymax, |x, _| f64::sin(PI * x));
        let nbcs = NaturalBcs2d::new();
        let source = Box::new(|x, y| -PI * PI * y * f64::sin(PI * x));
        let analytical = Box::new(|x, y| y * f64::sin(PI * x));
        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }

    /// 2D Problem # 03
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `u(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the Poisson equation:
    ///
    /// ```text
    /// ∂²u     ∂²u
    /// ———  +  ——— = s(x, y)
    /// ∂x²     ∂y²
    /// ```
    ///
    /// on a (1.0 × 1.0) square with homogeneous essential boundary conditions
    ///
    /// The source term is given by (for a manufactured solution):
    ///
    /// ```text
    /// s(x, y) = 14 y³ - (16 - 12 x) y² - (-42 x² + 54 x - 2) y + 4 x³ - 16 x² + 12 x
    /// ```
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// u(x, y) = x (1 - x) y (1 - y) (1 + 2 x + 7 y)
    /// ```
    pub fn d2_problem_03() -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        EssentialBcs2d<'static>,
        NaturalBcs2d<'static>,
        Box<dyn Fn(f64, f64) -> f64>,
        Box<dyn Fn(f64, f64) -> f64>,
    ) {
        let (xmin, xmax, ymin, ymax) = (0.0, 1.0, 0.0, 1.0);
        let (kx, ky) = (-1.0, -1.0);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let source = Box::new(|x, y| {
            let (xx, yy) = (x * x, y * y);
            let (xxx, yyy) = (xx * x, yy * y);
            14.0 * yyy - (16.0 - 12.0 * x) * yy - (-42.0 * xx + 54.0 * x - 2.0) * y + 4.0 * xxx - 16.0 * xx + 12.0 * x
        });
        let analytical = Box::new(|x, y| x * (1.0 - x) * y * (1.0 - y) * (1.0 + 2.0 * x + 7.0 * y));
        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }
}
