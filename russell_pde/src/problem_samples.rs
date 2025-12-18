use crate::{EssentialBcs1d, NaturalBcs1d, Side};
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
}
