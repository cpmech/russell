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
    /// # Problem
    ///
    /// Solve the equation:
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
    /// # Problem
    ///
    /// Solve the equation:
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
    /// # Problem
    ///
    /// Solve the equation:
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
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    /// ∂²ϕ   ∂²ϕ
    /// ——— + ——— = s(x, y)
    /// ∂x²   ∂y²
    /// ```
    ///
    /// on a [0,1]×[0,1] square with homogeneous essential boundary conditions
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
    /// ϕ(x, y) = x y (x - 1) (y - 1) exp(x - y)
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
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    /// ∂²ϕ   ∂²ϕ
    /// ——— + ——— = s(x, y)
    /// ∂x²   ∂y²
    /// ```
    ///
    /// on a [0,1]×[0,1] square with the following essential boundary conditions:
    ///
    /// * Left: ϕ = 0
    /// * Right: ϕ = 0
    /// * Bottom: ϕ = 0
    /// * Top: ϕ = sin(π x)
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
    /// ϕ(x, y) = y sin(π x)
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
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    /// ∂²ϕ   ∂²ϕ
    /// ——— + ——— = s(x, y)
    /// ∂x²   ∂y²
    /// ```
    ///
    /// on a [0,1]×[0,1] square with homogeneous essential boundary conditions
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
    /// ϕ(x, y) = x (1 - x) y (1 - y) (1 + 2 x + 7 y)
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

    /// 2D Problem # 04
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Input
    ///
    /// * `ana_nsum` -- number of summation terms in the analytical solution.
    ///  **Warning:** For `ana_nsum > 227`, infinite values appear in the sum.
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    /// ∇²ϕ = -1
    /// ```
    ///
    /// on a [-1,1]×[-1,1] square with homogeneous boundary conditions.
    ///
    /// The analytical solution is:
    ///
    /// ```text
    ///          1 - x²   16    ∞
    /// ϕ(x,y) = —————— - ——    Σ    mₖ(x,y)
    ///            2      π³  k = 1
    ///                       k odd
    /// where
    ///           sin(aₖ) (sinh(bₖ) + sinh(cₖ))
    /// mₖ(x,y) = —————————————————————————————
    ///                   k³ sinh(k π)
    ///
    /// aₖ = k π (1 + x) / 2
    /// bₖ = k π (1 + y) / 2
    /// cₖ = k π (1 - y) / 2
    /// ```
    pub fn d2_problem_04(
        ana_nsum: usize,
    ) -> (
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
        let (xmin, xmax, ymin, ymax) = (-1.0, 1.0, -1.0, 1.0);
        let (kx, ky) = (-1.0, -1.0);
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_homogeneous();
        let nbcs = NaturalBcs2d::new();
        let source = Box::new(|_, _| -1.0);
        let analytical = Box::new(move |x, y| {
            let mut sum = 0.0;
            for k in (1..ana_nsum).step_by(2) {
                let k3 = (k * k * k) as f64;
                let kp = (k as f64) * PI;
                let ak = kp * (1.0 + x) / 2.0;
                let bk = kp * (1.0 + y) / 2.0;
                let ck = kp * (1.0 - y) / 2.0;
                let sak = f64::sin(ak);
                if sak != 0.0 {
                    sum += sak * (f64::sinh(bk) + f64::sinh(ck)) / (k3 * f64::sinh(kp));
                }
            }
            (1.0 - x * x) / 2.0 - 16.0 * sum / (PI * PI * PI)
        });
        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }

    /// 2D Problem # 05
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    ///   ∂²ϕ   ∂²ϕ
    /// - ——— - ——— = -6 x
    ///   ∂x²   ∂y²
    /// ```
    ///
    /// on a [-1,1]×[-1,1] square with the following boundary conditions:
    ///
    /// * Xmin: ϕ(-1, y) = 0
    /// * Xmax: ϕ(1, y) = 2
    /// * Ymin: wₙ = 0
    /// * Ymax: wₙ = 0
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// ϕ(x, y) = 1 + x³
    /// ```
    pub fn d2_problem_05() -> (
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
        let (xmin, xmax, ymin, ymax) = (-1.0, 1.0, -1.0, 1.0);
        let (kx, ky) = (1.0, 1.0);

        // analytical solution
        let analytical = Box::new(|x, _| 1.0 + f64::powi(x, 3));

        // essential boundary conditions
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set(Side::Xmin, |_, _| 0.0);
        ebcs.set(Side::Xmax, |_, _| 2.0);

        // natural boundary conditions
        let mut nbcs = NaturalBcs2d::new();
        nbcs.set_flux(Side::Ymin, |_, _| 0.0);
        nbcs.set_flux(Side::Ymax, |_, _| 0.0);

        // source term
        let source = Box::new(|x, _| -6.0 * x);

        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }

    /// 2D Problem # 06
    ///
    /// Returns `(xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)`, where:
    ///
    /// * `xmin`, `xmax`, `ymin`, `ymax` -- domain limits
    /// * `kx`, `ky` -- diffusion coefficients
    /// * `ebcs` -- essential boundary conditions
    /// * `nbcs` -- natural boundary conditions
    /// * `source` -- source term function `s(x, y)`
    /// * `analytical` -- analytical solution function `ϕ(x, y)`
    ///
    /// # Problem
    ///
    /// Solve the equation:
    ///
    /// ```text
    ///   ∂²ϕ   ∂²ϕ   4 tanh(1 - x + y)
    /// - ——— - ——— = —————————————————
    ///   ∂x²   ∂y²    cosh(1 - x + y)²
    /// ```
    ///
    /// on a [-1,1]×[-1,1] square with the following boundary conditions:
    ///
    /// * Xmin: ϕ(-1, y) = tanh(2+y)
    /// * Xmax: wₙ(1, y) = 1/cosh(y)²
    /// * Ymin: ϕ(x, -1) = -tanh(x)
    /// * Ymax: ϕ(x, 1) = tanh(2-x)
    ///
    /// The analytical solution is:
    ///
    /// ```text
    /// ϕ(x, y) = tanh(1 - x + y)
    /// ```
    pub fn d2_problem_06() -> (
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
        let (xmin, xmax, ymin, ymax) = (-1.0, 1.0, -1.0, 1.0);
        let (kx, ky) = (1.0, 1.0);

        // analytical solution
        let analytical = Box::new(|x, y| f64::tanh(1.0 - x + y));

        // essential boundary conditions
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set(Side::Xmin, |_, y| f64::tanh(2.0 + y));
        ebcs.set(Side::Ymin, |x, _| -f64::tanh(x));
        ebcs.set(Side::Ymax, |x, _| f64::tanh(2.0 - x));

        // natural boundary conditions
        let mut nbcs = NaturalBcs2d::new();
        nbcs.set_flux(Side::Xmax, |_, y| 1.0 / f64::powi(f64::cosh(y), 2));

        // source term
        let source = Box::new(|x, y| 4.0 * f64::tanh(1.0 - x + y) / f64::powi(f64::cosh(1.0 - x + y), 2));

        (xmin, xmax, ymin, ymax, kx, ky, ebcs, nbcs, source, analytical)
    }
}
