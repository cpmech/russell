use crate::{EssentialBcs1d, NaturalBcs1d};

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
}
