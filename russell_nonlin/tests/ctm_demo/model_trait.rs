pub trait ModelTrait {
    /// Calculates dy/dx = f(x,y)
    fn calc_f(&self, x: f64, y: f64) -> f64;

    /// Calculates L = ∂f/∂x
    fn calc_ll(&self, x: f64, y: f64) -> f64;

    /// Calculates J = ∂f/∂y
    fn calc_jj(&self, x: f64, y: f64) -> f64;
}
