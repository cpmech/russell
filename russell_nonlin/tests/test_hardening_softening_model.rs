#![allow(unused)]

use russell_nonlin::StrError;

pub struct HardeningSofteningModel {
    lambda_i: f64,
    lambda_r: f64,
    alpha: f64,
    beta: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl HardeningSofteningModel {
    pub fn new(lambda_i: f64, lambda_r: f64, y_r: f64, alpha: f64, beta: f64) -> Self {
        let c1 = beta * lambda_r;
        let c2 = 1.0;
        let c3 = f64::exp(beta * y_r) - c2;
        HardeningSofteningModel {
            lambda_i,
            lambda_r,
            alpha,
            beta,
            c1,
            c2,
            c3,
        }
    }

    pub fn dydx(&self, x: f64, y: f64) -> f64 {
        // reference curve
        let c1x = self.c1 * x;
        let y_ref = if c1x >= 500.0 {
            0.0
        } else {
            -self.lambda_r * x + f64::ln(self.c3 + self.c2 * f64::exp(c1x)) / self.beta
        };
        let dydx_ref = if c1x >= 500.0 {
            0.0
        } else {
            let ec1x = f64::exp(c1x);
            -self.lambda_r + (self.c1 * self.c2 * ec1x) / (self.beta * (self.c3 + self.c2 * ec1x))
        };

        // slope (modulus)
        let dist = f64::max(0.0, y_ref - y);
        let lambda_f = dydx_ref;
        self.lambda_i + (lambda_f - self.lambda_i) * f64::exp(-self.alpha * dist)
    }

    pub fn update(&mut self, x: &mut f64, y: &mut f64, delta_x: f64) -> Result<(), StrError> {
        //
        Ok(())
    }
}

#[test]
fn test_hardening_softening_model() {
    // parameters
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 3.0;
    let beta = 5.0;

    // create model
    let mut model = HardeningSofteningModel::new(lambda_i, lambda_r, y_r, alpha, beta);

    // test dydx method
    let x = 0.0;
    let y = 0.0;
    let dydx = model.dydx(x, y);
    println!("dydx at (x={}, y={}): {}", x, y, dydx);
}
