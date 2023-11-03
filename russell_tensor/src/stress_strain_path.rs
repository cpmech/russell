use super::{LinElasticity, Mandel, Tensor2};
use crate::StrError;
use russell_lab::{mat_inverse, mat_vec_mul, vec_add, Matrix};

pub struct StressStrainPath {
    two_dim: bool,
    mandel: Mandel,
    dd: Matrix, // σ = D : ε (w.r.t Mandel basis)
    cc: Matrix, // ε = C : σ = D⁻¹ : σ (w.r.t Mandel basis)
    stresses: Vec<Tensor2>,
    strains: Vec<Tensor2>,
    strain_driven: Vec<bool>,
    dsigma: Tensor2,
    depsilon: Tensor2,
}

impl StressStrainPath {
    pub fn new(young: f64, poisson: f64, two_dim: bool) -> Self {
        let ela = LinElasticity::new(young, poisson, two_dim, false);
        let dd_tensor = ela.get_modulus();
        let mandel = dd_tensor.mandel();
        let n = dd_tensor.mandel().dim();
        let mut cc = Matrix::new(n, n);
        mat_inverse(&mut cc, &dd_tensor.mat).unwrap();
        StressStrainPath {
            two_dim,
            mandel,
            dd: dd_tensor.mat.clone(),
            cc,
            stresses: Vec::new(),
            strains: Vec::new(),
            strain_driven: Vec::new(),
            dsigma: Tensor2::new(mandel),
            depsilon: Tensor2::new(mandel),
        }
    }

    pub fn push_stress_with_oct_invariants(
        &mut self,
        sigma_m: f64,
        sigma_d: f64,
        lode: f64,
        strain_driven: bool,
    ) -> Result<&mut Self, StrError> {
        let sigma = Tensor2::new_from_oct_invariants(sigma_m, sigma_d, lode, self.two_dim)?;
        self.push_stress(sigma, strain_driven)
    }

    pub fn push_stress(&mut self, sigma: Tensor2, strain_driven: bool) -> Result<&mut Self, StrError> {
        if sigma.mandel() != self.mandel {
            return Err("mandel representation is incompatible");
        }
        let mut epsilon = Tensor2::new(self.mandel);
        self.stresses.push(sigma);
        let n = self.stresses.len();
        if n >= 2 {
            let sigma_prev = &self.stresses[n - 2];
            let sigma_curr = &self.stresses[n - 1];
            vec_add(&mut self.dsigma.vec, 1.0, &sigma_curr.vec, -1.0, &sigma_prev.vec).unwrap();
            mat_vec_mul(&mut self.depsilon.vec, 1.0, &self.cc, &self.dsigma.vec).unwrap(); // ε = C : σ
            let epsilon_prev = &mut self.strains[n - 1]; // must use "1" here because epsilon hasn't been "pushed" yet
            vec_add(&mut epsilon.vec, 1.0, &epsilon_prev.vec, 1.0, &self.depsilon.vec).unwrap();
        }
        self.strains.push(epsilon);
        self.strain_driven.push(strain_driven);
        Ok(self)
    }

    pub fn push_strain(&mut self, epsilon: Tensor2, strain_driven: bool) -> Result<&mut Self, StrError> {
        if epsilon.mandel() != self.mandel {
            return Err("mandel representation is incompatible");
        }
        let mut sigma = Tensor2::new(self.mandel);
        self.strains.push(epsilon);
        let n = self.strains.len();
        if n >= 2 {
            let epsilon_prev = &self.strains[n - 2];
            let epsilon_curr = &self.strains[n - 1];
            vec_add(&mut self.depsilon.vec, 1.0, &epsilon_curr.vec, -1.0, &epsilon_prev.vec).unwrap();
            mat_vec_mul(&mut self.dsigma.vec, 1.0, &self.dd, &self.depsilon.vec).unwrap(); // σ = D : ε
            let sigma_prev = &mut self.stresses[n - 1]; // must use "1" here because sigma hasn't been "pushed" yet
            vec_add(&mut sigma.vec, 1.0, &sigma_prev.vec, 1.0, &self.dsigma.vec).unwrap();
        }
        self.stresses.push(sigma);
        self.strain_driven.push(strain_driven);
        Ok(self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::StressStrainPath;

    #[test]
    fn new_works() {
        let young = 1500.0;
        let poisson = 0.25;
        let two_dim = true;
        let path = StressStrainPath::new(young, poisson, two_dim);
        println!("{:?}", path.stresses);
    }
}
