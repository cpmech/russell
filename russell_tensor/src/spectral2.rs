use super::{vec_dyad_vec, Mandel, Tensor2, SQRT_2, SQRT_3, SQRT_6};
use crate::StrError;
use russell_lab::{mat_eigen_sym_jacobi, Matrix, Vector};

/// Holds the spectral representation of a symmetric second-order tensor
pub struct Spectral2 {
    /// The mandel representation
    mandel: Mandel,

    /// Holds the eigenvalues; dim = 3
    pub lambda: Vector,

    /// Holds the eigenprojectors; set of 3 symmetric Tensor2 (dim = 6 or 4)
    pub projectors: Vec<Tensor2>,
}

impl Spectral2 {
    /// Returns a new instance
    ///
    /// **Note:** Must call [Spectral2::compose] to calculate `lambda` and `projectors`.
    pub fn new(two_dim: bool) -> Self {
        let mandel = if two_dim {
            Mandel::Symmetric2D
        } else {
            Mandel::Symmetric
        };
        Spectral2 {
            mandel,
            lambda: Vector::new(3),
            projectors: vec![Tensor2::new(mandel), Tensor2::new(mandel), Tensor2::new(mandel)],
        }
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor
    ///
    /// # Results
    ///
    /// The results are available in [Spectral2::lambda] and [Spectral2::projectors].
    ///
    /// TODO: Rewrite this function to avoid temporary memory allocation
    pub fn decompose(&mut self, tt: &Tensor2) -> Result<(), StrError> {
        if tt.mandel() != self.mandel {
            return Err("the mandel representation is incompatible");
        }
        let dim = tt.vec.dim();
        if dim == 4 {
            // eigenvalues and eigenvectors
            let (t22, mut a) = tt.to_matrix_2d();
            let mut l = Vector::new(2);
            let mut v = Matrix::new(2, 2);
            mat_eigen_sym_jacobi(&mut l, &mut v, &mut a)?;
            self.lambda[0] = l[0];
            self.lambda[1] = l[1];
            self.lambda[2] = t22;

            // extract eigenvectors
            let u0 = Vector::from(&[v.get(0, 0), v.get(1, 0)]);
            let u1 = Vector::from(&[v.get(0, 1), v.get(1, 1)]);

            // compute eigenprojectors
            vec_dyad_vec(&mut self.projectors[0], 1.0, &u0, &u0).unwrap();
            vec_dyad_vec(&mut self.projectors[1], 1.0, &u1, &u1).unwrap();
            self.projectors[2].clear();
            self.projectors[2].vec[2] = 1.0;
        } else {
            // eigenvalues and eigenvectors
            let mut a = tt.to_matrix();
            let mut v = Matrix::new(3, 3);
            mat_eigen_sym_jacobi(&mut self.lambda, &mut v, &mut a)?;

            // extract eigenvectors
            let u0 = Vector::from(&[v.get(0, 0), v.get(1, 0), v.get(2, 0)]);
            let u1 = Vector::from(&[v.get(0, 1), v.get(1, 1), v.get(2, 1)]);
            let u2 = Vector::from(&[v.get(0, 2), v.get(1, 2), v.get(2, 2)]);

            // compute eigenprojectors
            vec_dyad_vec(&mut self.projectors[0], 1.0, &u0, &u0).unwrap();
            vec_dyad_vec(&mut self.projectors[1], 1.0, &u1, &u1).unwrap();
            vec_dyad_vec(&mut self.projectors[2], 1.0, &u2, &u2).unwrap();
        }
        Ok(())
    }

    /// Composes a new tensor from the eigenprojectors and diagonal values (lambda)
    pub fn compose(&self, composed: &mut Tensor2, lambda: &Vector) -> Result<(), StrError> {
        if composed.mandel() != self.mandel {
            return Err("the mandel representation is incompatible");
        }
        if lambda.dim() != 3 {
            return Err("lambda.dim must be equal to 3");
        }
        let n = self.projectors[0].vec.dim();
        for i in 0..n {
            composed.vec[i] = lambda[0] * self.projectors[0].vec[i]
                + lambda[1] * self.projectors[1].vec[i]
                + lambda[2] * self.projectors[2].vec[i];
        }
        Ok(())
    }

    /// Calculates the octahedral basis on the principal values space
    ///
    /// Returns `(λ_star_1, λ_star_2, λ_star_3)`
    pub fn octahedral_basis(&self) -> (f64, f64, f64) {
        let (s1, s2, s3) = (self.lambda[0], self.lambda[1], self.lambda[2]);
        let ls1 = (2.0 * s1 - s2 - s3) / SQRT_6;
        let ls2 = (s1 + s2 + s3) / SQRT_3;
        let ls3 = (s3 - s2) / SQRT_2;
        (ls1, ls2, ls3)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Spectral2;
    use crate::{Mandel, SampleTensor2, SamplesTensor2, Tensor2, SQRT_3, SQRT_3_BY_2};
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq, Matrix, Vector};

    #[test]
    fn decompose_captures_errors() {
        let mut spec = Spectral2::new(false);
        let tt = Tensor2::new(Mandel::General);
        assert_eq!(
            spec.decompose(&tt).err(),
            Some("the mandel representation is incompatible")
        );
    }

    #[test]
    fn compose_capture_errors() {
        let spec = Spectral2::new(false);
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        assert_eq!(
            spec.compose(&mut tt, &spec.lambda).err(),
            Some("the mandel representation is incompatible")
        );
        let mut tt = Tensor2::new(Mandel::Symmetric);
        let lambda = Vector::new(1);
        assert_eq!(
            spec.compose(&mut tt, &lambda).err(),
            Some("lambda.dim must be equal to 3")
        );
    }

    fn check(
        spec: &mut Spectral2,
        sample: &SampleTensor2,
        tol_lambda: f64,
        tol_proj: f64,
        tol_spectral: f64,
        verbose: bool,
    ) {
        if let Some(correct_lambda) = sample.eigenvalues {
            if let Some(correct_projectors) = sample.eigenprojectors {
                // perform spectral decomposition of symmetric matrix
                let mandel = spec.projectors[0].mandel();
                let tt = Tensor2::from_matrix(&sample.matrix, mandel).unwrap();
                spec.decompose(&tt).unwrap();

                // print results
                if verbose {
                    println!("a =\n{}", tt.to_matrix());
                    println!("λ = {}, {}, {}", spec.lambda[0], spec.lambda[1], spec.lambda[2]);
                    println!("P0 =\n{}", spec.projectors[0].to_matrix());
                    println!("P1 =\n{}", spec.projectors[1].to_matrix());
                    println!("P2 =\n{}", spec.projectors[2].to_matrix());
                }

                // check eigenvalues
                vec_approx_eq(spec.lambda.as_data(), &correct_lambda, tol_lambda);

                // check eigenprojectors
                let pp0 = spec.projectors[0].to_matrix();
                let pp1 = spec.projectors[1].to_matrix();
                let pp2 = spec.projectors[2].to_matrix();
                let correct0 = Matrix::from(&correct_projectors[0]);
                let correct1 = Matrix::from(&correct_projectors[1]);
                let correct2 = Matrix::from(&correct_projectors[2]);
                mat_approx_eq(&correct0, &pp0, tol_proj);
                mat_approx_eq(&correct1, &pp1, tol_proj);
                mat_approx_eq(&correct2, &pp2, tol_proj);

                // compose
                let mut tt_new = Tensor2::new(mandel);
                spec.compose(&mut tt_new, &spec.lambda).unwrap();
                let a_new = tt_new.to_matrix();
                let a = Matrix::from(&sample.matrix);
                mat_approx_eq(&a, &a_new, tol_spectral);
            }
        };
    }

    #[test]
    fn decompose_and_compose_work_3d() {
        let mut spec = Spectral2::new(false);
        check(&mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14, false);
        check(&mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14, false);
    }

    #[test]
    fn decompose_and_compose_work_2d() {
        let mut spec = Spectral2::new(true);
        check(&mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15, false);
        check(&mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15, false);
    }

    #[test]
    fn invariants_octahedral_and_octahedral_basis_work() {
        // the following data corresponds to sigma_m = 1 and sigma_d = 3
        #[rustfmt::skip]
        let principal_stresses_and_lode = [
            ( 3.0          ,  0.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  3.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  0.0          ,  3.0          ,  1.0 ),
            ( 1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0 + SQRT_3 ,  1.0          ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0          ,  1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0          ,  1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0          ,  1.0 + SQRT_3 ,  0.0 ),
            ( 2.0          , -1.0          ,  2.0          , -1.0 ),
            ( 2.0          ,  2.0          , -1.0          , -1.0 ),
            (-1.0          ,  2.0          ,  2.0          , -1.0 ),
        ];
        let two_dim = true;
        let mut spec = Spectral2::new(two_dim);
        let mut tt = Tensor2::new_sym(two_dim);
        for (sigma_1, sigma_2, sigma_3, lode_correct) in &principal_stresses_and_lode {
            tt.vec[0] = *sigma_1;
            tt.vec[1] = *sigma_2;
            tt.vec[2] = *sigma_3;
            spec.decompose(&tt).unwrap();
            let (ls1, ls2, ls3) = spec.octahedral_basis();
            let radius = f64::sqrt(ls3 * ls3 + ls1 * ls1);
            let distance = ls2;
            approx_eq(distance / SQRT_3, 1.0, 1e-15);
            approx_eq(radius * SQRT_3_BY_2, 3.0, 1e-15);
            if radius > 0.0 {
                let cos_theta = ls1 / radius;
                let lode = 4.0 * f64::powf(cos_theta, 3.0) - 3.0 * cos_theta;
                approx_eq(lode, *lode_correct, 1e-15);
            }
        }
    }
}
