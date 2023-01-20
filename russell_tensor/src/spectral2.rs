use crate::{vec_dyad_vec, StrError, Tensor2};
use russell_lab::{mat_eigen_sym_jacobi, Matrix, Vector};

/// Holds the spectral representation of a symmetric second-order tensor
pub struct Spectral2 {
    /// Eigenvalues
    pub lambda: Vector,

    /// Eigenprojectors
    pub projectors: Vec<Tensor2>,
}

impl Spectral2 {
    /// Returns a new instance
    pub fn new(tt: &Tensor2) -> Result<Self, StrError> {
        // check
        if !tt.is_symmetric() {
            return Err("tensor must be symmetric");
        }

        // eigenvalues and eigenvectors
        let mut a = tt.to_matrix();
        let mut l = Vector::new(3);
        let mut v = Matrix::new(3, 3);
        mat_eigen_sym_jacobi(&mut l, &mut v, &mut a)?;

        // eigenprojectors
        let zero = Tensor2::new(true, tt.is_two_dim());
        let mut projectors = vec![zero; 3];
        if tt.is_two_dim() {
            let u = Vector::from(&[v.get(0, 0), v.get(1, 0)]);
            vec_dyad_vec(&mut projectors[0], 1.0, &u, &u)?;
            panic!("TODO");
        } else {
            let u0 = Vector::from(&[v.get(0, 0), v.get(1, 0), v.get(2, 0)]);
            let u1 = Vector::from(&[v.get(0, 1), v.get(1, 1), v.get(2, 1)]);
            let u2 = Vector::from(&[v.get(0, 2), v.get(1, 2), v.get(2, 2)]);
            vec_dyad_vec(&mut projectors[0], 1.0, &u0, &u0)?;
            vec_dyad_vec(&mut projectors[1], 1.0, &u1, &u1)?;
            vec_dyad_vec(&mut projectors[2], 1.0, &u2, &u2)?;
        }

        // done
        Ok(Spectral2 { lambda: l, projectors })
    }

    /// Composes a new tensor from the eigenprojectors
    pub fn compose(&self, composed: &mut Tensor2, lambda: &Vector) -> Result<(), StrError> {
        let n = self.projectors[0].vec.dim();
        if composed.vec.dim() != n {
            return Err("composed tensor has incorrect dimension");
        }
        if lambda.dim() != 3 {
            return Err("lambda.dim must be equal to 3");
        }
        for i in 0..n {
            composed.vec[i] = lambda[0] * self.projectors[0].vec[i]
                + lambda[1] * self.projectors[1].vec[i]
                + lambda[2] * self.projectors[2].vec[i];
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Spectral2;
    use crate::Tensor2;
    use russell_chk::vec_approx_eq;
    use russell_lab::{mat_approx_eq, Matrix};

    #[test]
    fn new_and_compose_3d_work() {
        // perform spectral decomposition of symmetric matrix
        #[rustfmt::skip]
        let data = &[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ];
        let tt = Tensor2::from_matrix(data, true, false).unwrap();
        let s = Spectral2::new(&tt).unwrap();

        // check eigenvalues
        vec_approx_eq(
            s.lambda.as_data(),
            &[0.170915188827179, -0.515729471589257, 11.3448142827621],
            1e-13,
        );

        // check eigenprojectors
        let correct0 = Matrix::from(&[
            [0.34929169541608923, -0.4355596199317577, 0.19384226684174433],
            [-0.4355596199317577, 0.5431339622578344, -0.24171735309001413],
            [0.19384226684174433, -0.24171735309001413, 0.10757434232607645],
        ]);
        let correct1 = Matrix::from(&[
            [0.5431339622578346, 0.24171735309001352, -0.435559619931758],
            [0.24171735309001352, 0.10757434232607586, -0.1938422668417439],
            [-0.435559619931758, -0.1938422668417439, 0.3492916954160896],
        ]);
        let correct2 = Matrix::from(&[
            [0.10757434232607616, 0.19384226684174424, 0.24171735309001374],
            [0.19384226684174424, 0.3492916954160899, 0.43555961993175796],
            [0.24171735309001374, 0.43555961993175796, 0.5431339622578341],
        ]);
        let pp0 = s.projectors[0].to_matrix();
        let pp1 = s.projectors[1].to_matrix();
        let pp2 = s.projectors[2].to_matrix();
        mat_approx_eq(&correct0, &pp0, 1e-15);
        mat_approx_eq(&correct1, &pp1, 1e-15);
        mat_approx_eq(&correct2, &pp2, 1e-15);

        // compose
        let mut tt_new = Tensor2::new(true, false);
        s.compose(&mut tt_new, &s.lambda).unwrap();
        let a_new = tt_new.to_matrix();
        let a = Matrix::from(data);
        mat_approx_eq(&a, &a_new, 1e-14);
    }

    #[test]
    fn new_and_compose_2d_work() {
        // TODO
    }
}
