#![allow(unused)]

use crate::StrError;
use russell_lab::{mat_inverse, Matrix, Vector};

pub struct Metrics2d {
    /// Indicates that the base vectors and metrics do not vary with position
    homogeneous: bool,

    /// Covariant base vectors gᵢ
    pub g_cov: Vec<Vector>,

    /// Contravariant base vectors gⁱ
    pub g_ctr: Vec<Vector>,

    /// Matrix with the covariant metric tensor gᵢⱼ
    pub g_mat: Matrix,

    /// Matrix with the contravariant metric tensor gⁱʲ
    pub gg_mat: Matrix,
}

impl Metrics2d {
    /// Creates a new instance
    pub fn new(homogeneous: bool) -> Self {
        Metrics2d {
            homogeneous,
            g_cov: vec![Vector::new(2), Vector::new(2)],
            g_ctr: vec![Vector::new(2), Vector::new(2)],
            g_mat: Matrix::new(2, 2),
            gg_mat: Matrix::new(2, 2),
        }
    }

    /// Calculates the metrics at a given position and returns the determinant of the covariant matrix
    pub fn calculate(
        &mut self,
        dx_dr: &Vector,
        dx_ds: &Vector,
        d2x_dr2: Option<&Vector>,
        d2x_ds2: Option<&Vector>,
        d2x_drs: Option<&Vector>,
    ) -> Result<f64, StrError> {
        // check dimensions
        if dx_dr.dim() != 2 {
            return Err("dx_dr must have dimension 2");
        }
        if dx_ds.dim() != 2 {
            return Err("dx_ds must have dimension 2");
        }

        // covariant base vectors and metrics
        for d in 0..2 {
            self.g_cov[0][d] = dx_dr[d];
            self.g_cov[1][d] = dx_ds[d];
        }

        // covariant matrix
        for i in 0..2 {
            for j in 0..2 {
                let mut g_ij = 0.0;
                for d in 0..2 {
                    g_ij += self.g_cov[i][d] * self.g_cov[j][d];
                }
                self.g_mat.set(i, j, g_ij);
            }
        }

        // contravariant matrix and determinant of the covariant matrix
        let g = mat_inverse(&mut self.gg_mat, &self.g_mat)?;

        // contravariant base vectors
        for d in 0..2 {
            for i in 0..2 {
                self.g_ctr[i][d] = 0.0;
                for j in 0..2 {
                    self.g_ctr[i][d] += self.gg_mat.get(i, j) * self.g_cov[j][d];
                }
            }
        }

        // return the determinant of the covariant matrix
        Ok(g)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Metrics2d;
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq, Vector};

    #[test]
    fn calculate_works() {
        // Consider the mapping:
        //
        //     -1 ≤ r ≤ +1
        //     x(r) = xmin + (xmax - xmin) ⋅ (1 + r) / 2
        //     r(x) = -1 + 2⋅(x - xmin) / (xmax - xmin)
        //     dx/dr = (xmax - xmin) / 2
        //
        // On a rectangular domain, similar expressions apply for y(s) and s(y).

        // define derivatives
        let xmin = -6.0;
        let xmax = 6.0;
        let ymin = -3.0;
        let ymax = 3.0;
        let dx_dr = Vector::from(&[(xmax - xmin) / 2.0, 0.0]);
        let dx_ds = Vector::from(&[0.0, (ymax - ymin) / 2.0]);

        // calculate metrics
        let mut met = Metrics2d::new(false);
        let g = met.calculate(&dx_dr, &dx_ds, None, None, None).unwrap();

        // check [g] and [G] matrices
        println!("g = det([g]) = {}", g);
        println!("[g] =\n{}", met.g_mat);
        approx_eq(g, 36.0 * 9.0, 1e-15);
        mat_approx_eq(&met.g_mat, &[[36.0, 0.0], [0.0, 9.0]], 1e-15);
        mat_approx_eq(&met.gg_mat, &[[1.0 / 36.0, 0.0], [0.0, 1.0 / 9.0]], 1e-15);

        // check covariant and contravariant vectors
        assert_eq!(&met.g_cov[0].as_data(), &dx_dr.as_data());
        assert_eq!(&met.g_cov[1].as_data(), &dx_ds.as_data());
        vec_approx_eq(&met.g_ctr[0], &[1.0 / 6.0, 0.0], 1e-15);
        vec_approx_eq(&met.g_ctr[1], &[0.0, 1.0 / 3.0], 1e-15);
    }
}
