use russell_lab::{mat_inverse, vec_inner, Matrix, Vector};

/// Holds data related to a position in a space represented by curvilinear coordinates
pub struct Metrics {
    pub u: Vector,                   // reference coordinates {r,s,t}
    pub x: Vector,                   // physical coordinates {x,y,z}
    pub cov_g0: Vector,              // covariant basis g_0 = d{x}/dr
    pub cov_g1: Vector,              // covariant basis g_1 = d{x}/ds
    pub cov_g2: Vector,              // covariant basis g_2 = d{x}/dt
    pub cnt_g0: Vector,              // contravariant basis g_0 = dr/d{x} (gradients)
    pub cnt_g1: Vector,              // contravariant basis g_1 = ds/d{x} (gradients)
    pub cnt_g2: Vector,              // contravariant basis g_2 = dt/d{x} (gradients)
    pub cov_g_mat: Matrix,           // covariant metrics g_ij = g_i ⋅ g_j
    pub cnt_g_mat: Matrix,           // contravariant metrics g^ij = g^i ⋅ g^j
    pub det_cov_g_mat: f64,          // determinant of covariant g matrix = det(CovGmat)
    pub homogeneous: bool,           // homogeneous grid => nil second order derivatives and Christoffel symbols
    pub gamma_s: Vec<Vec<Vec<f64>>>, // [k][i][j] Christoffel coefficients of second kind (non-homogeneous)
    pub l: Vec<f64>,                 // [3] L-coefficients = sum(Γ_ij^k ⋅ g^ij) (non-homogeneous)
}

impl Metrics {
    /// Allocates new 2D metrics structure
    ///
    /// NOTE: the second order derivatives (from ddxdrr) may be None => homogeneous grid
    pub fn new_2d(
        u: &Vector,
        x: &Vector,
        dxdr: &Vector,
        dxds: &Vector,
        ddxdrr: Option<&Vector>,
        ddxdss: Option<&Vector>,
        ddxdrs: Option<&Vector>,
    ) -> Self {
        // input
        let u_copy = u.clone();
        let x_copy = x.clone();
        let cov_g0 = dxdr.clone();
        let cov_g1 = dxds.clone();
        let cov_g2 = Vector::new(0); // unused in 2D

        // covariant metrics
        let mut cov_g_mat = Matrix::new(2, 2);
        cov_g_mat.set(0, 0, vec_inner(&cov_g0, &cov_g0));
        cov_g_mat.set(1, 1, vec_inner(&cov_g1, &cov_g1));
        let g01 = vec_inner(&cov_g0, &cov_g1);
        cov_g_mat.set(0, 1, g01);
        cov_g_mat.set(1, 0, g01);

        // contravariant metrics
        let mut cnt_g_mat = Matrix::new(2, 2);
        let det_cov_g_mat = mat_inverse(&mut cnt_g_mat, &cov_g_mat).unwrap();

        // contravariant vectors
        let mut cnt_g0 = Vector::new(2);
        let mut cnt_g1 = Vector::new(2);
        let cnt_g2 = Vector::new(0); // unused in 2D

        for i in 0..2 {
            cnt_g0[i] += cnt_g_mat.get(0, 0) * cov_g0[i] + cnt_g_mat.get(0, 1) * cov_g1[i];
            cnt_g1[i] += cnt_g_mat.get(1, 0) * cov_g0[i] + cnt_g_mat.get(1, 1) * cov_g1[i];
        }

        // check if homogeneous grid
        let homogeneous = ddxdrr.is_none();
        let mut gamma_s = Vec::new();
        let mut l = Vec::new();

        if !homogeneous {
            // Christoffel vectors
            let gamma00 = ddxdrr.unwrap();
            let gamma11 = ddxdss.unwrap();
            let gamma01 = ddxdrs.unwrap();

            // Christoffel symbols of second kind
            // [k][i][j]
            gamma_s = vec![vec![vec![0.0; 2]; 2]; 2];

            gamma_s[0][0][0] = vec_inner(gamma00, &cnt_g0);
            gamma_s[0][1][1] = vec_inner(gamma11, &cnt_g0);
            gamma_s[0][0][1] = vec_inner(gamma01, &cnt_g0);
            gamma_s[0][1][0] = gamma_s[0][0][1];

            gamma_s[1][0][0] = vec_inner(gamma00, &cnt_g1);
            gamma_s[1][1][1] = vec_inner(gamma11, &cnt_g1);
            gamma_s[1][0][1] = vec_inner(gamma01, &cnt_g1);
            gamma_s[1][1][0] = gamma_s[1][0][1];

            // L-coefficients
            l = vec![0.0; 2];
            l[0] = gamma_s[0][0][0] * cnt_g_mat.get(0, 0)
                + gamma_s[0][1][1] * cnt_g_mat.get(1, 1)
                + 2.0 * gamma_s[0][0][1] * cnt_g_mat.get(0, 1);
            l[1] = gamma_s[1][0][0] * cnt_g_mat.get(0, 0)
                + gamma_s[1][1][1] * cnt_g_mat.get(1, 1)
                + 2.0 * gamma_s[1][0][1] * cnt_g_mat.get(0, 1);
        }

        Metrics {
            u: u_copy,
            x: x_copy,
            cov_g0,
            cov_g1,
            cov_g2,
            cnt_g0,
            cnt_g1,
            cnt_g2,
            cov_g_mat,
            cnt_g_mat,
            det_cov_g_mat,
            homogeneous,
            gamma_s,
            l,
        }
    }

    /// Allocates new 3D metrics structure
    ///
    /// NOTE: the second order derivatives (from ddxdrr) may be None => homogeneous grid
    pub fn new_3d(
        u: &Vector,
        x: &Vector,
        dxdr: &Vector,
        dxds: &Vector,
        dxdt: &Vector,
        ddxdrr: Option<&Vector>,
        ddxdss: Option<&Vector>,
        ddxdtt: Option<&Vector>,
        ddxdrs: Option<&Vector>,
        ddxdrt: Option<&Vector>,
        ddxdst: Option<&Vector>,
    ) -> Self {
        // input
        let u_copy = u.clone();
        let x_copy = x.clone();
        let cov_g0 = dxdr.clone();
        let cov_g1 = dxds.clone();
        let cov_g2 = dxdt.clone();

        // covariant metrics
        let mut cov_g_mat = Matrix::new(3, 3);
        cov_g_mat.set(0, 0, vec_inner(&cov_g0, &cov_g0));
        cov_g_mat.set(1, 1, vec_inner(&cov_g1, &cov_g1));
        cov_g_mat.set(2, 2, vec_inner(&cov_g2, &cov_g2));

        let g01 = vec_inner(&cov_g0, &cov_g1);
        let g12 = vec_inner(&cov_g1, &cov_g2);
        let g20 = vec_inner(&cov_g2, &cov_g0);

        cov_g_mat.set(0, 1, g01);
        cov_g_mat.set(1, 2, g12);
        cov_g_mat.set(2, 0, g20);

        cov_g_mat.set(1, 0, g01);
        cov_g_mat.set(2, 1, g12);
        cov_g_mat.set(0, 2, g20);

        // contravariant metrics
        let mut cnt_g_mat = Matrix::new(3, 3);
        let det_cov_g_mat = mat_inverse(&mut cnt_g_mat, &cov_g_mat).unwrap();

        // contravariant vectors
        let mut cnt_g0 = Vector::new(3);
        let mut cnt_g1 = Vector::new(3);
        let mut cnt_g2 = Vector::new(3);

        for i in 0..3 {
            cnt_g0[i] +=
                cnt_g_mat.get(0, 0) * cov_g0[i] + cnt_g_mat.get(0, 1) * cov_g1[i] + cnt_g_mat.get(0, 2) * cov_g2[i];
            cnt_g1[i] +=
                cnt_g_mat.get(1, 0) * cov_g0[i] + cnt_g_mat.get(1, 1) * cov_g1[i] + cnt_g_mat.get(1, 2) * cov_g2[i];
            cnt_g2[i] +=
                cnt_g_mat.get(2, 0) * cov_g0[i] + cnt_g_mat.get(2, 1) * cov_g1[i] + cnt_g_mat.get(2, 2) * cov_g2[i];
        }

        // check if homogeneous grid
        let homogeneous = ddxdrr.is_none();
        let mut gamma_s = Vec::new();
        let mut l = Vec::new();

        if !homogeneous {
            // Christoffel vectors
            let gamma00 = ddxdrr.unwrap();
            let gamma11 = ddxdss.unwrap();
            let gamma22 = ddxdtt.unwrap();
            let gamma01 = ddxdrs.unwrap();
            let gamma02 = ddxdrt.unwrap();
            let gamma12 = ddxdst.unwrap();

            // Christoffel symbols of second kind
            gamma_s = vec![vec![vec![0.0; 3]; 3]; 3];

            // k=0
            gamma_s[0][0][0] = vec_inner(gamma00, &cnt_g0);
            gamma_s[0][1][1] = vec_inner(gamma11, &cnt_g0);
            gamma_s[0][2][2] = vec_inner(gamma22, &cnt_g0);
            gamma_s[0][0][1] = vec_inner(gamma01, &cnt_g0);
            gamma_s[0][0][2] = vec_inner(gamma02, &cnt_g0);
            gamma_s[0][1][2] = vec_inner(gamma12, &cnt_g0);
            gamma_s[0][1][0] = gamma_s[0][0][1];
            gamma_s[0][2][0] = gamma_s[0][0][2];
            gamma_s[0][2][1] = gamma_s[0][1][2];

            // k=1
            gamma_s[1][0][0] = vec_inner(gamma00, &cnt_g1);
            gamma_s[1][1][1] = vec_inner(gamma11, &cnt_g1);
            gamma_s[1][2][2] = vec_inner(gamma22, &cnt_g1);
            gamma_s[1][0][1] = vec_inner(gamma01, &cnt_g1);
            gamma_s[1][0][2] = vec_inner(gamma02, &cnt_g1);
            gamma_s[1][1][2] = vec_inner(gamma12, &cnt_g1);
            gamma_s[1][1][0] = gamma_s[1][0][1];
            gamma_s[1][2][0] = gamma_s[1][0][2];
            gamma_s[1][2][1] = gamma_s[1][1][2];

            // k=2
            gamma_s[2][0][0] = vec_inner(gamma00, &cnt_g2);
            gamma_s[2][1][1] = vec_inner(gamma11, &cnt_g2);
            gamma_s[2][2][2] = vec_inner(gamma22, &cnt_g2);
            gamma_s[2][0][1] = vec_inner(gamma01, &cnt_g2);
            gamma_s[2][0][2] = vec_inner(gamma02, &cnt_g2);
            gamma_s[2][1][2] = vec_inner(gamma12, &cnt_g2);
            gamma_s[2][1][0] = gamma_s[2][0][1];
            gamma_s[2][2][0] = gamma_s[2][0][2];
            gamma_s[2][2][1] = gamma_s[2][1][2];

            // L-coefficients
            l = vec![0.0; 3];
            l[0] = gamma_s[0][0][0] * cnt_g_mat.get(0, 0)
                + gamma_s[0][1][1] * cnt_g_mat.get(1, 1)
                + gamma_s[0][2][2] * cnt_g_mat.get(2, 2)
                + 2.0 * gamma_s[0][0][1] * cnt_g_mat.get(0, 1)
                + 2.0 * gamma_s[0][0][2] * cnt_g_mat.get(0, 2)
                + 2.0 * gamma_s[0][1][2] * cnt_g_mat.get(1, 2);
            l[1] = gamma_s[1][0][0] * cnt_g_mat.get(0, 0)
                + gamma_s[1][1][1] * cnt_g_mat.get(1, 1)
                + gamma_s[1][2][2] * cnt_g_mat.get(2, 2)
                + 2.0 * gamma_s[1][0][1] * cnt_g_mat.get(0, 1)
                + 2.0 * gamma_s[1][0][2] * cnt_g_mat.get(0, 2)
                + 2.0 * gamma_s[1][1][2] * cnt_g_mat.get(1, 2);
            l[2] = gamma_s[2][0][0] * cnt_g_mat.get(0, 0)
                + gamma_s[2][1][1] * cnt_g_mat.get(1, 1)
                + gamma_s[2][2][2] * cnt_g_mat.get(2, 2)
                + 2.0 * gamma_s[2][0][1] * cnt_g_mat.get(0, 1)
                + 2.0 * gamma_s[2][0][2] * cnt_g_mat.get(0, 2)
                + 2.0 * gamma_s[2][1][2] * cnt_g_mat.get(1, 2);
        }

        Metrics {
            u: u_copy,
            x: x_copy,
            cov_g0,
            cov_g1,
            cov_g2,
            cnt_g0,
            cnt_g1,
            cnt_g2,
            cov_g_mat,
            cnt_g_mat,
            det_cov_g_mat,
            homogeneous,
            gamma_s,
            l,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Metrics;
    use russell_lab::Vector;

    #[test]
    fn new_2d_works() {
        let u = Vector::from(&[0.0, 0.0]);
        let x = Vector::from(&[0.0, 0.0]);
        let dxdr = Vector::from(&[1.0, 0.0]);
        let dxds = Vector::from(&[0.0, 1.0]);
        let ddxdrr = None;
        let ddxdss = None;
        let ddxdrs = None;

        let metrics = Metrics::new_2d(&u, &x, &dxdr, &dxds, ddxdrr, ddxdss, ddxdrs);
        assert_eq!(metrics.homogeneous, true);
        assert_eq!(metrics.det_cov_g_mat, 1.0);
    }
}
