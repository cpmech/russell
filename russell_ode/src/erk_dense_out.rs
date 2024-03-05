use crate::StrError;
use crate::{Method, System};
use crate::{DORMAND_PRINCE_5_D, DORMAND_PRINCE_8_AD, DORMAND_PRINCE_8_CD, DORMAND_PRINCE_8_D};
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Handles the dense output of explicit Runge-Kutta methods
pub(crate) struct ErkDenseOut {
    /// Holds the method
    method: Method,

    /// System dimension
    ndim: usize,

    /// Dense output values (nstage_dense * ndim)
    d: Vec<Vector>,

    /// k values for dense output
    kd: Vec<Vector>,

    /// y values for dense output
    yd: Vector,
}

impl ErkDenseOut {
    /// Allocates a new instance
    pub(crate) fn new(method: Method, ndim: usize) -> Result<Self, StrError> {
        match method {
            Method::Radau5 => Err("INTERNAL ERROR: cannot use Radau5 with ErkDenseOut"),
            Method::BwEuler => Err("INTERNAL ERROR: cannot use BwEuler with ErkDenseOut"),
            Method::FwEuler => Err("INTERNAL ERROR: cannot use FwEuler with ErkDenseOut"),
            Method::Rk2 => Err("dense output is not available for the Rk2 method"),
            Method::Rk3 => Err("dense output is not available for the Rk3 method"),
            Method::Heun3 => Err("dense output is not available for the Heun3 method"),
            Method::Rk4 => Err("dense output is not available for the Rk4 method"),
            Method::Rk4alt => Err("dense output is not available for the Rk4alt method"),
            Method::MdEuler => Err("dense output is not available for the MdEuler method"),
            Method::Merson4 => Err("dense output is not available for the Merson4 method"),
            Method::Zonneveld4 => Err("dense output is not available for the Zonneveld4 method"),
            Method::Fehlberg4 => Err("dense output is not available for the Fehlberg4 method"),
            Method::DoPri5 => Ok(ErkDenseOut {
                method,
                ndim,
                d: vec![Vector::new(ndim); 5],
                kd: Vec::new(),
                yd: Vector::new(0),
            }),
            Method::Verner6 => Err("dense output is not available for the Verner6 method"),
            Method::Fehlberg7 => Err("dense output is not available for the Fehlberg7 method"),
            Method::DoPri8 => Ok(ErkDenseOut {
                method,
                ndim,
                d: vec![Vector::new(ndim); 8],
                kd: vec![Vector::new(ndim); 3],
                yd: Vector::new(ndim),
            }),
        }
    }

    /// Updates the data and returns the number of function evaluations
    pub(crate) fn update<'a, F, J, A>(
        &mut self,
        system: &mut System<'a, F, J, A>,
        x: f64,
        y: &Vector,
        h: f64,
        w: &Vector,
        k: &Vec<Vector>,
        args: &mut A,
    ) -> Result<usize, StrError>
    where
        F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
        J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
    {
        match self.method {
            Method::DoPri5 => {
                let dd = &DORMAND_PRINCE_5_D;
                for m in 0..self.ndim {
                    let y_diff = w[m] - y[m];
                    let b_spl = h * k[0][m] - y_diff;
                    self.d[0][m] = y[m];
                    self.d[1][m] = y_diff;
                    self.d[2][m] = b_spl;
                    self.d[3][m] = y_diff - h * k[6][m] - b_spl;
                    self.d[4][m] = dd[0][0] * k[0][m]
                        + dd[0][2] * k[2][m]
                        + dd[0][3] * k[3][m]
                        + dd[0][4] * k[4][m]
                        + dd[0][5] * k[5][m]
                        + dd[0][6] * k[6][m];
                    self.d[4][m] *= h;
                }
                Ok(0)
            }
            Method::DoPri8 => {
                let aad = &DORMAND_PRINCE_8_AD;
                let ccd = &DORMAND_PRINCE_8_CD;
                let dd = &DORMAND_PRINCE_8_D;

                // first function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (aad[0][0] * k[0][m]
                            + aad[0][6] * k[6][m]
                            + aad[0][7] * k[7][m]
                            + aad[0][8] * k[8][m]
                            + aad[0][9] * k[9][m]
                            + aad[0][10] * k[10][m]
                            + aad[0][11] * k[11][m]
                            + aad[0][12] * k[11][m]);
                }
                let u = x + ccd[0] * h;
                (system.function)(&mut self.kd[0], u, &self.yd, args)?;

                // second function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (aad[1][0] * k[0][m]
                            + aad[1][5] * k[5][m]
                            + aad[1][6] * k[6][m]
                            + aad[1][7] * k[7][m]
                            + aad[1][10] * k[10][m]
                            + aad[1][11] * k[11][m]
                            + aad[1][12] * k[11][m]
                            + aad[1][13] * self.kd[0][m]);
                }
                let u = x + ccd[1] * h;
                (system.function)(&mut self.kd[1], u, &self.yd, args)?;

                // next third function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (aad[2][0] * k[0][m]
                            + aad[2][5] * k[5][m]
                            + aad[2][6] * k[6][m]
                            + aad[2][7] * k[7][m]
                            + aad[2][8] * k[8][m]
                            + aad[2][12] * k[11][m]
                            + aad[2][13] * self.kd[0][m]
                            + aad[2][14] * self.kd[1][m]);
                }
                let u = x + ccd[2] * h;
                (system.function)(&mut self.kd[2], u, &self.yd, args)?;

                // final results
                for m in 0..self.ndim {
                    let y_diff = w[m] - y[m];
                    let b_spl = h * k[0][m] - y_diff;
                    self.d[0][m] = y[m];
                    self.d[1][m] = y_diff;
                    self.d[2][m] = b_spl;
                    self.d[3][m] = y_diff - h * k[11][m] - b_spl;
                    self.d[4][m] = h
                        * (dd[0][0] * k[0][m]
                            + dd[0][5] * k[5][m]
                            + dd[0][6] * k[6][m]
                            + dd[0][7] * k[7][m]
                            + dd[0][8] * k[8][m]
                            + dd[0][9] * k[9][m]
                            + dd[0][10] * k[10][m]
                            + dd[0][11] * k[11][m]
                            + dd[0][12] * k[11][m]
                            + dd[0][13] * self.kd[0][m]
                            + dd[0][14] * self.kd[1][m]
                            + dd[0][15] * self.kd[2][m]);
                    self.d[5][m] = h
                        * (dd[1][0] * k[0][m]
                            + dd[1][5] * k[5][m]
                            + dd[1][6] * k[6][m]
                            + dd[1][7] * k[7][m]
                            + dd[1][8] * k[8][m]
                            + dd[1][9] * k[9][m]
                            + dd[1][10] * k[10][m]
                            + dd[1][11] * k[11][m]
                            + dd[1][12] * k[11][m]
                            + dd[1][13] * self.kd[0][m]
                            + dd[1][14] * self.kd[1][m]
                            + dd[1][15] * self.kd[2][m]);
                    self.d[6][m] = h
                        * (dd[2][0] * k[0][m]
                            + dd[2][5] * k[5][m]
                            + dd[2][6] * k[6][m]
                            + dd[2][7] * k[7][m]
                            + dd[2][8] * k[8][m]
                            + dd[2][9] * k[9][m]
                            + dd[2][10] * k[10][m]
                            + dd[2][11] * k[11][m]
                            + dd[2][12] * k[11][m]
                            + dd[2][13] * self.kd[0][m]
                            + dd[2][14] * self.kd[1][m]
                            + dd[2][15] * self.kd[2][m]);
                    self.d[7][m] = h
                        * (dd[3][0] * k[0][m]
                            + dd[3][5] * k[5][m]
                            + dd[3][6] * k[6][m]
                            + dd[3][7] * k[7][m]
                            + dd[3][8] * k[8][m]
                            + dd[3][9] * k[9][m]
                            + dd[3][10] * k[10][m]
                            + dd[3][11] * k[11][m]
                            + dd[3][12] * k[11][m]
                            + dd[3][13] * self.kd[0][m]
                            + dd[3][14] * self.kd[1][m]
                            + dd[3][15] * self.kd[2][m]);
                }
                Ok(3)
            }
            _ => Err("INTERNAL ERROR: dense output is not available for this method"),
        }
    }

    /// Calculates the dense output
    pub(crate) fn calculate(&self, y_out: &mut Vector, x_out: f64, x: f64, h: f64) -> Result<(), StrError> {
        match self.method {
            Method::DoPri5 => {
                let x_prev = x - h;
                let theta = (x_out - x_prev) / h;
                let u_theta = 1.0 - theta;
                for m in 0..self.ndim {
                    y_out[m] = self.d[0][m]
                        + theta
                            * (self.d[1][m]
                                + u_theta * (self.d[2][m] + theta * (self.d[3][m] + u_theta * self.d[4][m])));
                }
                Ok(())
            }
            Method::DoPri8 => {
                let x_prev = x - h;
                let theta = (x_out - x_prev) / h;
                let u_theta = 1.0 - theta;
                for m in 0..self.ndim {
                    let par = self.d[4][m] + theta * (self.d[5][m] + u_theta * (self.d[6][m] + theta * self.d[7][m]));
                    y_out[m] = self.d[0][m]
                        + theta * (self.d[1][m] + u_theta * (self.d[2][m] + theta * (self.d[3][m] + u_theta * par)));
                }
                Ok(())
            }
            _ => Err("INTERNAL ERROR: dense output is not available for this method"),
        }
    }
}
