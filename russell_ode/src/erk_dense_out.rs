use crate::StrError;
use crate::{Method, System};
use crate::{DORMAND_PRINCE_5_D, DORMAND_PRINCE_8_AD, DORMAND_PRINCE_8_CD, DORMAND_PRINCE_8_D};
use russell_lab::{Matrix, Vector};
use russell_sparse::CooMatrix;

/// Handles the dense output of explicit Runge-Kutta methods
pub(crate) struct ErkDenseOut {
    /// Holds the method
    method: Method,

    /// System dimension
    ndim: usize,

    /// A coefficients for dense output
    aad: Matrix,

    /// C coefficients for dense output
    ccd: Vector,

    /// D coefficients for dense output
    dd: Matrix,

    /// Dense output values (nstage_dense * ndim)
    d: Vec<Vector>,

    /// k values for dense output
    kd: Vec<Vector>,

    /// y values for dense output
    yd: Vector,
}

impl ErkDenseOut {
    /// Allocates a new instance
    pub(crate) fn new(method: Method, ndim: usize) -> Self {
        match method {
            Method::DoPri5 => ErkDenseOut {
                method,
                ndim,
                aad: Matrix::new(0, 0),
                ccd: Vector::new(0),
                dd: Matrix::from(&DORMAND_PRINCE_5_D),
                d: vec![Vector::new(ndim); 5],
                kd: Vec::new(),
                yd: Vector::new(0),
            },
            Method::DoPri8 => ErkDenseOut {
                method,
                ndim,
                aad: Matrix::from(&DORMAND_PRINCE_8_AD),
                ccd: Vector::from(&DORMAND_PRINCE_8_CD),
                dd: Matrix::from(&DORMAND_PRINCE_8_D),
                d: vec![Vector::new(ndim); 8],
                kd: vec![Vector::new(ndim); 3],
                yd: Vector::new(ndim),
            },
            _ => ErkDenseOut {
                method,
                ndim,
                aad: Matrix::new(0, 0),
                ccd: Vector::new(0),
                dd: Matrix::new(0, 0),
                d: Vec::new(),
                kd: Vec::new(),
                yd: Vector::new(0),
            },
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
                for m in 0..self.ndim {
                    let y_diff = w[m] - y[m];
                    let b_spl = h * k[0][m] - y_diff;
                    self.d[0][m] = y[m];
                    self.d[1][m] = y_diff;
                    self.d[2][m] = b_spl;
                    self.d[3][m] = y_diff - h * k[6][m] - b_spl;
                    self.d[4][m] = self.dd.get(0, 0) * k[0][m]
                        + self.dd.get(0, 2) * k[2][m]
                        + self.dd.get(0, 3) * k[3][m]
                        + self.dd.get(0, 4) * k[4][m]
                        + self.dd.get(0, 5) * k[5][m]
                        + self.dd.get(0, 6) * k[6][m];
                    self.d[4][m] *= h;
                }
                Ok(0)
            }
            Method::DoPri8 => {
                // first function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (self.aad.get(0, 0) * k[0][m]
                            + self.aad.get(0, 6) * k[6][m]
                            + self.aad.get(0, 7) * k[7][m]
                            + self.aad.get(0, 8) * k[8][m]
                            + self.aad.get(0, 9) * k[9][m]
                            + self.aad.get(0, 10) * k[10][m]
                            + self.aad.get(0, 11) * k[11][m]
                            + self.aad.get(0, 12) * k[11][m]);
                }
                let u = x + self.ccd[0] * h;
                (system.function)(&mut self.kd[0], u, &self.yd, args)?;

                // second function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (self.aad.get(1, 0) * k[0][m]
                            + self.aad.get(1, 5) * k[5][m]
                            + self.aad.get(1, 6) * k[6][m]
                            + self.aad.get(1, 7) * k[7][m]
                            + self.aad.get(1, 10) * k[10][m]
                            + self.aad.get(1, 11) * k[11][m]
                            + self.aad.get(1, 12) * k[11][m]
                            + self.aad.get(1, 13) * self.kd[0][m]);
                }
                let u = x + self.ccd[1] * h;
                (system.function)(&mut self.kd[1], u, &self.yd, args)?;

                // next third function evaluation
                for m in 0..self.ndim {
                    self.yd[m] = y[m]
                        + h * (self.aad.get(2, 0) * k[0][m]
                            + self.aad.get(2, 5) * k[5][m]
                            + self.aad.get(2, 6) * k[6][m]
                            + self.aad.get(2, 7) * k[7][m]
                            + self.aad.get(2, 8) * k[8][m]
                            + self.aad.get(2, 12) * k[11][m]
                            + self.aad.get(2, 13) * self.kd[0][m]
                            + self.aad.get(2, 14) * self.kd[1][m]);
                }
                let u = x + self.ccd[2] * h;
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
                        * (self.dd.get(0, 0) * k[0][m]
                            + self.dd.get(0, 5) * k[5][m]
                            + self.dd.get(0, 6) * k[6][m]
                            + self.dd.get(0, 7) * k[7][m]
                            + self.dd.get(0, 8) * k[8][m]
                            + self.dd.get(0, 9) * k[9][m]
                            + self.dd.get(0, 10) * k[10][m]
                            + self.dd.get(0, 11) * k[11][m]
                            + self.dd.get(0, 12) * k[11][m]
                            + self.dd.get(0, 13) * self.kd[0][m]
                            + self.dd.get(0, 14) * self.kd[1][m]
                            + self.dd.get(0, 15) * self.kd[2][m]);
                    self.d[5][m] = h
                        * (self.dd.get(1, 0) * k[0][m]
                            + self.dd.get(1, 5) * k[5][m]
                            + self.dd.get(1, 6) * k[6][m]
                            + self.dd.get(1, 7) * k[7][m]
                            + self.dd.get(1, 8) * k[8][m]
                            + self.dd.get(1, 9) * k[9][m]
                            + self.dd.get(1, 10) * k[10][m]
                            + self.dd.get(1, 11) * k[11][m]
                            + self.dd.get(1, 12) * k[11][m]
                            + self.dd.get(1, 13) * self.kd[0][m]
                            + self.dd.get(1, 14) * self.kd[1][m]
                            + self.dd.get(1, 15) * self.kd[2][m]);
                    self.d[6][m] = h
                        * (self.dd.get(2, 0) * k[0][m]
                            + self.dd.get(2, 5) * k[5][m]
                            + self.dd.get(2, 6) * k[6][m]
                            + self.dd.get(2, 7) * k[7][m]
                            + self.dd.get(2, 8) * k[8][m]
                            + self.dd.get(2, 9) * k[9][m]
                            + self.dd.get(2, 10) * k[10][m]
                            + self.dd.get(2, 11) * k[11][m]
                            + self.dd.get(2, 12) * k[11][m]
                            + self.dd.get(2, 13) * self.kd[0][m]
                            + self.dd.get(2, 14) * self.kd[1][m]
                            + self.dd.get(2, 15) * self.kd[2][m]);
                    self.d[7][m] = h
                        * (self.dd.get(3, 0) * k[0][m]
                            + self.dd.get(3, 5) * k[5][m]
                            + self.dd.get(3, 6) * k[6][m]
                            + self.dd.get(3, 7) * k[7][m]
                            + self.dd.get(3, 8) * k[8][m]
                            + self.dd.get(3, 9) * k[9][m]
                            + self.dd.get(3, 10) * k[10][m]
                            + self.dd.get(3, 11) * k[11][m]
                            + self.dd.get(3, 12) * k[11][m]
                            + self.dd.get(3, 13) * self.kd[0][m]
                            + self.dd.get(3, 14) * self.kd[1][m]
                            + self.dd.get(3, 15) * self.kd[2][m]);
                }
                Ok(3)
            }
            _ => Ok(0),
        }
    }

    /// Calculates the dense output
    pub(crate) fn calculate(&self, y_out: &mut Vector, x_out: f64, x: f64, h: f64) {
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
            }
            _ => (),
        }
    }
}
