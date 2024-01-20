#![allow(unused, non_snake_case)]

use russell_lab::Vector;

use crate::OdeSolParams;

pub struct Output<'a> {
    ndim: usize,
    conf: &'a OdeSolParams,
    stepNmax: usize,
    denseNmax: usize,
    StepRS: Vec<f64>,
    StepH: Vec<f64>,
    StepX: Vec<f64>,
    StepY: Vec<Vector>,
    denseS: Vec<usize>,
    DenseX: Vec<f64>,
    DenseY: Vec<Vector>,
    yout: Vec<f64>,
    StepIdx: usize,
    DenseIdx: usize,
    xout: f64,
}

impl<'a> Output<'a> {
    pub fn new(ndim: usize, conf: &'a OdeSolParams) -> Output {
        let mut o = Output {
            ndim,
            conf,
            stepNmax: 0,
            denseNmax: 0,
            StepRS: vec![],
            StepH: vec![],
            StepX: vec![],
            StepY: vec![],
            denseS: vec![],
            DenseX: vec![],
            DenseY: vec![],
            yout: vec![],
            StepIdx: 0,
            DenseIdx: 0,
            xout: 0.0,
        };
        if o.conf.stepOut {
            if o.conf.fixed {
                o.stepNmax = o.conf.fixedNsteps + 1;
            } else {
                o.stepNmax = o.conf.NmaxSS + 1;
            }
            o.StepRS = vec![0.0; o.stepNmax as usize];
            o.StepH = vec![0.0; o.stepNmax as usize];
            o.StepX = vec![0.0; o.stepNmax as usize];
            // o.StepY = vec![vec![0.0; ndim as usize]; o.stepNmax as usize];
        }
        if o.conf.denseOut {
            o.denseNmax = o.conf.denseNstp + 1;
            o.denseS = vec![0; o.denseNmax as usize];
            o.DenseX = vec![0.0; o.denseNmax as usize];
            // o.DenseY = vec![vec![0.0; ndim as usize]; o.denseNmax as usize];
        }
        if o.conf.denseF.is_some() {
            o.yout = vec![0.0; ndim as usize];
        }
        o
    }

    pub fn execute(&mut self, istep: usize, last: bool, rho_s: f64, h: f64, x: f64, y: &Vector) -> bool {
        // step output using function
        if let Some(step_f) = &self.conf.stepF {
            let stop = step_f(istep, h, x, &y).unwrap();
            if stop {
                return true;
            }
        }

        // save step output
        if self.StepIdx < self.stepNmax {
            self.StepRS[self.StepIdx as usize] = rho_s;
            self.StepH[self.StepIdx as usize] = h;
            self.StepX[self.StepIdx as usize] = x;
            self.StepY[self.StepIdx as usize] = y.clone();
            self.StepIdx += 1;
        }

        // dense output using function
        let mut xo: f64 = 0.0;
        if let Some(dense_f) = &self.conf.denseF {
            if istep == 0 || last {
                xo = x;
                // self.yout = y.clone();
                // let stop = dense_f(istep, h, x, &y, xo, &self.yout).unwrap();
                // if stop {
                // return true;
                // }
                xo += self.conf.denseDx;
            } else {
                xo = self.xout;
                while x >= xo {
                    // self.dout(&self.yout, h, x, &y, xo);
                    // let stop = dense_f(istep, h, x, &y, xo, &self.yout);
                    // if stop {
                    // return true;
                    // }
                    xo += self.conf.denseDx;
                }
            }
        }

        // save dense output
        if self.DenseIdx < self.denseNmax {
            if istep == 0 || last {
                xo = x;
                // self.DenseS[self.DenseIdx as usize] = istep;
                self.DenseX[self.DenseIdx as usize] = xo;
                self.DenseY[self.DenseIdx as usize] = y.clone();
                self.DenseIdx += 1;
                xo = self.conf.denseDx;
            } else {
                xo = self.xout;
                while x >= xo {
                    // self.DenseS[self.DenseIdx as usize] = istep;
                    self.DenseX[self.DenseIdx as usize] = xo;
                    // self.dout(&self.DenseY[self.DenseIdx as usize], h, x, &y, xo);
                    self.DenseIdx += 1;
                    xo += self.conf.denseDx;
                }
            }
        }

        // set xout
        self.xout = xo;
        false
    }

    pub fn get_step_rs(&self) -> &[f64] {
        &self.StepRS[..self.StepIdx as usize]
    }

    pub fn get_step_h(&self) -> &[f64] {
        &self.StepH[..self.StepIdx as usize]
    }

    pub fn get_step_x(&self) -> &[f64] {
        &self.StepX[..self.StepIdx as usize]
    }

    pub fn get_step_y(&self, i: usize) -> Vec<f64> {
        if self.StepIdx > 0 {
            self.StepY.iter().map(|y| y[i]).collect()
        } else {
            vec![]
        }
    }

    /*
    pub fn get_step_y_table(&self) -> Vec<Vec<f64>> {
        if self.StepY.is_empty() {
            vec![]
        } else {
            let ndim = self.StepY[0].len();
            self.StepY.iter().map(|y| y.clone()).collect()
        }
    }

    pub fn get_step_y_table_t(&self) -> Vec<Vec<f64>> {
        if self.StepY.is_empty() {
            vec![]
        } else {
            let ndim = self.StepY[0].len();
            (0..ndim).map(|i| self.StepY.iter().map(|y| y[i]).collect()).collect()
        }
    }
    */

    pub fn get_dense_s(&self) -> &[usize] {
        &self.denseS[..self.DenseIdx as usize]
    }

    pub fn get_dense_x(&self) -> &[f64] {
        &self.DenseX[..self.DenseIdx as usize]
    }

    pub fn get_dense_y(&self, i: usize) -> Vec<f64> {
        if self.DenseIdx > 0 {
            self.DenseY.iter().map(|y| y[i]).collect()
        } else {
            vec![]
        }
    }

    /*
    pub fn get_dense_y_table(&self) -> Vec<Vec<f64>> {
        if self.DenseY.is_empty() {
            vec![]
        } else {
            let ndim = self.DenseY[0].len();
            self.DenseY.iter().map(|y| y.clone()).collect()
        }
    }

    pub fn get_dense_y_table_t(&self) -> Vec<Vec<f64>> {
        if self.DenseY.is_empty() {
            vec![]
        } else {
            let ndim = self.DenseY[0].len();
            (0..ndim).map(|i| self.DenseY.iter().map(|y| y[i]).collect()).collect()
        }
    }
    */
}
