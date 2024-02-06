#![allow(unused, non_snake_case)]

use crate::StrError;
use crate::{Method, NumSolver, ParamsRadau5, System, Workspace};
use russell_lab::math::{SQRT_3, SQRT_6};
use russell_lab::{ComplexVector, Matrix, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, SparseMatrix};

pub(crate) struct Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: ParamsRadau5,

    /// ODE system
    system: System<'a, F, J, A>,

    /// Linear solver
    solver: LinSolver<'a>,

    /// Normalized vectors, one for each of the 3 stages
    z: Vec<Vector>,

    /// Workspace, one for each of the 3 stages
    w: Vec<Vector>,

    /// Incremental workspace, one for each of the 3 stages
    dw: Vec<Vector>,

    /// Collocation values, one for each of the 3 stages
    ycol: Vec<Vector>,

    /// Packed vectors (v1, v2)
    v12: ComplexVector,

    /// Packed vectors (dw1, dw2)
    dw12: ComplexVector,

    /// Error estimate workspace
    ez: Vector,

    /// Error estimate workspace
    lerr: Vector,

    /// Error estimate workspace
    rhs: Vector,

    /// c coefficients
    C: Vector,

    /// T matrix
    T: Matrix,

    /// inv(T) matrix
    Ti: Matrix,

    /// alpha-hat
    Alp: f64,

    /// beta-hat
    Bet: f64,

    /// gamma-hat
    Gam: f64,

    /// gamma0 coefficient
    Gam0: f64,

    /// e0 coefficient
    E0: f64,

    /// e1 coefficient
    E1: f64,

    /// e2 coefficient
    E2: f64,

    /// collocation: C1 = (4.D0-SQ6)/10.D0
    Mu1: f64,

    /// collocation: C2 = (4.D0+SQ6)/10.D0
    Mu2: f64,

    /// collocation: C1M1 = C1-1.D0
    Mu3: f64,

    /// collocation: C2M1 = C2-1.D0
    Mu4: f64,

    /// collocation: C1MC2 = C1-C2
    Mu5: f64,
}

impl<'a, F, J, A> Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(params: ParamsRadau5, system: System<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        let nnz = system.jac_nnz + ndim; // +ndim corresponds to the diagonal I matrix
        let symmetry = system.jac_symmetry;
        let one_based = if params.lin_sol == Genie::Mumps { true } else { false };
        let c1 = f64::powf(9.0, 1.0 / 3.0);
        let c2 = f64::powf(3.0, 3.0 / 2.0);
        let c3 = f64::powf(9.0, 2.0 / 3.0);
        let Gam0 = c1 / (c3 + 3.0 * c1 - 3.0);
        let Mu1 = (4.0 - SQRT_6) / 10.0;
        let Mu2 = (4.0 + SQRT_6) / 10.0;
        Radau5 {
            params,
            system,
            solver: LinSolver::new(params.lin_sol).unwrap(),
            z: vec![Vector::new(ndim), Vector::new(ndim), Vector::new(ndim)],
            w: vec![Vector::new(ndim), Vector::new(ndim), Vector::new(ndim)],
            dw: vec![Vector::new(ndim), Vector::new(ndim), Vector::new(ndim)],
            ycol: vec![Vector::new(ndim), Vector::new(ndim), Vector::new(ndim)],
            v12: ComplexVector::new(ndim),
            dw12: ComplexVector::new(ndim),
            ez: Vector::new(ndim),
            lerr: Vector::new(ndim),
            rhs: Vector::new(ndim),
            C: Vector::from(&[Mu1, Mu2, 1.0]),
            T: Matrix::from(&[
                [
                    9.1232394870892942792e-02,
                    -0.14125529502095420843,
                    -3.0029194105147424492e-02,
                ],
                [0.24171793270710701896, 0.20412935229379993199, 0.38294211275726193779],
                [0.96604818261509293619, 1.0, 0.0],
            ]),
            Ti: Matrix::from(&[
                [4.3255798900631553510, 0.33919925181580986954, 0.54177053993587487119],
                [-4.1787185915519047273, -0.32768282076106238708, 0.47662355450055045196],
                [-0.50287263494578687595, 2.5719269498556054292, -0.59603920482822492497],
            ]),
            Alp: -c1 / 2.0 + 3.0 / (2.0 * c1) + 3.0,
            Bet: (SQRT_3 * c1) / 2.0 + c2 / (2.0 * c1),
            Gam: c1 - 3.0 / c1 + 3.0,
            Gam0,
            E0: Gam0 * (-13.0 - 7.0 * SQRT_6) / 3.0,
            E1: Gam0 * (-13.0 + 7.0 * SQRT_6) / 3.0,
            E2: Gam0 * (-1.0) / 3.0,
            Mu1,
            Mu2,
            Mu3: Mu1 - 1.0,
            Mu4: Mu2 - 1.0,
            Mu5: Mu1 - Mu2,
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, y: &Vector) {}

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        panic!("TODO");
        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        _work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        _args: &mut A,
    ) -> Result<(), StrError> {
        panic!("TODO");
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {
        panic!("TODO");
    }

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}
