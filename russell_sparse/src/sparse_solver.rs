use super::{CooMatrix, SolverMUMPS, SolverUMFPACK};
use crate::StrError;
use russell_lab::Vector;

pub trait SparseSolverTrait {
    fn initialize(&mut self, coo: &CooMatrix) -> Result<(), StrError>;
    fn factorize(&mut self, coo: &CooMatrix, verbose: bool) -> Result<(), StrError>;
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError>;
    fn get_effective_ordering(&self) -> String;
    fn get_effective_scaling(&self) -> String;
    fn get_name(&self) -> String;
}

pub struct GeneralSparseSolver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn SparseSolverTrait + 'a>,
}

pub enum SparseSolver {
    MUMPS,
    UMFPACK,
}

impl<'a> GeneralSparseSolver<'a> {
    /// Allocates new instance
    pub fn new(variant: SparseSolver) -> Result<Self, StrError> {
        let actual: Box<dyn SparseSolverTrait> = match variant {
            SparseSolver::MUMPS => Box::new(SolverMUMPS::new()?),
            SparseSolver::UMFPACK => Box::new(SolverUMFPACK::new()?),
        };
        Ok(GeneralSparseSolver { actual })
    }
}
