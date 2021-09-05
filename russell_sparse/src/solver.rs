use super::{ConfigSolver, SparseTriplet};
use russell_lab::Vector;

pub trait Solver {
    fn new(config: &ConfigSolver) -> Self;
    fn initialize(&mut self, trip: &SparseTriplet) -> Result<(), &'static str>;
    fn factorize(&mut self) -> Result<(), &'static str>;
    fn solve(&mut self, x: &mut Vector, rhs: &Vector) -> Result<(), &'static str>;
}

/*
use std::fmt;
use std::ptr;

pub struct Solver {
    kind: EnumSolverKind,
    solver_mmp: *const SolverMMP,
    solver_umf: *const SolverUMF,
}

impl Solver {
    /// Allocates a new solver
    pub fn new(kind: EnumSolverKind, symmetric: bool) -> Result<Self, &'static str> {
        match kind {
            EnumSolverKind::Mmp => Ok(Solver {
                kind,
                solver_mmp: &SolverMMP::new(
                    if symmetric {
                        EnumMmpSymmetry::General
                    } else {
                        EnumMmpSymmetry::No
                    },
                    false,
                )?,
                solver_umf: ptr::null(),
            }),
            EnumSolverKind::Umf => Ok(Solver {
                kind,
                solver_mmp: ptr::null(),
                solver_umf: &SolverUMF::new(symmetric)?,
            }),
        }
    }
}

impl fmt::Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            EnumSolverKind::Mmp => unsafe {
                if self.solver_mmp.is_null() {
                    write!(f, "<internal error: null pointer>")
                } else {
                    write!(f, "{}", *self.solver_mmp)
                }
            },
            EnumSolverKind::Umf => unsafe {
                if self.solver_umf.is_null() {
                    write!(f, "<internal error: null pointer>")
                } else {
                    write!(f, "{}", *self.solver_umf)
                }
            },
        }
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    // use russell_chk::*;

    #[test]
    fn new_works() -> Result<(), &'static str> {
        // let solver = Solver::new(EnumSolverKind::Umf, false)?;
        // assert_eq!(format!("{}", solver), "");
        Ok(())
    }
}
