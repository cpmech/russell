use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolverUMF {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_umf(symmetric: i32) -> *mut ExtSolverUMF;
    fn drop_solver_umf(solver: *mut ExtSolverUMF);
    fn solver_umf_initialize(
        solver: *mut ExtSolverUMF,
        n: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_a: *const f64,
        ordering: i32,
        scaling: i32,
        verbose: i32,
    ) -> i32;
    fn solver_umf_factorize(solver: *mut ExtSolverUMF, verbose: i32) -> i32;
    fn solver_umf_solve(solver: *mut ExtSolverUMF, rhs: *mut f64, verbose: i32) -> i32;
}

/// Implements Davis' UMFPACK Solver
pub struct SolverUMF {
    symmetric: i32,            // symmetric flag (0 or 1)
    ordering: i32,             // symmetric permutation (ordering)
    scaling: i32,              // scaling strategy
    done_initialize: bool,     // initialization completed
    done_factorize: bool,      // factorization completed
    ndim: usize,               // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolverUMF, // data allocated by the c-code
}

impl SolverUMF {
    pub fn new(symmetric: bool) -> Result<Self, &'static str> {
        let sym: i32 = if symmetric { 1 } else { 0 };
        unsafe {
            let solver = new_solver_umf(sym);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverUMF");
            }
            Ok(SolverUMF {
                symmetric: sym,
                ordering: code_ordering(EnumOrdering::Auto),
                scaling: code_scaling(EnumScaling::Auto),
                done_initialize: false,
                done_factorize: false,
                ndim: 0,
                solver,
            })
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let solver = SolverUMF::new(false)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }
}
