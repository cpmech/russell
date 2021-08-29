//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers

/// Returns package description
pub fn desc() -> String {
    "Sparse matrix tools and solvers".to_string()
}

mod sparse_triplet;
pub use crate::sparse_triplet::*;

pub const MUMPS_SYMMETRY_NONE: i32 = 0;
pub const MUMPS_SYMMETRY_POS_DEF: i32 = 1;
pub const MUMPS_SYMMETRY_GENERAL: i32 = 2;

#[repr(C)]
pub struct CppSolverMumps {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    pub fn new_solver_mumps(symmetry: i32, verbose: i32) -> *mut CppSolverMumps;
    pub fn drop_solver_mumps(solver: *mut CppSolverMumps);
}

#[derive(Debug)]

pub struct SolverMumps {
    solver: *mut CppSolverMumps,
}

impl SolverMumps {
    pub fn new(symmetry: i32, verbose: bool) -> Self {
        let verb = if verbose { 1 } else { 0 };
        unsafe {
            SolverMumps {
                solver: new_solver_mumps(symmetry, verb),
            }
        }
    }
}

impl Drop for SolverMumps {
    fn drop(&mut self) {
        println!("Dropping THIS!");
        unsafe {
            drop_solver_mumps(self.solver);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_code_works() {
        println!(">>>>>>>>>>>>>>>. hi from Rust");
        let res = SolverMumps::new(MUMPS_SYMMETRY_NONE, true);
        println!("{:?}", res);
    }
}
