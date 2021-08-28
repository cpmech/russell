//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers

/// Returns package description
pub fn desc() -> String {
    "Sparse matrix tools and solvers".to_string()
}

mod sparse_triplet;
pub use crate::sparse_triplet::*;

#[repr(C)]
pub struct CppSolverMumps {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    pub fn new_dmumps() -> *mut CppSolverMumps;
    pub fn drop_dmumps(data: *mut CppSolverMumps);
}

#[derive(Debug)]

pub struct SolverMumps {
    cdata: *mut CppSolverMumps,
}

impl SolverMumps {
    pub fn new() -> Self {
        unsafe {
            SolverMumps {
                cdata: new_dmumps(),
            }
        }
    }
}

impl Drop for SolverMumps {
    fn drop(&mut self) {
        println!("Dropping THIS!");
        unsafe {
            drop_dmumps(self.cdata);
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
        let res = SolverMumps::new();
        println!("{:?}", res);
    }
}
