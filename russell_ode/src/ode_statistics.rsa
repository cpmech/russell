#![allow(unused, non_snake_case)]

use russell_sparse::Genie;

pub struct OdeStatistics {
    pub Nfeval: usize,    // number of calls to fcn
    pub Njeval: usize,    // number of Jacobian matrix evaluations
    pub Nsteps: usize,    // total number of substeps
    pub Naccepted: usize, // number of accepted substeps
    pub Nrejected: usize, // number of rejected substeps
    pub Ndecomp: usize,   // number of matrix decompositions
    pub Nlinsol: usize,   // number of calls to linsolver
    pub Nitmax: usize,    // number max of iterations
    pub Hopt: f64,        // optimal step size at the end
    pub LsKind: String,   // kind of linear solver used
    pub Implicit: bool,   // method is implicit

    // benchmark
    pub NanosecondsStep: u128,       // maximum time elapsed during steps [nanoseconds]
    pub NanosecondsJeval: u128,      // maximum time elapsed during Jacobian evaluation [nanoseconds]
    pub NanosecondsIniSol: u128,     // maximum time elapsed during initialization of the linear solver [nanoseconds]
    pub NanosecondsFact: u128,       // maximum time elapsed during factorization (if any) [nanoseconds]
    pub NanosecondsLinSol: u128,     // maximum time elapsed during solution of linear system (if any) [nanoseconds]
    pub NanosecondsErrorEstim: u128, // maximum time elapsed during the error estimate [nanoseconds]
    pub NanosecondsTotal: u128,      // total time elapsed during solution of ODE system [nanoseconds]
}

impl OdeStatistics {
    pub fn new(implicit: bool, genie: Genie) -> Self {
        OdeStatistics {
            Nfeval: 0,
            Njeval: 0,
            Nsteps: 0,
            Naccepted: 0,
            Nrejected: 0,
            Ndecomp: 0,
            Nlinsol: 0,
            Nitmax: 0,
            Hopt: 0.0,
            LsKind: genie.to_string(),
            Implicit: implicit,
            NanosecondsStep: 0,
            NanosecondsJeval: 0,
            NanosecondsIniSol: 0,
            NanosecondsFact: 0,
            NanosecondsLinSol: 0,
            NanosecondsErrorEstim: 0,
            NanosecondsTotal: 0,
        }
    }
}
