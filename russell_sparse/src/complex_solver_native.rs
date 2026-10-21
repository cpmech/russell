use super::{ComplexCooMatrix, ComplexCscMatrix, ComplexLinSolTrait, LinSolParams, StatsLinSol, Sym};
use crate::StrError;
use crate::solver_native::{native_error, native_stats};
use algebra_linear::sparse::{CscPattern, CscView, LuMethod, LuSolver, LuSymbolic, SparseLuStats};
use russell_lab::{Complex64, ComplexVector, Stopwatch};

/// Solves complex sparse linear systems with the pure-Rust LU from `algebra_linear`
///
/// Complex twin of [`SolverNative`](crate::SolverNative). The supported
/// parameters, the determinant restriction and the statistics accessor are the
/// same; see that type for the details.
///
/// # Examples
///
/// ```
/// use russell_lab::cpx;
/// use russell_lab::ComplexVector;
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     let mut coo = ComplexCooMatrix::new(2, 2, 4, Sym::No)?;
///     coo.put(0, 0, cpx!(1.0, 2.0))?;
///     coo.put(0, 1, cpx!(1.0, 0.0))?;
///     coo.put(1, 0, cpx!(1.0, 0.0))?;
///     coo.put(1, 1, cpx!(1.0, -1.0))?;
///
///     let mut solver = ComplexSolverNative::new()?;
///     solver.factorize(&coo, None)?;
///
///     let rhs = ComplexVector::from(&[cpx!(1.0, 0.0), cpx!(1.0, 0.0)]);
///     let mut x = ComplexVector::new(2);
///     solver.solve(&mut x, &rhs, false)?;
///     Ok(())
/// }
/// ```
pub struct ComplexSolverNative {
    /// Holds the CSC matrix used in the last factorization
    csc: Option<ComplexCscMatrix>,

    /// The symbolic analysis, reused while the structure does not change
    symbolic: Option<LuSymbolic>,

    /// The numeric factors
    factors: Option<LuSolver<Complex64>>,

    /// Statistics of the last factorization
    stats: Option<SparseLuStats>,

    /// Indicates whether the solver has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetric flag saved at the first factorization
    initialized_sym: Sym,

    /// Holds the matrix dimension saved at the first factorization
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved at the first factorization
    initialized_nnz: usize,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on the symbolic analysis in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on the numeric factorization in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on the solve in nanoseconds
    time_solve_ns: u128,
}

impl ComplexSolverNative {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(ComplexSolverNative {
            csc: None,
            symbolic: None,
            factors: None,
            stats: None,
            initialized: false,
            factorized: false,
            initialized_sym: Sym::No,
            initialized_ndim: 0,
            initialized_nnz: 0,
            stopwatch: Stopwatch::new(),
            time_initialize_ns: 0,
            time_factorize_ns: 0,
            time_solve_ns: 0,
        })
    }

    /// Returns the statistics of the last factorization, if any
    pub fn stats(&self) -> Option<&SparseLuStats> {
        self.stats.as_ref()
    }
}

impl ComplexLinSolTrait for ComplexSolverNative {
    fn factorize(&mut self, mat: &ComplexCooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if mat.nrow != mat.ncol {
            return Err("the matrix must be square");
        }
        if mat.nnz < 1 {
            return Err("the COO matrix must have at least one non-zero value");
        }
        if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
            return Err("Native requires Sym::YesFull for symmetric matrices");
        }
        if let Some(par) = params {
            if par.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
        }

        // Pick the factorization before the first analysis; later calls reuse it.
        let method = match params {
            Some(par) if par.native_supernodal => LuMethod::Supernodal,
            _ => LuMethod::Simplicial,
        };

        if !self.initialized {
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let csc = ComplexCscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            let pattern = CscPattern::new(mat.nrow, mat.ncol, &col_pointers, &row_indices).map_err(native_error)?;
            let symbolic = LuSymbolic::analyze(pattern, method).map_err(native_error)?;
            self.time_initialize_ns = self.stopwatch.stop();
            self.csc = Some(csc);
            self.symbolic = Some(symbolic);
            self.initialized = true;
        } else {
            if mat.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if mat.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if mat.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
            let csc = ComplexCscMatrix::from_coo(mat)?;
            self.csc = Some(csc);
        }

        let csc = self.csc.as_ref().unwrap();
        let nnz = csc.get_info().2;
        let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
        let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
        let view = CscView::new(mat.nrow, mat.ncol, &col_pointers, &row_indices, &csc.get_values()[..nnz]).map_err(native_error)?;

        self.stopwatch.reset();
        let factors = self.symbolic.as_ref().unwrap().factorize(view).map_err(native_error)?;
        self.time_factorize_ns = self.stopwatch.stop();
        self.stats = Some(native_stats(&factors, mat.nrow, mat.nnz, self.time_initialize_ns, self.time_factorize_ns));
        self.factors = Some(factors);
        self.factorized = true;
        Ok(())
    }

    fn solve(&mut self, x: &mut ComplexVector, rhs: &ComplexVector, _verbose: bool) -> Result<(), StrError> {
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        x.as_mut_data().copy_from_slice(rhs.as_data());
        self.stopwatch.reset();
        self.factors.as_ref().unwrap().solve(x.as_mut_data()).map_err(native_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "Native".to_string();
        stats.output.effective_ordering = "Colamd".to_string();
        stats.output.effective_scaling = "No".to_string();
        stats.time_nanoseconds.initialize_array.push(self.time_initialize_ns);
        stats.time_nanoseconds.factorize_array.push(self.time_factorize_ns);
        stats.time_nanoseconds.solve_array.push(self.time_solve_ns);
    }

    fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    fn get_ns_fact(&self) -> u128 {
        self.time_factorize_ns
    }

    fn get_ns_solve(&self) -> u128 {
        self.time_solve_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Samples;
    use russell_lab::{complex_vec_approx_eq, cpx};

    #[test]
    fn factorize_handles_errors() {
        let mut solver = ComplexSolverNative::new().unwrap();
        assert!(!solver.factorized);

        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        assert_eq!(solver.factorize(&coo, None).err(), Some("the matrix must be square"));

        let coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );

        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("Native requires Sym::YesFull for symmetric matrices")
        );

        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.factorize(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );
        solver.factorize(&coo, None).unwrap();
    }

    #[test]
    fn factorize_and_solve_work() {
        let mut solver = ComplexSolverNative::new().unwrap();
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        let mut x = ComplexVector::new(3);
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "Native");
        assert_eq!(stats.output.effective_ordering, "Colamd");
        assert_eq!(stats.output.effective_scaling, "No");
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_fact() > 0);
        assert!(solver.get_ns_solve() > 0);
        assert!(solver.stats().is_some());

        // second solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);
    }

    #[test]
    fn singular_matrix_is_reported() {
        let mut solver = ComplexSolverNative::new().unwrap();
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(0.0, 0.0)).unwrap();
        assert_eq!(solver.factorize(&coo, None), Err("the matrix is singular"));
    }
}
