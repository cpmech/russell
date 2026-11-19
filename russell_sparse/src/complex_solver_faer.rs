use super::{ComplexCooMatrix, ComplexCscMatrix, ComplexLinSolTrait, LinSolParams, StatsLinSol, Sym};
use crate::StrError;
use crate::solver_faer::{faer_error, resolve_faer_method};
use algebra_linear::sparse::faer::{SparseLu, SparseLuStats};
use russell_lab::{Complex64, ComplexVector, Stopwatch};

/// Solves complex sparse linear systems with the pure-Rust Faer LU port from `algebra_linear`
///
/// Complex twin of [`SolverFaer`](crate::SolverFaer). The supported
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
///     let mut solver = ComplexSolverFaer::new()?;
///     solver.preprocess(&coo, None)?;
///
///     let rhs = ComplexVector::from(&[cpx!(1.0, 0.0), cpx!(1.0, 0.0)]);
///     let mut x = ComplexVector::new(2);
///     solver.solve(&mut x, &rhs, false)?;
///     Ok(())
/// }
/// ```
pub struct ComplexSolverFaer {
    /// Holds the CSC matrix used in the last preprocess
    csc: Option<ComplexCscMatrix>,

    /// The opaque LU handle
    lu: Option<SparseLu<Complex64>>,

    /// Indicates whether the solver has been initialized or not
    initialized: bool,

    /// Parameters used by the first `preprocess`, which a refactorization may repeat
    initialized_params: LinSolParams,

    /// Indicates whether the sparse matrix has been preprocessed or not
    preprocessed: bool,

    /// Holds the symmetric flag saved at the first preprocess
    initialized_sym: Sym,

    /// Holds the matrix dimension saved at the first preprocess
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved at the first preprocess
    initialized_nnz: usize,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on the symbolic analysis in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on the numeric factorization in nanoseconds
    time_preprocess_ns: u128,

    /// Time spent on the solve in nanoseconds
    time_solve_ns: u128,
}

impl ComplexSolverFaer {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(ComplexSolverFaer {
            csc: None,
            lu: None,
            initialized: false,
            initialized_params: LinSolParams::new(),
            preprocessed: false,
            initialized_sym: Sym::No,
            initialized_ndim: 0,
            initialized_nnz: 0,
            stopwatch: Stopwatch::new(),
            time_initialize_ns: 0,
            time_preprocess_ns: 0,
            time_solve_ns: 0,
        })
    }

    /// Returns the statistics of the last preprocess, if any
    pub fn stats(&self) -> Option<&SparseLuStats> {
        self.lu.as_ref().and_then(|lu| lu.stats())
    }
}

impl ComplexLinSolTrait for ComplexSolverFaer {
    fn preprocess(&mut self, mat: &ComplexCooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        if self.initialized {
            if mat.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if mat.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if mat.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
            if let Some(p) = params {
                if p != self.initialized_params {
                    return Err("subsequent factorizations must not change LinSolParams");
                }
            }
            self.stopwatch.reset();
            let csc = ComplexCscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            self.lu
                .as_mut()
                .unwrap()
                .factorize(&col_pointers, &row_indices, &csc.get_values()[..nnz])
                .map_err(faer_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();
            self.csc = Some(csc);
        } else {
            if mat.nrow != mat.ncol {
                return Err("the matrix must be square");
            }
            if mat.nnz < 1 {
                return Err("the COO matrix must have at least one non-zero value");
            }
            if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
                return Err("Faer requires Sym::YesFull for symmetric matrices");
            }
            let effective_params = params.unwrap_or_else(LinSolParams::new);
            if effective_params.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
            let method = resolve_faer_method(effective_params.faer_method)?;
            self.initialized_params = effective_params;
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let csc = ComplexCscMatrix::from_coo(mat)?;
            let nnz = csc.get_info().2;
            let col_pointers: Vec<usize> = csc.get_col_pointers()[..=mat.ncol].iter().map(|&p| p as usize).collect();
            let row_indices: Vec<usize> = csc.get_row_indices()[..nnz].iter().map(|&i| i as usize).collect();
            let mut lu = SparseLu::analyze(&col_pointers, &row_indices, method).map_err(faer_error)?;
            self.time_initialize_ns = self.stopwatch.stop();

            self.stopwatch.reset();
            lu.factorize(&col_pointers, &row_indices, &csc.get_values()[..nnz]).map_err(faer_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();

            self.csc = Some(csc);
            self.lu = Some(lu);
            self.initialized = true;
        }

        self.preprocessed = true;
        Ok(())
    }

    fn solve(&mut self, x: &mut ComplexVector, rhs: &ComplexVector, _verbose: bool) -> Result<(), StrError> {
        if !self.preprocessed {
            return Err("the function preprocess must be called before solve");
        }
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        x.as_mut_data().copy_from_slice(rhs.as_data());
        self.stopwatch.reset();
        self.lu.as_ref().unwrap().solve(x.as_mut_data()).map_err(faer_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "Faer".to_string();
        stats.output.effective_ordering = "Colamd".to_string();
        stats.output.effective_scaling = "No".to_string();
        stats.output.effective_faer_method = self.lu.as_ref().map(|lu| lu.effective_method().as_str().to_string());
        stats.output.faer_flop_ratio = self.lu.as_ref().and_then(|lu| lu.flop_ratio());
        stats.time_nanoseconds.initialize_array.push(self.time_initialize_ns);
        stats.time_nanoseconds.preprocess_array.push(self.time_preprocess_ns);
        stats.time_nanoseconds.solve_array.push(self.time_solve_ns);
    }

    fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    fn get_ns_preprocess(&self) -> u128 {
        self.time_preprocess_ns
    }

    fn get_ns_solve(&self) -> u128 {
        self.time_solve_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaerMethod, Samples};
    use russell_lab::{complex_vec_approx_eq, cpx};

    #[test]
    fn preprocess_handles_errors() {
        let mut solver = ComplexSolverFaer::new().unwrap();
        assert!(!solver.preprocessed);

        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        assert_eq!(solver.preprocess(&coo, None).err(), Some("the matrix must be square"));

        let coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );

        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower();
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("Faer requires Sym::YesFull for symmetric matrices")
        );

        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );

        for bad in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut bad_params = LinSolParams::new();
            bad_params.faer_method = FaerMethod::Auto { flop_ratio_threshold: bad };
            assert_eq!(
                solver.preprocess(&coo, Some(bad_params)).err(),
                Some("the Faer flop ratio threshold must be finite and nonnegative")
            );
        }

        solver.preprocess(&coo, None).unwrap();

        // structure changes are refused
        let mut other = ComplexCooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        other.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        other.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        let mut other = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        other.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        let mut other = ComplexCooMatrix::new(2, 2, 1, Sym::No).unwrap();
        other.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn preprocess_and_solve_work() {
        let mut solver = ComplexSolverFaer::new().unwrap();
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        let mut x = ComplexVector::new(3);
        solver.preprocess(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "Faer");
        assert_eq!(stats.output.effective_ordering, "Colamd");
        assert_eq!(stats.output.effective_scaling, "No");
        assert_eq!(stats.output.effective_faer_method.as_deref(), Some("Simplicial"));
        assert_eq!(stats.output.faer_flop_ratio, None);
        assert!(solver.get_ns_init() > 0);
        assert!(solver.get_ns_preprocess() > 0);
        assert!(solver.get_ns_solve() > 0);
        assert!(solver.stats().is_some());

        // second preprocess with None or identical default params reuses the analysis and second solve works
        solver.preprocess(&coo, None).unwrap();
        solver.preprocess(&coo, Some(LinSolParams::new())).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);
    }

    #[test]
    fn refactorization_validates_and_rejects_differing_params() {
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let first_params = LinSolParams {
            faer_method: FaerMethod::Auto { flop_ratio_threshold: 2.5 },
            ..LinSolParams::new()
        };
        let mut solver = ComplexSolverFaer::new().unwrap();
        solver.preprocess(&coo, Some(first_params)).unwrap();

        solver.preprocess(&coo, Some(first_params)).unwrap();
        solver.preprocess(&coo, None).unwrap();

        let diff_threshold = LinSolParams {
            faer_method: FaerMethod::Auto { flop_ratio_threshold: 3.0 },
            ..first_params
        };
        assert_eq!(
            solver.preprocess(&coo, Some(diff_threshold)),
            Err("subsequent factorizations must not change LinSolParams")
        );
        assert_eq!(
            solver.preprocess(&coo, Some(LinSolParams::new())),
            Err("subsequent factorizations must not change LinSolParams")
        );
    }

    #[test]
    fn preprocess_and_solve_work_supernodal_and_simplicial() {
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];

        let params = LinSolParams {
            faer_method: FaerMethod::Supernodal,
            ..LinSolParams::new()
        };
        let mut solver = ComplexSolverFaer::new().unwrap();
        solver.preprocess(&coo, Some(params)).unwrap();
        let mut x = ComplexVector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_faer_method.as_deref(), Some("Supernodal"));
        assert!(stats.output.faer_flop_ratio.is_some());

        let params = LinSolParams {
            faer_method: FaerMethod::Simplicial,
            ..LinSolParams::new()
        };
        let mut solver = ComplexSolverFaer::new().unwrap();
        solver.preprocess(&coo, Some(params)).unwrap();
        let mut x = ComplexVector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-13);
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_faer_method.as_deref(), Some("Simplicial"));
        assert_eq!(stats.output.faer_flop_ratio, None);
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        coo.put(1, 1, cpx!(456.0, 2.0)).unwrap();
        let mut solver = ComplexSolverFaer::new().unwrap();
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function preprocess must be called before solve")
        );
        let mut x = ComplexVector::new(1);
        solver.preprocess(&coo, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn preprocess_detects_a_reordered_pattern() {
        // Both matrices have the same sym, ndim and nnz, but the first stores
        // (0,0) (0,1) (1,1) and the second (0,1) (1,0) (1,1).
        let mut a = ComplexCooMatrix::new(2, 2, 3, Sym::No).unwrap();
        a.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        a.put(0, 1, cpx!(2.0, 0.0)).unwrap();
        a.put(1, 1, cpx!(3.0, 0.0)).unwrap();
        let mut b = ComplexCooMatrix::new(2, 2, 3, Sym::No).unwrap();
        b.put(0, 1, cpx!(2.0, 0.0)).unwrap();
        b.put(1, 0, cpx!(1.0, 0.0)).unwrap();
        b.put(1, 1, cpx!(3.0, 0.0)).unwrap();

        let mut solver = ComplexSolverFaer::new().unwrap();
        solver.preprocess(&a, None).unwrap();
        assert_eq!(
            solver.preprocess(&b, None),
            Err("the structure of the matrix changed; allocate a new solver")
        );
    }

    #[test]
    fn singular_matrix_is_reported() {
        let mut solver = ComplexSolverFaer::new().unwrap();
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(0.0, 0.0)).unwrap();
        assert_eq!(solver.preprocess(&coo, None), Err("the matrix is singular"));
    }
}
