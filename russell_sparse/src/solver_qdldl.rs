use super::{CooMatrix, LinSolParams, LinSolTrait, StatsLinSol, Sym};
use crate::StrError;
use algebra_linear::sparse::qdldl::{Error, Qdldl, QdldlStats};
use russell_lab::{Stopwatch, Vector};

/// Maps an `algebra_linear` QDLDL error to the crate's string error.
pub(crate) fn qdldl_error(error: Error) -> StrError {
    match error {
        Error::BadShape => "the matrix shape is inconsistent",
        Error::InvalidStructure => "the matrix has an invalid CSC structure",
        Error::UnsortedRows => "the matrix rows are not sorted",
        Error::DuplicateEntry => "the matrix has a duplicate entry",
        Error::LowerTriangularEntry { .. } => "the matrix has a lower-triangular entry",
        Error::EmptyColumn { .. } => "the matrix has an empty column",
        Error::IndexOverflow => "the matrix is too large for the QDLDL solver",
        Error::StructureMismatch => "the structure of the matrix changed; allocate a new solver",
        Error::NotFinite => "the matrix has a NaN or infinite entry",
        Error::ZeroDiagonal { .. } => "the matrix has a zero diagonal pivot",
        Error::NotFactorized => "the matrix has not been factorized",
    }
}

/// Builds the upper-triangular CSC arrays of a symmetric COO matrix
///
/// `Sym::YesUpper` is passed through as stored, `Sym::YesLower` is flipped directly
/// into upper CSC, and `Sym::YesFull` keeps only the entries with `i <= j`. Rows are
/// sorted per column and duplicate entries are summed, so the result has strictly
/// increasing row indices as `Qdldl::analyze` requires. Empty columns are left empty
/// for the analysis to reject.
fn upper_csc_from_coo(mat: &CooMatrix) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>), StrError> {
    let mut columns: Vec<Vec<(usize, f64)>> = vec![Vec::new(); mat.ncol];
    for k in 0..mat.nnz {
        let i = mat.indices_i[k] as usize;
        let j = mat.indices_j[k] as usize;
        let value = mat.values[k];
        let entry = match mat.symmetric {
            Sym::YesUpper => Some((j, i, value)),
            Sym::YesLower => Some((i, j, value)),
            Sym::YesFull => {
                if i <= j {
                    Some((j, i, value))
                } else {
                    None
                }
            }
            Sym::No => {
                return Err("QDLDL requires a symmetric matrix (Sym::YesUpper, Sym::YesLower, or Sym::YesFull)");
            }
        };
        if let Some((col, row, val)) = entry {
            columns[col].push((row, val));
        }
    }

    let mut total = 0;
    let mut col_pointers = Vec::with_capacity(mat.ncol + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_pointers.push(0);
    for mut col in columns {
        col.sort_by_key(|&(row, _)| row);
        let mut k = 0;
        while k < col.len() {
            let (row, mut value) = col[k];
            k += 1;
            while k < col.len() && col[k].0 == row {
                value += col[k].1;
                k += 1;
            }
            row_indices.push(row);
            values.push(value);
            total += 1;
        }
        col_pointers.push(row_indices.len());
    }
    if total == 0 {
        return Err("the upper triangle of the matrix has no entries");
    }
    Ok((col_pointers, row_indices, values))
}

/// Solves sparse linear systems with the pure-Rust QDLDL port from `algebra_linear`
///
/// This is the `Genie::Qdldl` backend. It computes an unpivoted
/// `A = (I + L) D (I + L)^T` factorization of the upper triangle of a symmetric
/// matrix: the form QDLDL was written for, aimed at symmetric quasi-definite (SQD)
/// systems, including KKT saddle-point matrices with a zero trailing block. It needs
/// no external library.
///
/// # Supported parameters
///
/// * `ordering`: there is no fill-reducing ordering; the factorization keeps the
///   caller's order, so values other than the reported `No` are ignored.
/// * `scaling`: no scaling is applied, so values other than the reported `No` are
///   ignored.
/// * `compute_determinant`: not available; `preprocess` returns an error.
/// * `compute_error_estimates` and `compute_condition_numbers`: not available;
///   the corresponding statistics stay zero.
///
/// # Input class
///
/// Symmetric systems stored as `Sym::YesUpper` (used directly), `Sym::YesLower`
/// (flipped into upper CSC while converting), or `Sym::YesFull` (the lower half is
/// dropped). The converted upper triangle has its rows sorted per column and its
/// duplicates summed, so the COO matrix may be unsorted and may carry repeated
/// entries. A diagonal entry may be missing or zero, which is what a zero dual block
/// needs; an empty column is rejected at analysis. `Sym::No` is rejected: factoring
/// one triangle of an unsymmetric matrix would answer a different problem.
///
/// QDLDL assumes quasi-definiteness and cannot check it at factorization time. The
/// reported positive inertia is the caller's evidence, not a guarantee: for an SQD
/// matrix it counts the positive pivots of the `(1,1)` block, while a symmetric
/// indefinite matrix can factor without complaint and still give a poor answer.
///
/// The elimination tree is computed once, in the first `preprocess`. A later
/// `preprocess` rebuilds the upper triangle and calls `factorize`, which enforces the
/// analyzed pattern, so a structural change surfaces as an error instead of reusing
/// a stale tree.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_approx_eq, Vector};
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     let ndim = 3;
///     let nnz = 5;
///
///     let mut solver = SolverQdldl::new()?;
///
///     // upper triangle of the SPD matrix
///     // ┌             ┐
///     // │  4  1  0    │
///     // │  1  3  1    │
///     // │  0  1  2    │
///     // └             ┘
///     let mut coo = CooMatrix::new(ndim, ndim, nnz, Sym::YesUpper)?;
///     coo.put(0, 0, 4.0)?;
///     coo.put(0, 1, 1.0)?;
///     coo.put(1, 1, 3.0)?;
///     coo.put(1, 2, 1.0)?;
///     coo.put(2, 2, 2.0)?;
///
///     solver.preprocess(&coo, None)?;
///
///     let rhs = Vector::from(&[6.0, 10.0, 8.0]);
///     let mut x = Vector::new(ndim);
///     solver.solve(&mut x, &rhs, false)?;
///
///     let correct = vec![1.0, 2.0, 3.0];
///     vec_approx_eq(&x, &correct, 1e-13);
///     Ok(())
/// }
/// ```
pub struct SolverQdldl {
    /// The opaque QDLDL handle
    qdldl: Option<Qdldl>,

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

impl SolverQdldl {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        Ok(SolverQdldl {
            qdldl: None,
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
    pub fn stats(&self) -> Option<&QdldlStats> {
        self.qdldl.as_ref().and_then(|qdldl| qdldl.stats())
    }
}

impl LinSolTrait for SolverQdldl {
    fn preprocess(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
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
            let (col_pointers, row_indices, values) = upper_csc_from_coo(mat)?;
            self.qdldl
                .as_mut()
                .unwrap()
                .factorize(&col_pointers, &row_indices, &values)
                .map_err(qdldl_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();
        } else {
            if mat.nrow != mat.ncol {
                return Err("the matrix must be square");
            }
            if mat.nnz < 1 {
                return Err("the COO matrix must have at least one non-zero value");
            }
            if mat.symmetric == Sym::No {
                return Err("QDLDL requires a symmetric matrix (Sym::YesUpper, Sym::YesLower, or Sym::YesFull)");
            }
            let effective_params = params.unwrap_or_else(LinSolParams::new);
            if effective_params.compute_determinant {
                return Err("determinant evaluation is not available for this Genie");
            }
            self.initialized_params = effective_params;
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.stopwatch.reset();
            let (col_pointers, row_indices, values) = upper_csc_from_coo(mat)?;
            let mut qdldl = Qdldl::analyze(&col_pointers, &row_indices).map_err(qdldl_error)?;
            self.time_initialize_ns = self.stopwatch.stop();

            self.stopwatch.reset();
            qdldl.factorize(&col_pointers, &row_indices, &values).map_err(qdldl_error)?;
            self.time_preprocess_ns = self.stopwatch.stop();

            self.qdldl = Some(qdldl);
            self.initialized = true;
        }

        self.preprocessed = true;
        Ok(())
    }

    fn solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
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
        self.qdldl.as_ref().unwrap().solve(x.as_mut_data()).map_err(qdldl_error)?;
        self.time_solve_ns = self.stopwatch.stop();
        Ok(())
    }

    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "QDLDL".to_string();
        stats.output.effective_ordering = "No".to_string();
        stats.output.effective_scaling = "No".to_string();
        stats.output.qdldl_positive_inertia = self.stats().map(|s| s.positive_inertia);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::vec_approx_eq;

    /// Returns the upper triangle of the SPD matrix [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    fn spd_3x3_upper() -> CooMatrix {
        let mut coo = CooMatrix::new(3, 3, 5, Sym::YesUpper).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(1, 2, 1.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo
    }

    #[test]
    fn solve_works_spd_3x3() {
        let coo = spd_3x3_upper();
        let rhs = Vector::from(&[6.0, 10.0, 8.0]);
        let x_correct = &[1.0, 2.0, 3.0];

        let mut solver = SolverQdldl::new().unwrap();
        solver.preprocess(&coo, None).unwrap();
        assert!(solver.get_ns_init() > 0);
        assert_eq!(solver.stats().unwrap().positive_inertia, 3);

        let mut x = Vector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "QDLDL");
        assert_eq!(stats.output.effective_ordering, "No");
        assert_eq!(stats.output.effective_scaling, "No");
        assert_eq!(stats.output.qdldl_positive_inertia, Some(3));
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.preprocess_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);
        assert!(solver.get_ns_preprocess() > 0);
        assert!(solver.get_ns_solve() > 0);
    }

    #[test]
    fn solve_works_sqd_kkt_4x4() {
        // ┌             ┐
        // │  2  1  1  0 │
        // │  1  2  0  1 │
        // │  1  0  0  0 │
        // │  0  1  0  0 │
        // └             ┘
        let mut coo = CooMatrix::new(4, 4, 7, Sym::YesUpper).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(0, 2, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(1, 3, 1.0).unwrap();
        coo.put(2, 2, 0.0).unwrap();
        coo.put(3, 3, 0.0).unwrap();

        let rhs = Vector::from(&[7.0, 9.0, 1.0, 2.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0];

        let mut solver = SolverQdldl::new().unwrap();
        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(4);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.qdldl_positive_inertia, Some(2));
    }

    #[test]
    fn builders_sort_and_sum_duplicates() {
        let rhs = Vector::from(&[6.0, 10.0, 8.0]);
        let x_correct = &[1.0, 2.0, 3.0];

        // lower triangle, unsorted, with duplicates and a mirrored value that is not
        // stored here
        let mut lower = CooMatrix::new(3, 3, 8, Sym::YesLower).unwrap();
        lower.put(2, 1, 0.5).unwrap();
        lower.put(1, 0, 1.0).unwrap();
        lower.put(0, 0, 2.0).unwrap();
        lower.put(2, 2, 2.0).unwrap();
        lower.put(1, 1, 1.0).unwrap();
        lower.put(1, 1, 2.0).unwrap();
        lower.put(2, 1, 0.5).unwrap();
        lower.put(0, 0, 2.0).unwrap();
        let mut solver = SolverQdldl::new().unwrap();
        solver.preprocess(&lower, None).unwrap();
        let mut x = Vector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);

        // full matrix, unsorted, with duplicates in the upper triangle and lower
        // entries that must be dropped
        let mut full = CooMatrix::new(3, 3, 8, Sym::YesFull).unwrap();
        full.put(1, 0, 7.0).unwrap();
        full.put(0, 1, 0.4).unwrap();
        full.put(2, 1, 9.0).unwrap();
        full.put(0, 0, 4.0).unwrap();
        full.put(1, 2, 1.0).unwrap();
        full.put(1, 1, 3.0).unwrap();
        full.put(0, 1, 0.6).unwrap();
        full.put(2, 2, 2.0).unwrap();
        let mut solver = SolverQdldl::new().unwrap();
        solver.preprocess(&full, None).unwrap();
        let mut x = Vector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-13);
    }

    #[test]
    fn preprocess_errors_work() {
        let mut solver = SolverQdldl::new().unwrap();

        // unsymmetric storage
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        assert_eq!(
            solver.preprocess(&coo, None).err(),
            Some("QDLDL requires a symmetric matrix (Sym::YesUpper, Sym::YesLower, or Sym::YesFull)")
        );

        // rectangular
        let (coo, _, _, _) = Samples::rectangular_1x7();
        assert_eq!(solver.preprocess(&coo, None).err(), Some("the matrix must be square"));

        // determinant is not available
        let coo = spd_3x3_upper();
        let mut params = LinSolParams::new();
        params.compute_determinant = true;
        assert_eq!(
            solver.preprocess(&coo, Some(params)).err(),
            Some("determinant evaluation is not available for this Genie")
        );

        // nothing in the upper triangle
        let mut below = CooMatrix::new(2, 2, 1, Sym::YesFull).unwrap();
        below.put(1, 0, 1.0).unwrap();
        assert_eq!(
            solver.preprocess(&below, None).err(),
            Some("the upper triangle of the matrix has no entries")
        );

        // zero pivot
        let mut zero = CooMatrix::new(2, 2, 3, Sym::YesUpper).unwrap();
        zero.put(0, 0, 1.0).unwrap();
        zero.put(0, 1, 1.0).unwrap();
        zero.put(1, 1, 1.0).unwrap();
        assert_eq!(solver.preprocess(&zero, None).err(), Some("the matrix has a zero diagonal pivot"));
    }

    #[test]
    fn refactorization_contract_works() {
        let coo = spd_3x3_upper();
        let mut solver = SolverQdldl::new().unwrap();
        solver.preprocess(&coo, None).unwrap();
        let init = solver.get_ns_init();

        // None and identical params are both accepted and keep the analysis
        solver.preprocess(&coo, None).unwrap();
        solver.preprocess(&coo, Some(LinSolParams::new())).unwrap();
        assert_eq!(solver.get_ns_init(), init);

        // a changed parameter is rejected
        let changed = LinSolParams {
            verbose: true,
            ..LinSolParams::new()
        };
        assert_eq!(
            solver.preprocess(&coo, Some(changed)),
            Err("subsequent factorizations must not change LinSolParams")
        );

        // a different upper pattern with the same dimension and non-zero count
        let mut other = CooMatrix::new(3, 3, 5, Sym::YesUpper).unwrap();
        other.put(0, 0, 4.0).unwrap();
        other.put(0, 1, 1.0).unwrap();
        other.put(0, 2, 1.0).unwrap();
        other.put(1, 1, 3.0).unwrap();
        other.put(2, 2, 2.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("the structure of the matrix changed; allocate a new solver")
        );

        // symmetric mismatch
        let mut other = CooMatrix::new(3, 3, 5, Sym::YesFull).unwrap();
        other.put(0, 0, 4.0).unwrap();
        other.put(0, 1, 1.0).unwrap();
        other.put(1, 0, 1.0).unwrap();
        other.put(1, 1, 3.0).unwrap();
        other.put(2, 2, 2.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );

        // dimension mismatch
        let mut other = CooMatrix::new(2, 2, 2, Sym::YesUpper).unwrap();
        other.put(0, 0, 4.0).unwrap();
        other.put(1, 1, 3.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );

        // non-zero count mismatch
        let mut other = CooMatrix::new(3, 3, 4, Sym::YesUpper).unwrap();
        other.put(0, 0, 4.0).unwrap();
        other.put(0, 1, 1.0).unwrap();
        other.put(1, 1, 3.0).unwrap();
        other.put(2, 2, 2.0).unwrap();
        assert_eq!(
            solver.preprocess(&other, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::YesUpper).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        let mut solver = SolverQdldl::new().unwrap();
        let mut x = Vector::new(2);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function preprocess must be called before solve")
        );
        let mut x = Vector::new(1);
        solver.preprocess(&coo, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = Vector::new(2);
        let rhs = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn qdldl_error_maps_every_variant() {
        assert_eq!(qdldl_error(Error::BadShape), "the matrix shape is inconsistent");
        assert_eq!(qdldl_error(Error::InvalidStructure), "the matrix has an invalid CSC structure");
        assert_eq!(qdldl_error(Error::UnsortedRows), "the matrix rows are not sorted");
        assert_eq!(qdldl_error(Error::DuplicateEntry), "the matrix has a duplicate entry");
        assert_eq!(
            qdldl_error(Error::LowerTriangularEntry { row: 1, col: 0 }),
            "the matrix has a lower-triangular entry"
        );
        assert_eq!(qdldl_error(Error::EmptyColumn { col: 2 }), "the matrix has an empty column");
        assert_eq!(qdldl_error(Error::IndexOverflow), "the matrix is too large for the QDLDL solver");
        assert_eq!(
            qdldl_error(Error::StructureMismatch),
            "the structure of the matrix changed; allocate a new solver"
        );
        assert_eq!(qdldl_error(Error::NotFinite), "the matrix has a NaN or infinite entry");
        assert_eq!(qdldl_error(Error::ZeroDiagonal { index: 1 }), "the matrix has a zero diagonal pivot");
        assert_eq!(qdldl_error(Error::NotFactorized), "the matrix has not been factorized");
    }
}
