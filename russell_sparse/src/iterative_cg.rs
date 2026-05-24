use crate::CsrMatrix;
use crate::StrError;
use russell_lab::{vec_norm, Norm, Vector};

/// Conjugate Gradient iterative solver for symmetric positive-definite matrices
///
/// # Algorithm
/// The CG method finds the solution to Ax = b where A is symmetric and positive-definite.
/// It minimizes the energy function 0.5*x^T*A*x - b^T*x.
///
/// # Preconditioner
/// Currently supports Jacobi (diagonal) preconditioner or no preconditioner.
pub struct IterCgSolver {
    tol: f64,
    max_iter: usize,
    num_iter: usize,
    final_residual: f64,
    precond_data: Vec<f64>,
    use_jacobi: bool,
}

impl IterCgSolver {
    /// Creates a new CG solver
    pub fn new() -> Self {
        IterCgSolver {
            tol: 1e-6,
            max_iter: 1000,
            num_iter: 0,
            final_residual: 0.0,
            precond_data: Vec::new(),
            use_jacobi: false,
        }
    }

    /// Enables Jacobi preconditioner (diagonal preconditioning)
    pub fn with_jacobi(&mut self) {
        self.use_jacobi = true;
    }

    /// Disables Jacobi preconditioner
    pub fn without_precond(&mut self) {
        self.use_jacobi = false;
    }

    /// Sets the convergence tolerance
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tol = tol;
    }

    /// Sets the maximum number of iterations
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    /// Returns the number of iterations performed
    pub fn iterations(&self) -> usize {
        self.num_iter
    }

    /// Returns the final residual norm
    pub fn residual_norm(&self) -> f64 {
        self.final_residual
    }

    /// Builds the Jacobi preconditioner
    fn build_jacobi_preconditioner(&mut self, csr: &CsrMatrix) {
        let n = csr.nrow;
        self.precond_data.resize(n, 0.0);
        for i in 0..n {
            let start = csr.row_pointers[i] as usize;
            let end = csr.row_pointers[i + 1] as usize;
            for j in start..end {
                if csr.col_indices[j] as usize == i {
                    self.precond_data[i] = 1.0 / csr.values[j];
                    break;
                }
            }
        }
    }

    /// Sparse matrix-vector product: y = A * x
    fn spmv(&self, y: &mut Vector, x: &Vector, csr: &CsrMatrix) {
        let n = csr.nrow;
        let mirror_required = csr.symmetric.triangular();
        y.fill(0.0);
        for i in 0..n {
            let start = csr.row_pointers[i] as usize;
            let end = csr.row_pointers[i + 1] as usize;
            let mut sum = 0.0;
            for j in start..end {
                let col = csr.col_indices[j] as usize;
                let aij = csr.values[j];
                sum += aij * x[col];
                if mirror_required && i != col {
                    y[col] += aij * x[i];
                }
            }
            y[i] += sum;
        }
    }

    /// Applies the preconditioner: z = M^{-1} * r
    fn apply_preconditioner(&self, z: &mut Vector, r: &Vector) {
        if self.use_jacobi {
            for i in 0..z.dim() {
                z[i] = self.precond_data[i] * r[i];
            }
        } else {
            z.clone_from(r);
        }
    }

    /// Solves the linear system A*x = b using CG
    ///
    /// # Input
    /// * `x` - Initial guess (in/out)
    /// * `b` - Right-hand side vector
    /// * `mat` - Coefficient matrix (CSR format, must be symmetric positive-definite)
    pub fn solve(&mut self, x: &mut Vector, b: &Vector, mat: &CsrMatrix) -> Result<(), StrError> {
        let n = x.dim();

        // Build preconditioner
        if self.use_jacobi {
            self.build_jacobi_preconditioner(mat);
        }

        // Initialize working vectors
        let mut r = Vector::new(n);
        let mut z = Vector::new(n);
        let mut p = Vector::new(n);
        let mut ap = Vector::new(n);

        // Initial residual: r0 = b - A*x0
        self.spmv(&mut r, x, mat);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }

        let mut rho_prev = 0.0;
        self.num_iter = 0;

        loop {
            // Apply preconditioner: z = M^{-1}*r
            self.apply_preconditioner(&mut z, &r);

            // rho = r^T * z
            let mut rho_curr = 0.0;
            for i in 0..n {
                rho_curr += r[i] * z[i];
            }

            // Check convergence
            self.final_residual = vec_norm(&r, Norm::Euc);
            if self.final_residual < self.tol {
                return Ok(());
            }

            if self.num_iter >= self.max_iter {
                return Err("CG solver did not converge within maximum iterations");
            }

            // Update search direction p
            if self.num_iter == 0 {
                p.clone_from(&z);
            } else {
                let beta = rho_curr / rho_prev;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
            }

            // Compute A*p
            self.spmv(&mut ap, &p, mat);

            // Compute step size alpha = rho / (p^T * A*p)
            let mut pap = 0.0;
            for i in 0..n {
                pap += p[i] * ap[i];
            }

            if pap.abs() < 1e-15 {
                return Ok(());
            }

            let alpha = rho_curr / pap;

            // Update solution: x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // Update residual: r = r - alpha * A*p
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }

            rho_prev = rho_curr;
            self.num_iter += 1;
        }
    }
}

impl Default for IterCgSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, CsrMatrix, Sym};
    use russell_lab::{vec_approx_eq, Vector};

    fn create_test_csr() -> CsrMatrix {
        let mut coo = CooMatrix::new(2, 2, 3, Sym::YesLower).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        CsrMatrix::from_coo(&coo).unwrap()
    }

    #[test]
    fn cg_solver_with_jacobi_works() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.with_jacobi();

        let mut x = Vector::new(2);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, &csr).unwrap();

        let correct = vec![1.4, 2.2];
        vec_approx_eq(&x, &correct, 1e-9);

        assert!(solver.iterations() < 10);
        assert!(solver.residual_norm() < 1e-10);
    }

    #[test]
    fn cg_solver_without_precond_works() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.without_precond();

        let mut x = Vector::new(2);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, &csr).unwrap();

        let correct = vec![1.4, 2.2];
        vec_approx_eq(&x, &correct, 1e-9);

        assert!(solver.iterations() < 10);
        assert!(solver.residual_norm() < 1e-10);
    }

    #[test]
    fn cg_solver_larger_matrix_works() {
        let mut coo = CooMatrix::new(4, 4, 10, Sym::YesLower).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();
        coo.put(2, 2, 6.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(3, 3, 4.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(1000);
        solver.with_jacobi();

        let b = Vector::from(&[5.0, 12.0, 18.0, 10.0]);
        let mut x = Vector::new(4);
        solver.solve(&mut x, &b, &csr).unwrap();

        assert!(solver.iterations() < 50);
        assert!(solver.residual_norm() < 1e-10);
    }

    #[test]
    fn cg_solver_returns_iteration_count() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.with_jacobi();

        let mut x = Vector::new(2);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, &csr).unwrap();

        let iters = solver.iterations();
        assert!(iters > 0 && iters < 10);
    }

    #[test]
    fn cg_solver_handles_max_iterations() {
        let mut coo = CooMatrix::new(10, 10, 10, Sym::No).unwrap();
        for i in 0..10 {
            coo.put(i, i, (i + 1) as f64).unwrap();
        }
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-15);
        solver.set_max_iterations(2);
        solver.without_precond();

        let b = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut x = Vector::new(10);
        let result = solver.solve(&mut x, &b, &csr);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "CG solver did not converge within maximum iterations");
    }

    #[test]
    fn cg_solver_identity_matrix() {
        let mut coo = CooMatrix::new(3, 3, 3, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 1.0).unwrap();
        coo.put(2, 2, 1.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(10);
        solver.without_precond();

        let b = Vector::from(&[1.0, 2.0, 3.0]);
        let mut x = Vector::new(3);
        solver.solve(&mut x, &b, &csr).unwrap();

        let correct = vec![1.0, 2.0, 3.0];
        vec_approx_eq(&x, &correct, 1e-9);
    }

    #[test]
    fn cg_solver_diagonal_matrix() {
        let mut coo = CooMatrix::new(3, 3, 3, Sym::No).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(2, 2, 4.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(10);
        solver.without_precond();

        let b = Vector::from(&[2.0, 6.0, 8.0]);
        let mut x = Vector::new(3);
        solver.solve(&mut x, &b, &csr).unwrap();

        let correct = vec![1.0, 2.0, 2.0];
        vec_approx_eq(&x, &correct, 1e-9);
    }

    #[test]
    fn cg_solver_symmetric_full_storage() {
        let mut coo = CooMatrix::new(3, 3, 9, Sym::No).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(0, 2, 1.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(1, 2, 2.0).unwrap();
        coo.put(2, 0, 1.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();
        coo.put(2, 2, 6.0).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.without_precond();

        let b = Vector::from(&[6.0, 12.0, 15.0]);
        let mut x = Vector::new(3);
        solver.solve(&mut x, &b, &csr).unwrap();

        assert!(solver.residual_norm() < 1e-10);
    }

    #[test]
    fn cg_solver_single_iteration_convergence() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1.0);
        solver.set_max_iterations(100);
        solver.without_precond();

        let mut x = Vector::new(2);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, &csr).unwrap();

        assert_eq!(solver.iterations(), 1);
    }

    #[test]
    fn cg_solver_exact_solution_initial_guess() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.without_precond();

        let mut x = Vector::from(&[1.4, 2.2]);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, &csr).unwrap();

        assert_eq!(solver.iterations(), 0);
        assert!(solver.residual_norm() < 1e-10);
    }

    #[test]
    fn cg_solver_solve_called_twice() {
        let csr = create_test_csr();
        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        solver.without_precond();

        let b1 = Vector::from(&[5.0, 8.0]);
        let mut x1 = Vector::new(2);
        solver.solve(&mut x1, &b1, &csr).unwrap();

        let correct1 = vec![1.4, 2.2];
        vec_approx_eq(&x1, &correct1, 1e-9);

        let b2 = Vector::from(&[2.0, 3.0]);
        let mut x2 = Vector::new(2);
        solver.solve(&mut x2, &b2, &csr).unwrap();

        let correct2 = vec![0.6, 0.8];
        vec_approx_eq(&x2, &correct2, 1e-9);
    }

    #[test]
    fn cg_solver_default_values() {
        let solver = IterCgSolver::new();
        assert_eq!(solver.tol, 1e-6);
        assert_eq!(solver.max_iter, 1000);
        assert_eq!(solver.iterations(), 0);
    }

    #[test]
    fn cg_solver_large_matrix() {
        let n = 100;
        let mut coo = CooMatrix::new(n, n, n * 2, Sym::YesLower).unwrap();
        for i in 0..n {
            coo.put(i, i, (i + 1) as f64 * 10.0).unwrap();
            if i > 0 {
                coo.put(i, i - 1, -1.0).unwrap();
            }
        }
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-6);
        solver.set_max_iterations(2000);
        solver.with_jacobi();

        let mut b = Vector::new(n);
        for i in 0..n {
            b[i] = (i + 1) as f64;
        }
        let mut x = Vector::new(n);
        solver.solve(&mut x, &b, &csr).unwrap();

        assert!(solver.iterations() < 2000);
        assert!(solver.residual_norm() < 1e-6);
    }

    #[test]
    fn cg_solver_with_jacobi_improves_convergence() {
        let csr = create_test_csr();

        let mut solver_no_precond = IterCgSolver::new();
        solver_no_precond.set_tolerance(1e-10);
        solver_no_precond.set_max_iterations(100);
        solver_no_precond.without_precond();
        let mut x1 = Vector::new(2);
        solver_no_precond.solve(&mut x1, &Vector::from(&[5.0, 8.0]), &csr).unwrap();
        let iter_no_precond = solver_no_precond.iterations();

        let mut solver_jacobi = IterCgSolver::new();
        solver_jacobi.set_tolerance(1e-10);
        solver_jacobi.set_max_iterations(100);
        solver_jacobi.with_jacobi();
        let mut x2 = Vector::new(2);
        solver_jacobi.solve(&mut x2, &Vector::from(&[5.0, 8.0]), &csr).unwrap();
        let iter_jacobi = solver_jacobi.iterations();

        assert!(iter_jacobi <= iter_no_precond);
    }

    #[test]
    fn cg_solver_default_trait_works() {
        let solver = IterCgSolver::default();
        assert_eq!(solver.tol, 1e-6);
        assert_eq!(solver.max_iter, 1000);
    }

    #[test]
    fn cg_solver_zero_projection_early_exit() {
        let n = 2;
        let mut coo = CooMatrix::new(n, n, n * 2, Sym::YesLower).unwrap();
        coo.put(0, 0, 1e-10).unwrap();
        coo.put(1, 0, 0.0).unwrap();
        coo.put(1, 1, 1e-10).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let mut solver = IterCgSolver::new();
        solver.set_tolerance(1e-20);
        solver.set_max_iterations(10);

        let b = Vector::from(&[1.0, 1.0]);
        let mut x = Vector::new(n);
        let _ = solver.solve(&mut x, &b, &csr);
    }
}