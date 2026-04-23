use crate::{CsrMatrix, SparseMatrix, LinSolParams, StrError, Vector};
use russell_lab::vec_norm;

/// Iterative method variants
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterMethod {
    /// Conjugate Gradient method (for symmetric positive-definite matrices)
    CG,
    /// Generalized Minimal Residual method (for general non-symmetric matrices)
    GMRES,
    /// Bi-Conjugate Gradient Stabilized method (for general non-symmetric matrices, lower memory footprint)
    BiCGSTAB,
}

/// Preconditioner variants
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preconditioner {
    /// Jacobi preconditioner (diagonal preconditioning)
    Jacobi,
    /// ILU0 incomplete factorization preconditioner
    ILU0,
    /// No preconditioner
    None,
}

/// Main iterative solver structure
pub struct IterativeSolver {
    /// Iterative method type
    method: IterMethod,
    /// Preconditioner type
    precond: Preconditioner,
    /// Convergence tolerance
    tol: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Actual number of iterations performed
    num_iter: usize,
    /// Final residual norm
    final_residual: f64,
    /// Preconditioner data (Jacobi preconditioner stores inverse of diagonal elements)
    precond_data: Option<Vec<f64>>,
    /// CSR matrix reference (for SpMV operations)
    csr_mat: Option<CsrMatrix>,
}

impl IterativeSolver {
    /// Creates a new iterative solver
    ///
    /// # Input
    /// * `method` - Iterative method type
    /// * `precond` - Preconditioner type
    pub fn new(method: IterMethod, precond: Preconditioner) -> Self {
        IterativeSolver {
            method,
            precond,
            tol: 1e-6,
            max_iter: 1000,
            num_iter: 0,
            final_residual: 0.0,
            precond_data: None,
            csr_mat: None,
        }
    }

    /// Sets the convergence tolerance
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tol = tol;
    }

    /// Sets the maximum number of iterations
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    /// Returns the actual number of iterations performed
    pub fn iterations(&self) -> usize {
        self.num_iter
    }

    /// Returns the final residual norm
    pub fn residual_norm(&self) -> f64 {
        self.final_residual
    }

    /// Builds the preconditioner (internal method)
    fn build_preconditioner(&mut self, csr: &CsrMatrix) -> Result<(), StrError> {
        match self.precond {
            Preconditioner::Jacobi => {
                // Jacobi preconditioner: extract diagonal elements and take their reciprocal
                let mut diag = vec![0.0; csr.nrow];
                for i in 0..csr.nrow {
                    let start = csr.col_pointers[i] as usize;
                    let end = csr.col_pointers[i + 1] as usize;
                    for j in start..end {
                        if csr.row_indices[j] as usize == i {
                            diag[i] = 1.0 / csr.values[j];
                            break;
                        }
                    }
                }
                self.precond_data = Some(diag);
            }
            Preconditioner::ILU0 => {
                // TODO: Implement ILU0 preconditioner
                return Err("ILU0 preconditioner not yet implemented");
            }
            Preconditioner::None => {
                self.precond_data = None;
            }
        }
        self.csr_mat = Some(csr.clone());
        Ok(())
    }

    /// Sparse Matrix-Vector product (SpMV): y = A * x
    fn spmv(&self, y: &mut Vector, x: &Vector) {
        let csr = self.csr_mat.as_ref().unwrap();
        y.fill(0.0);
        for i in 0..csr.nrow {
            let start = csr.col_pointers[i] as usize;
            let end = csr.col_pointers[i + 1] as usize;
            for j in start..end {
                let col = csr.row_indices[j] as usize;
                y[i] += csr.values[j] * x[col];
            }
        }
    }

    /// Applies the preconditioner: z = M^{-1} * r
    fn apply_preconditioner(&self, z: &mut Vector, r: &Vector) {
        match self.precond {
            Preconditioner::Jacobi => {
                let diag = self.precond_data.as_ref().unwrap();
                for i in 0..z.dim() {
                    z[i] = diag[i] * r[i];
                }
            }
            Preconditioner::None => {
                z.update(1.0, r, 0.0);
            }
            _ => {
                z.update(1.0, r, 0.0);
            }
        }
    }

    /// Executes the CG iterative solver
    fn solve_cg(&mut self, x: &mut Vector, b: &Vector) -> Result<(), StrError> {
        let n = x.dim();
        let mut r = Vector::new(n);
        let mut p = Vector::new(n);
        let mut z = Vector::new(n);
        let mut ap = Vector::new(n);

        // Initial residual r0 = b - A*x0
        self.spmv(&mut r, x);
        r.update(-1.0, &r, 1.0, b);

        let mut rho_prev = 0.0;
        self.num_iter = 0;

        loop {
            // Apply preconditioner z = M^{-1}*r
            self.apply_preconditioner(&mut z, &r);

            // rho = r^T * z
            let rho_curr = r.dot(&z);

            // Check convergence
            self.final_residual = vec_norm(&r, 2);
            if self.final_residual < self.tol || self.num_iter >= self.max_iter {
                break;
            }

            // Update search direction p
            if self.num_iter == 0 {
                p.update(1.0, &z, 0.0);
            } else {
                let beta = rho_curr / rho_prev;
                p.update(1.0, &z, beta, &p);
            }

            // Compute A*p
            self.spmv(&mut ap, &p);

            // Compute step size alpha
            let alpha = rho_curr / p.dot(&ap);

            // Update solution x and residual r
            x.update(1.0, x, alpha, &p);
            r.update(1.0, r, -alpha, &ap);

            rho_prev = rho_curr;
            self.num_iter += 1;
        }

        if self.num_iter >= self.max_iter && self.final_residual >= self.tol {
            return Err("CG solver did not converge within maximum iterations");
        }

        Ok(())
    }
}

impl crate::LinSolTrait for IterativeSolver {
    fn setup(&mut self, mat: &SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        let csr = mat.as_csr()?;
        self.build_preconditioner(csr)?;
        
        // Optional: Read iterative parameters from params
        if let Some(par) = params {
            if par.verbose {
                println!("Iterative solver setup complete with {:?} method and {:?} preconditioner", self.method, self.precond);
            }
        }
        
        Ok(())
    }

    #[deprecated(since = "0.6.0", note = "Please use `setup()` instead with the unified `SparseMatrix` type")]
    fn factorize(&mut self, _mat: &crate::CooMatrix, _params: Option<LinSolParams>) -> Result<(), StrError> {
        Err("Iterative solver does not support factorize(); use setup() with CSR matrix instead")
    }

    fn solve(&mut self, x: &mut Vector, b: &Vector, _verbose: bool) -> Result<(), StrError> {
        match self.method {
            IterMethod::CG => self.solve_cg(x, b),
            IterMethod::GMRES => Err("GMRES not yet implemented"),
            IterMethod::BiCGSTAB => Err("BiCGSTAB not yet implemented"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, CsrMatrix, SparseMatrix, Sym, Vector};
    use russell_lab::vec_approx_eq;

    fn create_test_csr() -> CsrMatrix {
        let mut coo = CooMatrix::new(2, 2, 3, Sym::YesLower).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        CsrMatrix::from(coo)
    }

    #[test]
    fn cg_solver_works() {
        let csr = create_test_csr();
        let sparse = SparseMatrix::from(csr);
        
        let mut solver = IterativeSolver::new(IterMethod::CG, Preconditioner::Jacobi);
        solver.set_tolerance(1e-10);
        solver.set_max_iterations(100);
        
        solver.setup(&sparse, None).unwrap();
        
        let mut x = Vector::new(2);
        let b = Vector::from(&[5.0, 8.0]);
        solver.solve(&mut x, &b, false).unwrap();
        
        let correct = vec![1.0, 2.0];
        vec_approx_eq(&x, &correct, 1e-9);
        
        assert!(solver.iterations() < 10);
        assert!(solver.residual_norm() < 1e-10);
    }
}