use crate::{HasJacobian, StrError};
use russell_lab::Vector;
use russell_sparse::{CooMatrix, Symmetry};
use std::marker::PhantomData;

/// Indicates that the system functions do not require extra arguments
pub type NoArgs = u8;

/// Defines a system of first order ordinary differential equations (ODE) or a differential-algebraic system (DAE) of Index-1
///
/// The system is defined by:
///
/// ```text
///     d{y}
/// [M] ———— = {f}(x, {y})
///      dx
/// ```
///
/// where `x` is the independent scalar variable (e.g., time), `{y}` is the solution vector,
/// `{f}` is the right-hand side vector, and `[M]` is the so-called "mass matrix".
///
/// **Note:** The mass matrix is optional and need not be specified.
///
/// The Jacobian is defined by:
///
/// ```text
///               ∂{f}
/// [J](x, {y}) = ————
///               ∂{y}
/// ```
///
/// where `[J]` is the Jacobian matrix.
///
/// # Generics
///
/// The generic arguments here are:
///
/// * `F` -- function to compute the `f` vector: `(f: &mut Vector, x: f64, y: &Vector, args: &mut A)`
/// * `J` -- function to compute the Jacobian: `(jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut A)`
/// * `A` -- generic argument to assist in the `F` and `J` functions. It may be simply the [NoArgs] type indicating that no arguments are needed.
///
/// # Important
///
/// The internal implementation requires that the `multiplier` parameter in
/// the Jacobian function `J` be used used to scale the Jacobian matrix. For example:
///
/// ```text
/// |jj: &mut CooMatrix, x: f64, y: &Vector, multiplier: f64, args: &mut Args| {
///     jj.reset();
///     jj.put(0, 0, multiplier * LAMBDA)?;
///     Ok(())
/// },
/// ```
pub struct System<F, J, A>
where
    F: Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// System dimension
    pub(crate) ndim: usize,

    /// ODE system function
    pub(crate) function: F,

    /// Jacobian function
    pub(crate) jacobian: J,

    /// Indicates whether the analytical Jacobian is available or not
    pub(crate) jac_available: bool,

    /// Number of non-zeros in the Jacobian matrix
    pub(crate) jac_nnz: usize,

    /// Symmetry properties of the Jacobian matrix
    pub(crate) jac_symmetry: Symmetry,

    /// Holds the mass matrix
    pub(crate) mass_matrix: Option<CooMatrix>,

    /// Handle generic argument
    phantom: PhantomData<fn() -> A>,
}

impl<'a, F, J, A> System<F, J, A>
where
    F: Send + Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + Fn(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `ndim` -- dimension of the ODE system (number of equations)
    /// * `function` -- implements the function: `dy/dx = f(x, y)`
    /// * `jacobian` -- implements the Jacobian: `J = df/dy`
    /// * `has_jacobian` -- indicates that the analytical Jacobian is available (input by `jacobian`)
    /// * `jac_nnz` -- the number of non-zeros in the Jacobian; use None to indicate a full matrix (i.e., nnz = ndim * ndim)
    /// * `jac_symmetry` -- specifies the type of symmetry representation for the Jacobian matrix
    ///
    /// # Generics
    ///
    /// See [System] for an explanation of the generic parameters.
    pub fn new(
        ndim: usize,
        function: F,
        jacobian: J,
        has_ana_jacobian: HasJacobian,
        jac_nnz: Option<usize>,
        jac_symmetry: Option<Symmetry>,
    ) -> Self {
        let jac_available = match has_ana_jacobian {
            HasJacobian::Yes => true,
            HasJacobian::No => false,
        };
        System {
            ndim,
            function,
            jacobian,
            jac_available,
            jac_nnz: if let Some(n) = jac_nnz { n } else { ndim * ndim },
            jac_symmetry: if let Some(s) = jac_symmetry { s } else { Symmetry::No },
            mass_matrix: None,
            phantom: PhantomData,
        }
    }

    /// Initializes and enables the mass matrix
    ///
    /// **Note:** Later, call [System::mass_put] to "put" elements in the mass matrix.
    ///
    /// # Input
    ///
    /// * `max_nnz` -- Max number of non-zero values
    /// * `one_based` -- Flag indicating whether the Sparse matrix can be used with Fortran code (e.g., MUMPS) or not.
    ///   Make sure that this flag is the same used for the Jacobian matrix.
    pub fn init_mass_matrix(&mut self, max_nnz: usize, one_based: bool) -> Result<(), StrError> {
        let sym = if self.jac_symmetry == Symmetry::No {
            None
        } else {
            Some(self.jac_symmetry)
        };
        self.mass_matrix = Some(CooMatrix::new(self.ndim, self.ndim, max_nnz, sym, one_based).unwrap());
        Ok(())
    }

    /// Puts a new element in the mass matrix (duplicates allowed)
    ///
    /// See also [russell_sparse::CooMatrix::put].
    ///
    /// # Input
    ///
    /// * `i` -- row index (indices start at zero; zero-based)
    /// * `j` -- column index (indices start at zero; zero-based)
    /// * `value` -- the value M(i,j)
    pub fn mass_put(&mut self, i: usize, j: usize, value: f64) -> Result<(), StrError> {
        match self.mass_matrix.as_mut() {
            Some(mass) => mass.put(i, j, value),
            None => Err("mass matrix has not been initialized/enabled"),
        }
    }

    /// Returns the dimension of the ODE system
    pub fn get_ndim(&self) -> usize {
        self.ndim
    }

    /// Computes the numerical Jacobian
    ///
    /// ```text
    /// ∂{f}                          Δfᵢ
    /// ———— = [J](x, {y})      Jᵢⱼ ≈ ———
    /// ∂{y}                          Δyⱼ
    /// ```
    ///
    /// **Note:** Will call `function` ndim times.
    pub(crate) fn numerical_jacobian(
        &self,
        jj: &mut CooMatrix,
        x: f64,
        y: &mut Vector,
        fxy: &Vector,
        multiplier: f64,
        args: &mut A,
        aux: &mut Vector,
    ) -> Result<(), StrError> {
        assert_eq!(aux.dim(), self.ndim);
        const THRESHOLD: f64 = 1e-5;
        jj.reset();
        for j in 0..self.ndim {
            let yj_original = y[j]; // create copy
            let delta_yj = f64::sqrt(f64::EPSILON * f64::max(THRESHOLD, f64::abs(y[j])));
            y[j] += delta_yj; // y[j] := y[j] + Δy
            (self.function)(aux, x, y, args)?; // work := f(x, y + Δy)
            for i in 0..self.ndim {
                let delta_fi = aux[i] - fxy[i]; // compute Δf[..]
                jj.put(i, j, multiplier * delta_fi / delta_yj).unwrap(); // Δfi/Δfj
            }
            y[j] = yj_original; // restore value
        }
        Ok(())
    }
}

/// Implements a placeholder function for when the analytical Jacobian is unavailable
///
/// **Note:** Use this function with the [crate::HasJacobian::No] option.
pub fn no_jacobian<A>(
    _jj: &mut CooMatrix,
    _x: f64,
    _y: &Vector,
    _multiplier: f64,
    _args: &mut A,
) -> Result<(), StrError> {
    Err("analytical Jacobian is not available")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{no_jacobian, System};
    use crate::HasJacobian;
    use russell_lab::Vector;
    use russell_sparse::CooMatrix;

    #[test]
    fn ode_system_most_none_works() {
        struct Args {
            n_function_eval: usize,
            more_data_goes_here: bool,
        }
        let mut args = Args {
            n_function_eval: 0,
            more_data_goes_here: false,
        };
        let ode = System::new(
            2,
            |f, x, y, args: &mut Args| {
                args.n_function_eval += 1;
                f[0] = -x * y[1];
                f[1] = x * y[0];
                args.more_data_goes_here = true;
                Ok(())
            },
            no_jacobian,
            HasJacobian::No,
            None,
            None,
        );
        // call system function
        let x = 0.0;
        let y = Vector::new(2);
        let mut k = Vector::new(2);
        (ode.function)(&mut k, x, &y, &mut args).unwrap();
        // call jacobian function
        let mut jj = CooMatrix::new(2, 2, 2, None, false).unwrap();
        let m = 1.0;
        assert_eq!(
            (ode.jacobian)(&mut jj, x, &y, m, &mut args),
            Err("analytical Jacobian is not available")
        );
        // check
        println!("n_function_eval = {}", args.n_function_eval);
        assert_eq!(args.n_function_eval, 1);
        assert_eq!(args.more_data_goes_here, true);
    }

    #[test]
    fn ode_system_some_none_works() {
        struct Args {
            n_function_eval: usize,
            n_jacobian_eval: usize,
            more_data_goes_here_fn: bool,
            more_data_goes_here_jj: bool,
        }
        let mut args = Args {
            n_function_eval: 0,
            n_jacobian_eval: 0,
            more_data_goes_here_fn: false,
            more_data_goes_here_jj: false,
        };
        let mut ode = System::new(
            2,
            |f, x, y, args: &mut Args| {
                args.n_function_eval += 1;
                f[0] = -x * y[1];
                f[1] = x * y[0];
                args.more_data_goes_here_fn = true;
                Ok(())
            },
            |jj, x, _y, _multiplier, args: &mut Args| {
                args.n_jacobian_eval += 1;
                jj.reset();
                jj.put(0, 1, -x).unwrap();
                jj.put(1, 0, x).unwrap();
                args.more_data_goes_here_jj = true;
                Ok(())
            },
            HasJacobian::Yes,
            Some(2),
            None,
        );
        // analytical_solution:
        // y[0] = f64::cos(x * x / 2.0) - 2.0 * f64::sin(x * x / 2.0);
        // y[1] = 2.0 * f64::cos(x * x / 2.0) + f64::sin(x * x / 2.0);
        ode.init_mass_matrix(2, false).unwrap(); // diagonal mass matrix => OK, but not needed
        ode.mass_put(0, 0, 1.0).unwrap();
        ode.mass_put(1, 1, 1.0).unwrap();
        // call system function
        let x = 0.0;
        let y = Vector::new(2);
        let mut k = Vector::new(2);
        (ode.function)(&mut k, x, &y, &mut args).unwrap();
        // call jacobian function
        let mut jj = CooMatrix::new(2, 2, 2, None, false).unwrap();
        let m = 1.0;
        (ode.jacobian)(&mut jj, x, &y, m, &mut args).unwrap();
        // check
        println!("n_function_eval = {}", args.n_function_eval);
        println!("n_jacobian_eval = {}", args.n_jacobian_eval);
        assert_eq!(args.n_function_eval, 1);
        assert_eq!(args.n_jacobian_eval, 1);
        assert_eq!(args.more_data_goes_here_fn, true);
        assert_eq!(args.more_data_goes_here_jj, true);
    }
}
