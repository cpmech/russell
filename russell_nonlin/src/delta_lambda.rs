use crate::{Stop, StrError, CONFIG_H_MIN};

/// Defines how Δλ is adjusted between steps
///
/// Three strategies are available:
///
/// * [`DeltaLambda::auto`] — automatic step-size control driven by Newton-Raphson
///   iteration statistics and/or the tangent vector angle. The solver selects and
///   adjusts Δλ adaptively to maintain a target convergence rate.
/// * [`DeltaLambda::constant`] — a fixed Δλ applied at every step. Useful for
///   simple problems or when a uniform spacing along λ is required.
/// * [`DeltaLambda::list`] — a prescribed sequence of Δλ values, one per step.
///   Useful when specific λ-levels must be visited exactly.
#[derive(Debug, Clone)]
pub struct DeltaLambda {
    /// Automatic Δλ
    pub(crate) auto: bool,

    /// Initial Δλ (or constant)
    pub(crate) ddl_ini: f64,

    /// List-based Δλ
    pub(crate) list: Vec<f64>,
}

impl DeltaLambda {
    /// New automatic Δλ
    ///
    /// The solver will adapt the stepsize automatically using convergence
    /// statistics and tangent vector angle information.
    ///
    /// # Arguments
    ///
    /// * `ddl_ini` -- the initial Δλ value; the solver adjusts from here
    pub fn auto(ddl_ini: f64) -> Self {
        Self {
            auto: true,
            ddl_ini,
            list: Vec::new(),
        }
    }

    /// New constant Δλ
    ///
    /// A fixed Δλ is applied at every step.
    ///
    /// # Arguments
    ///
    /// * `ddl` -- the fixed Δλ value to use at every step
    pub fn constant(ddl: f64) -> Self {
        Self {
            auto: false,
            ddl_ini: ddl,
            list: Vec::new(),
        }
    }

    /// New list-based Δλ
    ///
    /// # Panics
    ///
    /// Panics if the list is empty.
    pub fn list(list: &[f64]) -> Self {
        assert!(list.len() > 0);
        Self {
            auto: false,
            ddl_ini: list[0],
            list: list.to_vec(),
        }
    }

    /// Returns true if automatic Δλ
    ///
    /// Automatic mode means the solver adapts the stepsize based on
    /// convergence behavior rather than using a fixed or list-based Δλ.
    pub fn is_auto(&self) -> bool {
        self.auto
    }

    /// Calculates the initial stepsize Δλ
    pub(crate) fn ini(&self, stop: &Stop, l0: f64) -> Result<f64, StrError> {
        let mut ddl_ini = self.ddl_ini;
        match stop {
            Stop::MinCompU(..) => (),
            Stop::MaxCompU(..) => (),
            Stop::MaxNormU(..) => (),
            Stop::MinLambda(l1) => {
                ddl_ini = f64::min(ddl_ini, f64::abs(l0 - l1));
            }
            Stop::MaxLambda(l1) => {
                ddl_ini = f64::min(ddl_ini, f64::abs(l1 - l0));
            }
            Stop::Steps(..) => (),
        }
        if ddl_ini <= CONFIG_H_MIN {
            Err("requirement: ddl_ini > 1e-10")
        } else {
            Ok(ddl_ini)
        }
    }
}
