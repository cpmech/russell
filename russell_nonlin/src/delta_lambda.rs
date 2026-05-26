use crate::{Stop, StrError, CONFIG_H_MIN};

/// Defines how Δλ is adjusted
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
    pub fn auto(ddl_ini: f64) -> Self {
        Self {
            auto: true,
            ddl_ini,
            list: Vec::new(),
        }
    }

    /// New constant Δλ
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
