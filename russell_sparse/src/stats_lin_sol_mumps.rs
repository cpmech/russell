use serde::{Deserialize, Serialize};

/// Holds the results of MUMPS error analysis ("stats")
///
/// See page 40 of MUMPS User's guide
///
/// MUMPS: computes the backward errors omega1 and omega2 (page 14):
///
/// ```text
///                                       |b - A · x_bar|ᵢ
/// omega1 = largest_scaled_residual_of ————————————————————
///                                     (|b| + |A| |x_bar|)ᵢ
///
///                                            |b - A · x_bar|ᵢ
/// omega2 = largest_scaled_residual_of ——————————————————————————————————
///                                     (|A| |x_approx|)ᵢ + ‖Aᵢ‖∞ ‖x_bar‖∞
///
/// where x_bar is the actual (approximate) solution returned by the linear solver
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMUMPS {
    /// Holds the infinite norm of the input matrix, RINFOG(4)
    pub inf_norm_a: f64,

    /// Holds the infinite norm of the computed solution, RINFOG(5)
    pub inf_norm_x: f64,

    /// Holds the scaled residual, RINFOG(6)
    pub scaled_residual: f64,

    /// Holds the backward error estimate omega1, RINFOG(7)
    pub backward_error_omega1: f64,

    /// Holds the backward error estimate omega2, RINFOG(8)
    pub backward_error_omega2: f64,

    /// Holds the normalized variation of the solution vector, RINFOG(9)
    ///
    /// Requires the full "stat" analysis.
    pub normalized_delta_x: f64,

    /// Holds the condition number1, RINFOG(10)
    ///
    /// Requires the full "stat" analysis.
    pub condition_number1: f64,

    /// Holds the condition number2, RINFOG(11)
    ///
    /// Requires the full "stat" analysis.
    pub condition_number2: f64,
}
