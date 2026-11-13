use super::{Config, Status};
use russell_lab::{Norm, Vector, vec_norm};

/// Calculates the iteration error
pub(crate) struct IterationError {
    /// Holds max(‖G‖∞,|N|)
    pub(crate) residual_max: f64,

    /// Holds max(‖δu‖∞,|δλ|)
    pub(crate) delta_max: f64,

    /// Holds ‖(δu, δλ)‖₂
    pub(crate) delta_l2: f64,

    /// RMS error on (δu,δλ)
    pub(crate) delta_rms: f64,

    /// Holds ‖(u_cand − u₀, λ_cand − λ₀)‖₂ for the cumulative step chord
    pub(crate) chord_l2: f64,

    /// Converged on (‖G‖∞,|N|)
    pub(crate) residual_converged: bool,

    /// Diverging on (‖G‖∞,|N|)
    pub(crate) residual_diverging: bool,

    /// Converged on RMS(δu,δλ)
    pub(crate) delta_converged: bool,

    /// Diverging on RMS(δu,δλ)
    pub(crate) delta_diverging: bool,

    /// Previous `residual_max`
    residual_max_prev: f64,

    /// Previous `delta_max`
    delta_max_prev: f64,

    /// Tolerance on max(‖G‖∞,|N|)
    tol_abs_residual: f64,

    /// Absolute tolerance on RMS(δu,δλ)
    tol_abs_delta: f64,

    /// Relative tolerance on RMS(δu,δλ)
    tol_rel_delta: f64,

    /// Allowed `delta_max`
    allowed_delta_max: f64,

    /// Allowed Euclidean step length for the single update and the cumulative chord
    allowed_step_l2: f64,

    /// Previous `residual_diverging`
    prev_residual_diverging: bool,

    /// Previous `delta_diverging`
    prev_delta_diverging: bool,

    /// Number of large max(‖δu‖∞,|δλ|)
    n_large_delta: usize,

    /// Number of continued `residual_diverging`
    n_continued_residual_divergence: usize,

    /// Number of continued `delta_diverging`
    n_continued_delta_divergence: usize,

    /// Maximum allowed number of continued divergence on ‖G,N‖∞
    n_cont_residual_divergence_max: usize,

    /// Maximum allowed number of continued divergence on ‖δu,δλ‖∞
    n_cont_delta_divergence_max: usize,

    /// Maximum allowed number of iterations
    n_iteration_max: usize,

    /// Disables the relative delta analysis
    disable_rel_delta_analysis: bool,

    /// Scaling vector for RMS(δu,δλ)
    scaling: Vector,
}

impl IterationError {
    /// Creates a new instance
    pub fn new(config: &Config, ndim: usize) -> Self {
        Self {
            residual_max: 0.0,
            delta_max: 0.0,
            delta_l2: 0.0,
            delta_rms: 0.0,
            chord_l2: 0.0,
            residual_converged: false,
            residual_diverging: false,
            delta_converged: false,
            delta_diverging: false,
            residual_max_prev: 0.0,
            delta_max_prev: 0.0,
            tol_abs_residual: config.tol_abs_residual,
            tol_abs_delta: config.tol_abs_delta,
            tol_rel_delta: config.tol_rel_delta,
            allowed_delta_max: config.delta_max_allowed,
            allowed_step_l2: f64::INFINITY,
            prev_residual_diverging: false,
            prev_delta_diverging: false,
            n_large_delta: 0,
            n_continued_residual_divergence: 0,
            n_continued_delta_divergence: 0,
            n_cont_residual_divergence_max: config.n_cont_residual_divergence_max,
            n_cont_delta_divergence_max: config.n_cont_delta_divergence_max,
            n_iteration_max: config.n_iteration_max,
            disable_rel_delta_analysis: config.disable_rel_delta_analysis,
            scaling: Vector::new(ndim + 1), // +1 for λ
        }
    }

    /// Resets the convergence flags and divergence counter
    pub fn reset(&mut self, u: &Vector, l: f64) {
        self.residual_converged = false;
        self.residual_diverging = false;
        self.delta_converged = false;
        self.delta_diverging = false;
        self.prev_residual_diverging = false;
        self.prev_delta_diverging = false;
        self.n_large_delta = 0;
        self.n_continued_residual_divergence = 0;
        self.n_continued_delta_divergence = 0;
        self.delta_l2 = 0.0;
        self.chord_l2 = 0.0;
        self.allowed_step_l2 = f64::INFINITY;
        let ndim = self.scaling.dim() - 1; // -1 due to λ
        for i in 0..ndim {
            self.scaling[i] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(u[i]);
        }
        self.scaling[ndim] = self.tol_abs_delta + self.tol_rel_delta * f64::abs(l);
    }

    /// Sets the allowed Euclidean step length for the arclength trust region
    ///
    /// The same bound applies to the single update `‖(δu, δλ)‖₂` and to the cumulative chord
    /// `‖(u_cand − u₀, λ_cand − λ₀)‖₂`. The default `f64::INFINITY` leaves the Natural method
    /// without a relative bound and is also set by [IterationError::reset].
    pub(crate) fn set_allowed_step_l2(&mut self, value: f64) {
        self.allowed_step_l2 = value;
    }

    /// Checks if the solution has converged based on any criterion
    pub fn converged(&self) -> bool {
        self.residual_converged || self.delta_converged
    }

    /// Analyzes convergence on max(‖G‖∞,|N|)
    ///
    /// Returns true if NaN or Inf is found, false otherwise (success)
    pub fn analyze_residual(&mut self, iteration: usize, gg: &Vector, nn: f64) -> bool {
        // compute max norm
        self.residual_max = f64::max(vec_norm(gg, Norm::Max), f64::abs(nn));

        // check for NaN or Inf
        if !self.residual_max.is_finite() {
            return true;
        }

        // check convergence
        self.residual_converged = self.residual_max < self.tol_abs_residual;

        // check if diverging
        self.prev_residual_diverging = self.residual_diverging;
        self.residual_diverging = if iteration == 0 {
            false
        } else {
            self.residual_max > self.residual_max_prev
        };

        // increment continued divergence counter
        if self.prev_residual_diverging && self.residual_diverging {
            self.n_continued_residual_divergence += 1;
        }

        // for subsequent iterations
        self.residual_max_prev = self.residual_max;

        // success, no NaN or Inf found
        false
    }

    /// Analyzes convergence on RMS(‖δu‖∞,|δλ|)
    ///
    /// `x` may be simply `δu` or the augmented vector `(δu, δλ)`
    ///
    /// Returns true if NaN or Inf is found, false otherwise (success)
    pub fn analyze_delta(&mut self, iteration: usize, x: &Vector) -> bool {
        // One pass accumulates everything the three measures need: the max-abs value for the
        // ℓ∞ norm, the sum of squares for the ℓ₂ norm, and the scaled sum for the RMS. The
        // 2-norm uses the same Blue scaling as `dnrm2` (which `Norm::Euc` calls), accumulated
        // in units of the running maximum with no second pass and no allocation. The norms are
        // unconditional because the absolute ceiling and the arclength trust region need them
        // even when the relative analysis is disabled.
        //
        // A NaN never satisfies `abs > delta_max`, so `delta_max` stays finite and it is the
        // sum of squares -- hence `delta_l2` -- that carries the NaN to the check below.
        let dim = x.dim();
        let mut delta_max = 0.0;
        let mut scale = 0.0;
        let mut ssq = 0.0;
        let mut sum_scaled = 0.0;
        for i in 0..dim {
            let xi = x[i];
            let abs = f64::abs(xi);
            if abs > delta_max {
                delta_max = abs;
            }
            let yi = xi / self.scaling[i];
            sum_scaled += yi * yi;

            // Accumulate in units of the running maximum so the squares neither overflow nor
            // underflow. The `abs != 0.0` guard only drops exact zeros, whose ratio would be
            // 0/0; a NaN fails it too, so `NaN/scale` still lands in `ssq` and is not lost.
            if abs > scale {
                ssq = 1.0 + ssq * (scale / abs) * (scale / abs);
                scale = abs;
            } else if abs != 0.0 {
                let ratio = abs / scale;
                ssq += ratio * ratio;
            }
        }
        self.delta_max = delta_max;
        self.delta_l2 = scale * f64::sqrt(ssq);

        // check for NaN or Inf
        if !self.delta_max.is_finite() || !self.delta_l2.is_finite() {
            return true;
        }

        // check if diverging
        self.prev_delta_diverging = self.delta_diverging;
        self.delta_diverging = if iteration == 0 { false } else { self.delta_max > self.delta_max_prev };

        // increment continued divergence counter
        if self.prev_delta_diverging && self.delta_diverging {
            self.n_continued_delta_divergence += 1;
        }

        // for subsequent iterations
        self.delta_max_prev = self.delta_max;

        // Gate only the relative analysis; the trust region still works when it is disabled. The
        // scaled sum above is simply ignored here instead of being skipped.
        if self.disable_rel_delta_analysis {
            self.delta_rms = 0.0;
            self.delta_converged = false;
            return false; // ok
        }

        // compute RMS(δu,δλ) from the scaled sum accumulated in the loop above
        self.delta_rms = f64::sqrt(sum_scaled / (dim as f64));

        // check convergence
        self.delta_converged = self.delta_rms <= 1.0;

        // success, no NaN or Inf found
        false
    }

    /// Analyzes the cumulative chord of the current step
    ///
    /// The chord is `‖(u_candidate − u₀, λ_candidate − λ₀)‖₂`, with `u_candidate = u_cur + x[0..ndim]` and
    /// `λ_candidate = l_cur + x[ndim]`, where `x = (δu, δλ)`. `u_candidate` is the corrected iterate after
    /// the current Newton update — the point the iteration would land on if the step were
    /// accepted — not the predictor of the step, which is a different point that the chord does
    /// not measure. It measures how far the corrected point has drifted from the step origin,
    /// not just the size of the last update, and it is what catches a run of individually small
    /// corrections that walk out of the trust region.
    ///
    /// The loop writes only scalars: no heap allocation is performed.
    ///
    /// Returns true if NaN or Inf is found, false otherwise (success)
    pub(crate) fn analyze_chord(&mut self, u0: &Vector, l0: f64, u_cur: &Vector, l_cur: f64, x: &Vector) -> bool {
        let ndim = u0.dim();
        let mut sum = 0.0;
        for i in 0..ndim {
            let du = u_cur[i] + x[i] - u0[i];
            sum += du * du;
        }
        let dl = l_cur + x[ndim] - l0;
        sum += dl * dl;
        self.chord_l2 = f64::sqrt(sum);
        !self.chord_l2.is_finite()
    }

    /// Checks the absolute and arclength trust-region bounds
    ///
    /// The checks are, in order:
    ///
    /// 1. the absolute ℓ∞ ceiling `delta_max > allowed_delta_max`;
    /// 2. the single-update Euclidean bound `delta_l2 > allowed_step_l2`;
    /// 3. the cumulative chord bound `chord_l2 > allowed_step_l2`.
    ///
    /// A violation increments `n_large_delta` once and returns [Status::LargeDelta]. The caller
    /// runs this before [IterationError::converged], so a step that crept outside the ball
    /// cannot be accepted on its delta convergence. When it returns None,
    /// [IterationError::capture_failures] cannot return `LargeDelta` for the same iteration:
    /// its own ℓ∞ test is the same one made here, and the call ordering keeps the counter from
    /// being incremented twice.
    pub(crate) fn check_large_delta(&mut self) -> Option<Status> {
        if self.delta_max > self.allowed_delta_max || self.delta_l2 > self.allowed_step_l2 || self.chord_l2 > self.allowed_step_l2 {
            self.n_large_delta += 1;
            return Some(Status::LargeDelta);
        }
        None
    }

    /// Captures eventual failures
    pub fn capture_failures(&mut self, iteration: usize) -> Status {
        // large (‖δu‖∞,|δλ|)
        //
        // The trust-region path calls check_large_delta first, so this arm only fires for a
        // caller that did not run it; in either case n_large_delta is incremented once.
        if self.delta_max > self.allowed_delta_max {
            self.n_large_delta += 1;
            return Status::LargeDelta;
        }

        // continued divergence on residual
        if self.n_continued_residual_divergence >= self.n_cont_residual_divergence_max {
            return Status::ContinuedResidualDivergence;
        }

        // continued divergence on delta
        if self.n_continued_delta_divergence >= self.n_cont_delta_divergence_max {
            return Status::ContinuedDeltaDivergence;
        }

        // max number of iterations reached
        if iteration == self.n_iteration_max - 1 {
            return Status::ReachedMaxIterations;
        }

        // no failure
        Status::Success
    }
}

#[cfg(test)]
mod tests {
    use super::IterationError;
    use crate::Config;
    use russell_lab::{Norm, Vector, vec_norm};

    /// The 2-norm the loop computes must agree with `Norm::Euc` for vectors in the normal range.
    #[test]
    fn delta_l2_matches_vec_norm_euc_in_range() {
        let config = Config::new();
        let vectors = [
            &[2.0, -2.0, 2.0, -2.0, -3.0][..],
            &[-3.0, 2.0, 1.0, 1.0, 1.0][..],
            &[1.0e-9, 0.0][..],
            &[0.1, -0.2, 0.3, 0.4][..],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..],
            &[1.0e6, 3.0, -7.0, 1.0e-3][..],
            &[5.0e-12, -2.5e-12, 7.0e-12, 1.0e-13][..],
        ];
        for data in vectors {
            let x = Vector::from(&data);
            let mut err = IterationError::new(&config, x.dim());
            let u = Vector::new(x.dim());
            err.reset(&u, 0.0);
            let nan_or_inf = err.analyze_delta(0, &x);
            assert!(!nan_or_inf);
            let reference = vec_norm(&x, Norm::Euc);
            let rel = f64::abs(err.delta_l2 - reference) / reference;
            assert!(
                rel < 1.0e-15,
                "delta_l2={:?} vec_norm={:?} rel={:e} for {:?}",
                err.delta_l2,
                reference,
                rel,
                data
            );
        }
    }

    /// Out-of-range entries must not overflow or underflow: `[1e200, 1e200]` keeps a finite norm
    /// (the is_finite guard must not fire) and `[1e-200, 1e-200]` must not underflow to zero.
    #[test]
    fn delta_l2_is_scaled_for_huge_and_tiny_entries() {
        let config = Config::new();

        let expected = f64::sqrt(2.0) * 1.0e200;
        let x = Vector::from(&[1.0e200, 1.0e200]);
        let mut err = IterationError::new(&config, x.dim());
        let u = Vector::new(x.dim());
        err.reset(&u, 0.0);
        let nan_or_inf = err.analyze_delta(0, &x);
        assert!(!nan_or_inf);
        assert_eq!(err.delta_max, 1.0e200);
        assert!(err.delta_l2.is_finite());
        assert_eq!(err.delta_l2, expected);

        let expected = f64::sqrt(2.0) * 1.0e-200;
        let x = Vector::from(&[1.0e-200, 1.0e-200]);
        let mut err = IterationError::new(&config, x.dim());
        let u = Vector::new(x.dim());
        err.reset(&u, 0.0);
        let nan_or_inf = err.analyze_delta(0, &x);
        assert!(!nan_or_inf);
        assert_eq!(err.delta_l2, expected);
    }
}
