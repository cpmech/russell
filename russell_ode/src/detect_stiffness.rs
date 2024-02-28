use crate::StrError;
use crate::{Params, Workspace};

/// Detects whether the problem becomes stiff or not
///
/// Returns `true` if stiffness has been detected.
pub(crate) fn detect_stiffness(work: &mut Workspace, params: &Params) -> Result<bool, StrError> {
    if work.bench.n_accepted <= params.stiffness.skip_first_n_accepted_step {
        return Ok(false);
    }
    if work.stiff_h_times_lambda > params.stiffness.h_times_lambda_max {
        work.stiff_n_detection_no = 0;
        work.stiff_n_detection_yes += 1;
        if work.stiff_n_detection_yes == params.stiffness.ratified_after_nstep {
            if params.stiffness.stop_with_error {
                return Err("stiffness detected");
            }
            Ok(true)
        } else {
            Ok(false)
        }
    } else {
        work.stiff_n_detection_no += 1;
        if work.stiff_n_detection_no == params.stiffness.ignored_after_nstep {
            work.stiff_n_detection_yes = 0;
        }
        Ok(false)
    }
}
