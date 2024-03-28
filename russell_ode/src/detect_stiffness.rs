use crate::StrError;
use crate::{Params, Workspace};

/// Detects whether the problem becomes stiff or not
pub(crate) fn detect_stiffness(work: &mut Workspace, x: f64, params: &Params) -> Result<(), StrError> {
    work.stiff_detected = false;
    if work.stats.n_accepted <= params.stiffness.skip_first_n_accepted_step {
        return Ok(());
    }
    if work.stiff_h_times_rho > params.stiffness.h_times_rho_max {
        work.stiff_x_first_detect = f64::min(x, work.stiff_x_first_detect);
        work.stiff_n_detection_no = 0;
        work.stiff_n_detection_yes += 1;
        if work.stiff_n_detection_yes == params.stiffness.ratified_after_nstep {
            work.stiff_detected = true;
            if params.stiffness.stop_with_error {
                return Err("stiffness detected");
            }
        }
    } else {
        work.stiff_n_detection_no += 1;
        if work.stiff_n_detection_no == params.stiffness.ignored_after_nstep {
            work.stiff_x_first_detect = f64::MAX;
            work.stiff_n_detection_yes = 0;
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::detect_stiffness;
    use crate::{Method, Params, Workspace};

    #[test]
    fn detect_stiffness_works() {
        let mut params = Params::new(Method::DoPri5);
        let mut work = Workspace::new(Method::DoPri5);

        params.stiffness.skip_first_n_accepted_step = 10;
        detect_stiffness(&mut work, 0.0, &params).unwrap();
        assert_eq!(work.stiff_detected, false);

        work.stats.n_accepted = params.stiffness.skip_first_n_accepted_step + 1;
        work.stiff_h_times_rho = 3.25 + 0.01; // DoPri5
        work.stiff_n_detection_yes = params.stiffness.ratified_after_nstep - 1; // will add 1
        assert_eq!(
            detect_stiffness(&mut work, 0.0, &params).err(),
            Some("stiffness detected")
        );

        params.stiffness.stop_with_error = false;
        work.stiff_n_detection_yes = params.stiffness.ratified_after_nstep - 1; // will add 1
        detect_stiffness(&mut work, 0.0, &params).unwrap();
        assert_eq!(work.stiff_detected, true);
        assert_eq!(work.stiff_n_detection_no, 0);
        assert_eq!(work.stiff_n_detection_yes, params.stiffness.ratified_after_nstep);

        work.stiff_h_times_rho = 3.25 - 0.01; // DoPri5
        detect_stiffness(&mut work, 0.0, &params).unwrap();
        assert_eq!(work.stiff_detected, false);
        assert_eq!(work.stiff_n_detection_no, 1);
        assert_eq!(work.stiff_n_detection_yes, params.stiffness.ratified_after_nstep);

        work.stiff_n_detection_no = params.stiffness.ignored_after_nstep - 1; // will add 1
        detect_stiffness(&mut work, 0.0, &params).unwrap();
        assert_eq!(work.stiff_detected, false);
        assert_eq!(work.stiff_n_detection_no, params.stiffness.ignored_after_nstep);
        assert_eq!(work.stiff_n_detection_yes, 0);
    }
}
