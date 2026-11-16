//! Tests for the secondary-state callback protocol
//!
//! The protocol is:
//!
//! 1. A **fresh** step attempt copies the external state (`backup`) before it prepares to iterate.
//! 2. A **rejected** step restores the snapshot (`restore`) and retries without a new copy,
//!    because the retry starts again from the same step-start state.
//! 3. A step that fails **terminally** (an `Err` from the system, a failure status, a failed
//!    `polish`) restores the snapshot exactly once, after the failure.
//! 4. An **accepted** step is never restored, even when the caller fails right after `accept`.
//!
//! Every test records the callback events and asserts on the recorded sequence, and prints it, so
//! a failure says what actually ran.

use russell_lab::Vector;
use russell_nonlin::{Config, DeltaLambda, IniDir, Method, Output, Solver, Status, Stop, SupStatus, System, TrialPhase};
use russell_sparse::Sym;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Extra arguments of the test systems
#[derive(Default)]
struct Args {
    /// Makes the residual function fail, regardless of λ (set from an output callback)
    fail_now: bool,

    /// Makes the residual function fail for λ greater than this value
    fail_above: Option<f64>,
}

/// Records the events of the secondary-state callbacks
#[derive(Clone)]
struct Recorder {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl Recorder {
    fn new() -> Self {
        Recorder {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn handle(&self) -> Arc<Mutex<Vec<&'static str>>> {
        self.events.clone()
    }

    fn take(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

/// Builds G(u, λ) = u - a λ² - b λ + q u², wired to the recording callbacks
///
/// ```text
/// Gu = ∂G/∂u = 1 + 2 q u
/// Gλ = ∂G/∂λ = -2 a λ - b
/// ```
///
/// With `q = 10` and `a = 0, b = 1` the problem is nonlinear in `u` (the predictor is not exact,
/// so correctors run). With `q = 0`, `a = 1`, `b = 1e-3` it is a curved path with a nonzero
/// initial tangent, used by the arclength/polish tests.
///
/// When `retry_on_first_corrector` is set, the first corrector update returns
/// [`SupStatus::Retry`], which makes the solver reject the step and retry it.
fn one_equation_system(rec: &Recorder, q: f64, a: f64, b: f64, retry_on_first_corrector: bool) -> System<'static, Args> {
    let mut system = System::new(
        1,
        Some(1),
        Sym::No,
        move |gg, l, u, args: &mut Args| {
            if args.fail_now || args.fail_above.is_some_and(|limit| l > limit) {
                return Err("the test system failed at the requested λ");
            }
            gg[0] = u[0] - a * l * l - b * l + q * u[0] * u[0];
            Ok(())
        },
        move |ggu, ggl, l, u, _args: &mut Args| {
            ggu.put(0, 0, 1.0 + 2.0 * q * u[0]).unwrap();
            ggl[0] = -2.0 * a * l - b;
            Ok(())
        },
    )
    .unwrap();

    // wire the callbacks to the recorder
    let backup_events = rec.handle();
    let prepare_events = rec.handle();
    let restore_events = rec.handle();
    let update_events = rec.handle();
    let n_corrector = Arc::new(AtomicUsize::new(0));
    system
        .set_backup_secondary_state(move |_args: &mut Args| {
            backup_events.lock().unwrap().push("backup");
        })
        .set_prepare_to_iterate(move |_args: &mut Args| {
            prepare_events.lock().unwrap().push("prepare");
        })
        .set_restore_secondary_state(move |_args: &mut Args| {
            restore_events.lock().unwrap().push("restore");
        })
        .set_update_secondary_state(move |phase, _u0, _u1, _l0, _l1, _args: &mut Args| {
            update_events.lock().unwrap().push(match phase {
                TrialPhase::Predictor => "predictor",
                TrialPhase::Corrector => "corrector",
            });
            if retry_on_first_corrector && phase == TrialPhase::Corrector {
                if n_corrector.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(SupStatus::Retry);
                }
            }
            Ok(SupStatus::Continue)
        });
    system
}

/// Collapses consecutive repetitions, so a run of correctors counts as one event
fn collapse(events: &[&'static str]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for &event in events {
        if out.last() != Some(&event) {
            out.push(event);
        }
    }
    out
}

/// Counts the occurrences of an event
fn count(events: &[&'static str], event: &'static str) -> usize {
    events.iter().filter(|&&e| e == event).count()
}

/// 1. Fixed/list steps: one `backup` per step, before `prepare_to_iterate`, and no restore
#[test]
fn fixed_steps_back_up_once_per_step_before_prepare() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let config = Config::new();
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(0.3),
            &DeltaLambda::constant(0.1),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::Success);
    assert_eq!(solver.get_stats().n_accepted, 3);

    let events = rec.take();
    println!("fixed_steps_back_up_once_per_step_before_prepare: {events:?}");

    // each step is exactly backup → prepare → predictor → corrector (correctors collapsed)
    let mut expected = Vec::new();
    for _ in 0..3 {
        expected.extend_from_slice(&["backup", "prepare", "predictor", "corrector"]);
    }
    assert_eq!(collapse(&events), expected, "sequence: {events:?}");
    assert_eq!(count(&events, "backup"), 3, "sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 0, "sequence: {events:?}");
    assert_eq!(&events[0..3], &["backup", "prepare", "predictor"], "sequence: {events:?}");
}

/// 2. Auto step with a rejection: one backup before attempt 1, a restore on the reject, and no
///    second backup on the retry
#[test]
fn auto_retry_restores_and_does_not_back_up_again() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, true);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let config = Config::new();
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(2), &DeltaLambda::auto(1e-4), None)
        .unwrap();
    assert_eq!(status, Status::Success);
    let stats = solver.get_stats();
    assert_eq!((stats.n_steps, stats.n_accepted, stats.n_rejected), (2, 1, 1));

    let events = rec.take();
    println!("auto_retry_restores_and_does_not_back_up_again: {events:?}");

    assert_eq!(
        collapse(&events),
        vec![
            "backup",
            "prepare",
            "predictor",
            "corrector",
            "restore",
            "prepare",
            "predictor",
            "corrector"
        ],
        "sequence: {events:?}"
    );
    assert_eq!(count(&events, "backup"), 1, "sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 1, "sequence: {events:?}");
}

/// 3a. Terminal failure in flight: an `Err` from the system restores exactly once, after it
#[test]
fn terminal_error_in_flight_restores_exactly_once() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args {
        fail_above: Some(0.35),
        ..Default::default()
    };
    let config = Config::new();
    let mut solver = Solver::new(&config, system).unwrap();
    let res = solver.solve(
        &mut args,
        &mut u,
        &mut l,
        IniDir::Pos,
        Stop::MaxLambda(1.0),
        &DeltaLambda::constant(0.1),
        None,
    );
    assert_eq!(res.err(), Some("the test system failed at the requested λ"));
    assert_eq!(solver.get_stats().n_accepted, 3);

    let events = rec.take();
    println!("terminal_error_in_flight_restores_exactly_once: {events:?}");

    let mut expected = Vec::new();
    for _ in 0..3 {
        expected.extend_from_slice(&["backup", "prepare", "predictor", "corrector"]);
    }
    expected.extend_from_slice(&["backup", "prepare", "predictor", "restore"]);
    assert_eq!(collapse(&events), expected, "sequence: {events:?}");
    assert_eq!(count(&events, "backup"), 4, "sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 1, "sequence: {events:?}");
    assert_eq!(events.last(), Some(&"restore"), "sequence: {events:?}");
}

/// 3b. Terminal failure in flight: a `Status::ReachedMaxIterations` seen by the fixed-step loop
///     restores exactly once
#[test]
fn reached_max_iterations_in_flight_restores_exactly_once() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let mut config = Config::new();
    config.set_n_iteration_max(1);
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            &DeltaLambda::constant(0.1),
            None,
        )
        .unwrap();
    assert_eq!(status, Status::ReachedMaxIterations);
    assert_eq!(solver.get_stats().n_accepted, 0);

    let events = rec.take();
    println!("reached_max_iterations_in_flight_restores_exactly_once: {events:?}");

    assert_eq!(
        collapse(&events),
        vec!["backup", "prepare", "predictor", "restore"],
        "sequence: {events:?}"
    );
    assert_eq!(count(&events, "restore"), 1, "sequence: {events:?}");
    assert_eq!(events.last(), Some(&"restore"), "sequence: {events:?}");
}

/// 3c. Terminal failure in flight: a `Status::ContinuedFailure` restores exactly once
///
/// With `n_cont_failure_max = 1` the first failure ends the run before any `reject`, so the
/// terminal restore is the only one.
#[test]
fn continued_failure_in_flight_restores_exactly_once() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let mut config = Config::new();
    config.set_n_iteration_max(1).set_n_cont_failure_max(1);
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(2), &DeltaLambda::auto(1e-4), None)
        .unwrap();
    assert_eq!(status, Status::ContinuedFailure);
    assert_eq!(solver.get_stats().n_rejected, 0);

    let events = rec.take();
    println!("continued_failure_in_flight_restores_exactly_once: {events:?}");

    assert_eq!(
        collapse(&events),
        vec!["backup", "prepare", "predictor", "restore"],
        "sequence: {events:?}"
    );
    assert_eq!(count(&events, "restore"), 1, "sequence: {events:?}");
    assert_eq!(events.last(), Some(&"restore"), "sequence: {events:?}");
}

/// 3d. Terminal failure in flight: a failed arclength `polish` restores exactly once
///
/// The listed steps land short of the λ target, so `polish` runs; the output callback then makes
/// the residual function fail, which can only happen inside the polish step.
#[test]
fn failed_polish_in_flight_restores_exactly_once() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 0.0, 1.0, 1e-3, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    let mut solver = Solver::new(&config, system).unwrap();
    let mut out = Output::new();
    out.set_callback(|stats, _u, _l, _h, args: &mut Args| {
        if stats.n_accepted >= 3 {
            args.fail_now = true;
        }
        Ok(false)
    });
    let res = solver.solve(
        &mut args,
        &mut u,
        &mut l,
        IniDir::Pos,
        Stop::MaxLambda(1.0),
        &DeltaLambda::list(&[0.3, 0.3, 0.3]),
        Some(&mut out),
    );
    assert_eq!(res.err(), Some("the test system failed at the requested λ"));
    assert_eq!(solver.get_stats().n_accepted, 3);
    assert!(l < 1.0, "the listed steps must land short of the target so that polish runs; l = {l}");

    let events = rec.take();
    println!("failed_polish_in_flight_restores_exactly_once: {events:?}");

    let collapsed = collapse(&events);
    let mut expected = Vec::new();
    for _ in 0..3 {
        expected.extend_from_slice(&["backup", "prepare", "predictor", "corrector"]);
    }
    expected.extend_from_slice(&["backup", "prepare", "predictor", "restore"]);
    assert_eq!(collapsed, expected, "sequence: {events:?}");
    assert_eq!(count(&events, "backup"), 4, "3 listed steps + the polish; sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 1, "sequence: {events:?}");
    assert_eq!(events.last(), Some(&"restore"), "sequence: {events:?}");
}

/// 4a. Post-accept exit: a listed Δλ that is too small fails after `accept` and must not restore
#[test]
fn listed_step_error_after_accept_does_not_restore() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 10.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let config = Config::new();
    let mut solver = Solver::new(&config, system).unwrap();
    let res = solver.solve(
        &mut args,
        &mut u,
        &mut l,
        IniDir::Pos,
        Stop::Steps(3),
        &DeltaLambda::list(&[0.1, 1e-12]),
        None,
    );
    assert_eq!(res.err(), Some("requirement: ddl > 1e-10"));
    assert_eq!(solver.get_stats().n_accepted, 1);

    let events = rec.take();
    println!("listed_step_error_after_accept_does_not_restore: {events:?}");

    assert_eq!(
        collapse(&events),
        vec!["backup", "prepare", "predictor", "corrector"],
        "sequence: {events:?}"
    );
    assert_eq!(count(&events, "backup"), 1, "sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 0, "the accepted step must be preserved; sequence: {events:?}");
}

/// 4b. Post-accept exit: `Status::SmallStepsize` reached after `accept` must not restore
///
/// `h_max = 1e-12` caps the stepsize below the solver minimum after the first accepted step, so
/// the `SmallStepsize` check at the bottom of the automatic loop fires on an accepted step.
#[test]
fn small_stepsize_after_accept_does_not_restore() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 0.0, 0.0, 1.0, false);
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let mut config = Config::new();
    config.set_h_max(1e-12);
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(2), &DeltaLambda::auto(1e-4), None)
        .unwrap();
    assert_eq!(status, Status::SmallStepsize);
    assert_eq!(solver.get_stats().n_accepted, 1);

    let events = rec.take();
    println!("small_stepsize_after_accept_does_not_restore: {events:?}");

    assert_eq!(collapse(&events), vec!["backup", "prepare", "predictor"], "sequence: {events:?}");
    assert_eq!(count(&events, "restore"), 0, "the accepted step must be preserved; sequence: {events:?}");
}

/// 5. `polish` reset: after a `polish` that returns `Err`, the same solver is reusable and reaches
///    the target normally
///
/// The private `force_target`/`target_mode` flags are asserted directly by the unit test
/// `failed_polish_clears_the_targeting_flags` in `src/solver_arclength.rs`, because `#[cfg(test)]`
/// is not active for an integration test.
#[test]
fn solver_is_reusable_after_a_failed_polish() {
    let rec = Recorder::new();
    let system = one_equation_system(&rec, 0.0, 1.0, 1e-3, false);
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    let mut solver = Solver::new(&config, system).unwrap();

    // first solve: the polish fails
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let mut out = Output::new();
    out.set_callback(|stats, _u, _l, _h, args: &mut Args| {
        if stats.n_accepted >= 3 {
            args.fail_now = true;
        }
        Ok(false)
    });
    let res = solver.solve(
        &mut args,
        &mut u,
        &mut l,
        IniDir::Pos,
        Stop::MaxLambda(1.0),
        &DeltaLambda::list(&[0.3, 0.3, 0.3]),
        Some(&mut out),
    );
    assert_eq!(res.err(), Some("the test system failed at the requested λ"));
    println!("solver_is_reusable_after_a_failed_polish (first solve, failed polish): {:?}", rec.take());

    // second solve on the same solver: no targeting flag may have survived the failed polish
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = Args::default();
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            &DeltaLambda::list(&[0.3, 0.3, 0.3]),
            None,
        )
        .unwrap();
    let events = rec.take();
    println!("solver_is_reusable_after_a_failed_polish (second solve): {events:?}");
    assert_eq!(status, Status::Success);
    assert!((l - 1.0).abs() < 1e-12, "the reused solver must land on the target; l = {l}");
    assert_eq!(count(&events, "restore"), 0, "a successful run must not restore; sequence: {events:?}");
}
