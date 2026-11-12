use russell_lab::{Vector, approx_eq};
use russell_nonlin::{Config, DeltaLambda, IniDir, Method, NoArgs, Solver, Status, Stop, System};
use russell_sparse::Sym;
use std::sync::atomic::{AtomicUsize, Ordering};

/// G(u, λ) = u - a λ² - b λ, with Gu = 1 and Gλ = -2 a λ - b
///
/// The small linear term keeps the initial du/ds nonzero without changing the curvature that
/// drives the test corrections.
fn quadratic_path(a: f64, b: f64) -> System<'static, NoArgs> {
    System::new(
        1,
        Some(1),
        Sym::No,
        move |gg, l, u, _args: &mut NoArgs| {
            gg[0] = u[0] - a * l * l - b * l;
            Ok(())
        },
        move |ggu, ggl, l, _u, _args: &mut NoArgs| {
            ggu.put(0, 0, 1.0).unwrap();
            ggl[0] = -2.0 * (a * l) - b;
            Ok(())
        },
    )
    .unwrap()
}

/// A single wild Newton update is rejected, automatic stepping retries with `m_failure * h`, and
/// disabling the relative bound removes the rejection on the same fixture.
#[test]
fn single_wild_update_is_rejected_and_retried_with_half_step() {
    let h = 0.015;
    let system = quadratic_path(1000.0, 1.0e-6);

    // Fixed Δλ: the first correction is 0.225 > 10·0.015 = 0.15.
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    let mut solver = Solver::new(&config, system.clone()).unwrap();
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = 0;
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::constant(h), None)
        .unwrap();
    assert_eq!(status, Status::LargeDelta);
    assert_eq!(solver.get_stats().n_accepted, 0);

    // Automatic Δλ: the retry uses m_failure·h = 0.0075, and the correction becomes
    // 0.05625 < 10·0.0075 = 0.075. The attempt bound ends the run right after the retry.
    let mut config_auto = config.clone();
    config_auto.set_n_step_max(2).set_tg_control_enabled(false);
    let mut solver = Solver::new(&config_auto, system.clone()).unwrap();
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::auto(h), None)
        .unwrap();
    assert_eq!(status, Status::Success);
    let stats = solver.get_stats();
    assert_eq!(stats.n_rejected, 1);
    assert_eq!(stats.n_steps, 2);
    assert_eq!(stats.n_accepted, 1);
    approx_eq(stats.h_accepted, 0.5 * h, 1e-12);

    // Non-vacuous: f64::INFINITY disables the relative bound, so the first attempt is accepted.
    let mut config = Config::new();
    config.set_method(Method::Arclength).set_delta_max_allowed_h(f64::INFINITY);
    let mut solver = Solver::new(&config, system).unwrap();
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::constant(h), None)
        .unwrap();
    assert_eq!(status, Status::Success);
    assert_eq!(solver.get_stats().n_rejected, 0);
    assert_eq!(solver.get_stats().n_accepted, 1);
}

/// In `θ = 0` target mode `Δs_eff = |h| / (2 (dλ/ds)²)`, so a tiny `|dλ/ds|` does not produce a
/// false `LargeDelta` on the initial target attempt or on the halved retry.
#[test]
fn target_mode_uses_the_scaled_arclength_bound_on_the_retry() {
    let scale = 1.0e4;
    let mut system = System::new(
        1,
        Some(1),
        Sym::No,
        move |gg, l, u, _args: &mut NoArgs| {
            gg[0] = u[0] - scale * l;
            Ok(())
        },
        move |ggu, ggl, _l, _u, _args: &mut NoArgs| {
            ggu.put(0, 0, 1.0).unwrap();
            ggl[0] = -scale;
            Ok(())
        },
    )
    .unwrap();

    // The first secondary update performed during an iteration (do_backup = false) forces a
    // retry; that first step is already in target mode because λ is within the initial step of
    // the target.
    let calls = AtomicUsize::new(0);
    system.set_update_secondary_state(move |do_backup, _, _, _, _, _| {
        if !do_backup && calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(russell_nonlin::SupStatus::Retry)
        } else {
            Ok(russell_nonlin::SupStatus::Continue)
        }
    });

    let mut u = Vector::from(&[scale * 0.9999]);
    let mut l = 0.9999;
    let mut args = 0;
    let mut config = Config::new();
    config.set_method(Method::Arclength).set_n_step_max(2);
    let mut solver = Solver::new(&config, system).unwrap();
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut l,
            IniDir::Pos,
            Stop::MaxLambda(1.0),
            &DeltaLambda::auto(1e-3),
            None,
        )
        .unwrap();

    assert_eq!(status, Status::Success);
    let stats = solver.get_stats();
    assert_eq!(stats.n_rejected, 1);
    assert_eq!(stats.n_accepted, 2);
    assert_eq!(stats.n_steps, 3);
    approx_eq(l, 1.0, 1e-12);
    approx_eq(u[0], scale, 1e-8);
}

/// With `disable_rel_delta_analysis = true` the absolute ceiling and the non-finite delta check
/// still fire, while only `delta_rms` and `delta_converged` are disabled.
#[test]
fn trust_region_checks_work_when_relative_delta_analysis_is_disabled() {
    // LargeDelta from the absolute / Euclidean bounds.
    let h = 0.015;
    let system = quadratic_path(1000.0, 1.0e-6);
    let mut config = Config::new();
    config.set_method(Method::Arclength).set_disable_rel_delta_analysis(true);
    let mut solver = Solver::new(&config, system).unwrap();
    let mut u = Vector::from(&[0.0]);
    let mut l = 0.0;
    let mut args = 0;
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::constant(h), None)
        .unwrap();
    assert_eq!(status, Status::LargeDelta);

    // NanOrInfDelta: G stays finite (1e304) while Gu[0,0] = 1e-10 makes δu₀ overflow. The
    // second equation supplies a nonzero du/ds so the initial tangent is accepted, while the
    // first component remains decoupled from the normalization row.
    let a = 1.0e308;
    let gu = 1.0e-10;
    let system = System::new(
        2,
        Some(2),
        Sym::No,
        move |gg, l, u, _args: &mut NoArgs| {
            gg[0] = gu * u[0] - a * l * l;
            gg[1] = u[1] - l;
            Ok(())
        },
        move |ggu, ggl, l, _u, _args: &mut NoArgs| {
            ggu.put(0, 0, gu).unwrap();
            ggu.put(1, 1, 1.0).unwrap();
            ggl[0] = -2.0 * (a * l);
            ggl[1] = -1.0;
            Ok(())
        },
    )
    .unwrap();
    let mut config = Config::new();
    config
        .set_method(Method::Arclength)
        .set_bordering(false)
        .set_disable_rel_delta_analysis(true);
    let mut solver = Solver::new(&config, system).unwrap();
    let mut u = Vector::from(&[0.0, 0.0]);
    let mut l = 0.0;
    let status = solver
        .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(1), &DeltaLambda::constant(0.01), None)
        .unwrap();
    assert_eq!(status, Status::NanOrInfDelta);
}
