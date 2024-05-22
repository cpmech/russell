use russell_lab::{approx_eq, format_fortran, format_scientific};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_amplifier1t() {
    // get get ODE system
    let (system, x0, mut y0, mut args) = Samples::amplifier1t();

    // final x
    let x1 = 0.05;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-4, 1e-4, None).unwrap();

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // enable output of accepted steps
    solver
        .enable_output()
        .set_dense_h_out(0.001)
        .unwrap()
        .set_dense_recording(&[0, 4]);

    // solve the ODE system
    solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with radau5.f
    approx_eq(y0[0], -2.226517868073645E-02, 1e-10);
    approx_eq(y0[1], 3.068700099735197E+00, 1e-10);
    approx_eq(y0[2], 2.898340496450958E+00, 1e-9);
    approx_eq(y0[3], 2.033525366489690E+00, 1e-7);
    approx_eq(y0[4], -2.269179823457655E+00, 1e-7);
    approx_eq(stat.h_accepted, 7.791381954171996E-04, 1e-6);

    // compare dense output with Mathematica
    let n_dense = solver.out_dense_x().len();
    for i in 0..n_dense {
        approx_eq(solver.out_dense_x()[i], X_MATH[i], 1e-15);
        let diff0 = f64::abs(solver.out_dense_y(0)[i] - Y0_MATH[i]);
        let diff4 = f64::abs(solver.out_dense_y(4)[i] - Y4_MATH[i]);
        println!(
            "x ={:7.4}, y1and5 ={}{}, diff1and5 ={}{}",
            solver.out_dense_x()[i],
            format_fortran(solver.out_dense_y(0)[i]),
            format_fortran(solver.out_dense_y(4)[i]),
            format_scientific(diff0, 8, 1),
            format_scientific(diff4, 8, 1)
        );
        assert!(diff0 < 1e-4);
        assert!(diff4 < 1e-3);
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!(
        "y1to3 ={}{}{}",
        format_fortran(y0[0]),
        format_fortran(y0[1]),
        format_fortran(y0[2]),
    );
    println!("y4to5 ={}{}", format_fortran(y0[3]), format_fortran(y0[4]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 1511);
    assert_eq!(stat.n_jacobian, 126);
    assert_eq!(stat.n_factor, 166);
    assert_eq!(stat.n_lin_sol, 461);
    assert_eq!(stat.n_steps, 166);
    assert_eq!(stat.n_accepted, 127);
    assert_eq!(stat.n_rejected, 6);
    assert_eq!(stat.n_iterations_max, 5);
}

// Mathematica code
// eqs = {
//    Ue[t]/Subscript[R, 0] - U1[t]/Subscript[R, 0] + Subscript[C, 1] (U2'[t] - U1'[t]) == 0,
//    Ub/Subscript[R, 2] - U2[t] (1/Subscript[R, 1] + 1/Subscript[R, 2]) + Subscript[C, 1] (U1'[t] - U2'[t]) - (1 - \[Alpha]) f[U2[t] - U3[t]] == 0,
//    f[U2[t] - U3[t]] - U3[t]/Subscript[R, 3] - Subscript[C, 2] U3'[t] == 0,
//    Ub/Subscript[R, 4] - U4[t]/Subscript[R, 4] + Subscript[C, 3] (U5'[t] - U4'[t]) - \[Alpha] f[U2[t] - U3[t]] == 0,
//    -(U5[t]/Subscript[R, 5]) + Subscript[C, 3] (U4'[t] - U5'[t]) == 0
//    };
// ini = {
//    U1[0] == 0,
//    U2[0] == (Ub Subscript[R, 1])/(Subscript[R, 1] + Subscript[R, 2]),
//    U3[0] == (Ub Subscript[R, 1])/(Subscript[R, 1] + Subscript[R, 2]),
//    U4[0] == Ub,
//    U5[0] == 0
//    };
// sub1 = {Ue[t] -> A Sin[\[Omega] t], f[U2[t] - U3[t]] -> \[Beta] (Exp[(U2[t] - U3[t])/Uf] - 1)};
// sub2 = {Subscript[R, 0] -> R, Subscript[R, 1] -> S, Subscript[R, 2] -> S, Subscript[R, 3] -> S, Subscript[R, 4] -> S, Subscript[R, 5] -> S};
// sub3 = {\[Alpha] -> 0.99, \[Beta] -> 10^-6, A -> 0.4, \[Omega] -> 200 \[Pi], Ub -> 6, Uf -> 0.026, R -> 1000, S -> 9000};
// sub4 = {Subscript[C, 1] -> 1 10^-6, Subscript[C, 2] -> 2 10^-6, Subscript[C, 3] -> 3 10^-6};
// DAE = {eqs, ini} /. sub1 /. sub2 /. sub3 /. sub4;
// tMax = 0.05;
// sol = NDSolve[DAE, {U1, U2, U3, U4, U5}, {t, 0, tMax}, Method -> {"EquationSimplification" -> "Residual"}]
// xx = Table[0.001 k, {k, 0, 50}];
// yy1 = Table[Evaluate[U1[0.001 k] /. sol][[1]], {k, 0, 50}];
// yy5 = Table[Evaluate[U5[0.001 k] /. sol][[1]], {k, 0, 50}];
// NumberForm[xx, 10]
// NumberForm[yy1, 20]
// NumberForm[yy5, 20]

const X_MATH: [f64; 51] = [
    0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
    0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032,
    0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048,
    0.049, 0.05,
];

const Y0_MATH: [f64; 51] = [
    0.0,
    0.19163123318374464,
    0.32185367879057564,
    0.3335960606551303,
    0.2215635695367279,
    0.028220795080320382,
    -0.17260350843054062,
    -0.3052891397454663,
    -0.3208689996160992,
    -0.21128708362065052,
    -0.019646365460276864,
    0.18055684137277278,
    0.3126183722955318,
    0.3258766828442236,
    0.2151028597022245,
    0.022815953071239736,
    -0.17710807778952675,
    -0.3089263949913023,
    -0.32400009272978636,
    -0.21390596328872188,
    -0.02183398021705521,
    0.17872956293627157,
    0.3110919501345503,
    0.32460094432393466,
    0.21403514145654773,
    0.021922703389574156,
    -0.17785264281824917,
    -0.3095276078905207,
    -0.3245174337185357,
    -0.21433875579866524,
    -0.0221955020760429,
    0.178427591579212,
    0.31083968403700063,
    0.32439008481420134,
    0.2138585580647758,
    0.021774982739371514,
    -0.17797577774000792,
    -0.30962704584498146,
    -0.3246029836040378,
    -0.21441032868740578,
    -0.022255288051796426,
    0.1783776539665792,
    0.31079792296298414,
    0.3243550120745409,
    0.213829224711314,
    0.021750447911481622,
    -0.17799622886350802,
    -0.3096435634057654,
    -0.32461719161580643,
    -0.21442221448662638,
    -0.022265217137233992,
];

const Y4_MATH: [f64; 51] = [
    0.0,
    -2.688717471445241,
    -1.744875771519701,
    -0.7595061649951028,
    -0.1395497865888305,
    0.05034402841100122,
    0.06067077896330943,
    -0.407284001780353,
    -2.00356303745131,
    -2.77121845975808,
    -2.909034623992957,
    -2.3988776721015097,
    -1.4609137736344042,
    -0.49209021995023644,
    0.10853093997216126,
    0.2829974543994493,
    0.28987804919359145,
    -0.11445233279337079,
    -1.7735050363366236,
    -2.5504062030565406,
    -2.6947100730211817,
    -2.1904643464620728,
    -1.2580259788765373,
    -0.29482512838806,
    0.2996980482820194,
    0.4688423063296886,
    0.47239083114525215,
    0.0758904784124937,
    -1.5959310434406633,
    -2.3769733662474115,
    -2.5248624462491884,
    -2.0240604680324514,
    -1.0949668210442012,
    -0.1350714004629226,
    0.45611288978154885,
    0.6220924827404174,
    0.6228408471704201,
    0.22542205043554753,
    -1.45066522572926,
    -2.234520653243942,
    -2.3850899205464375,
    -1.886906532659037,
    -0.9604028944163061,
    -0.00304971461906603,
    0.5856876627762617,
    0.7492431432249377,
    0.7476605884948642,
    0.3482562599754222,
    -1.330336048141704,
    -2.116423719696165,
    -2.2691710591230145,
];
