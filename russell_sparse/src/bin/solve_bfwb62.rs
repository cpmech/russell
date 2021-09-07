use russell_chk::*;
use russell_lab::*;
use russell_sparse::*;

fn test_solver(kind: EnumSolverKind, verb_fact: bool, verb_sol: bool) {
    let mut sym_mirror = false;
    let mut tolerance = 1e-10;
    match kind {
        EnumSolverKind::Mmp => println!("Testing MMP solver\n"),
        EnumSolverKind::Umf => {
            println!("Testing UMF solver\n");
            sym_mirror = true;
            tolerance = 1e-15;
        }
    }

    let filepath = "./data/matrix_market/bfwb62.mtx".to_string();
    let trip = match read_matrix_market(&filepath, sym_mirror) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(read_matrix_market): {}", e);
            return;
        }
    };

    let mut config = ConfigSolver::new();
    config.set_solver_kind(kind);
    config.set_symmetry(EnumSymmetry::General);
    let mut solver = match Solver::new(config) {
        Ok(v) => v,
        Err(e) => {
            println!("FAIL(new solver): {}", e);
            return;
        }
    };

    match solver.initialize(&trip, false) {
        Err(e) => {
            println!("FAIL(initialize): {}", e);
            return;
        }
        _ => (),
    };

    match solver.factorize(verb_fact) {
        Err(e) => {
            println!("FAIL(factorize): {}", e);
            return;
        }
        _ => (),
    };

    let m = trip.dims().0;
    let mut x = Vector::new(m);
    let rhs = Vector::filled(m, 1.0);

    match solver.solve(&mut x, &rhs, verb_sol) {
        Err(e) => {
            println!("FAIL(solve again): {}", e);
            return;
        }
        _ => (),
    }

    println!("{}", trip);
    println!("{}", solver);

    let x_correct = &[
        -1.02570048377040759e+05,
        -1.08800418159713998e+05,
        -7.87848688672370918e+04,
        -6.12550631774225840e+04,
        -1.16611533352550643e+05,
        -8.91949258261042705e+04,
        -5.57584825429375196e+04,
        -3.37535346291137103e+04,
        -6.74159236038033268e+04,
        -5.61065283435406673e+04,
        -3.69561341372605821e+04,
        -2.67385128650871302e+04,
        -4.67349124343154253e+04,
        -4.18861901056076676e+04,
        -4.34393771636046149e+04,
        -1.11210692731083000e+04,
        -1.16010526640020762e+04,
        -4.31993854681577286e+04,
        -5.82924327463857844e+03,
        -2.42374319876188747e+04,
        -2.39432136682168457e+04,
        5.27355041927211232e+02,
        -1.24769422505944240e+04,
        -1.47005934749971748e+04,
        -4.95701604733381391e+04,
        -1.38451884223610182e+03,
        -1.57972501695015781e+04,
        -5.19172705598900066e+04,
        -4.99494464999615593e+04,
        -1.19678659380488571e+04,
        -1.56190973892000347e+04,
        -6.18809904102459404e+03,
        -1.05693761694190998e+04,
        -2.93013328593191145e+04,
        -9.15514607143451940e+03,
        -1.27058094439569140e+04,
        -1.93936053067287430e+04,
        -6.84836276779992295e+03,
        -1.07869319688850719e+04,
        -4.61926223513438963e+04,
        -1.99579363156562504e+04,
        -7.83564896339727693e+03,
        -6.37173129434054590e+03,
        -1.88075622025074267e+03,
        -8.71648101674354621e+03,
        -1.21683775603205122e+04,
        -1.91184585274694587e+03,
        -5.64233479410600103e+03,
        -6.47747230904305070e+03,
        -4.47783973932844674e+03,
        -9.82971659947420812e+03,
        -1.95594295004403466e+04,
        -2.09457080830507803e+04,
        -5.46686114796283709e+03,
        -5.28888244321673483e+03,
        -2.07962090362636227e+04,
        -9.33272319073228937e+03,
        1.96672299472196187e+02,
        -4.40813445835840230e+03,
        -4.87188111893421956e+03,
        -1.75640594405328884e+04,
        -1.77959327708208002e+04,
    ];
    assert_vec_approx_eq!(x.as_data(), x_correct, tolerance);
}

fn main() {
    println!("Running Mem Check\n");
    test_solver(EnumSolverKind::Mmp, false, false);
    test_solver(EnumSolverKind::Umf, false, false);
    println!("Done\n");
}
