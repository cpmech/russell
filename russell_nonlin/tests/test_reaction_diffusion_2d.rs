#![allow(unused)]

use russell_lab::{array_approx_eq, mat_approx_eq, num_jacobian, Vector};
use russell_nonlin::{Config, Method, NoArgs, Solver, Stop, System};
use russell_pde::FdmLaplacian2d;
use russell_sparse::{CooMatrix, Sym};

const CHECK_JACOBIAN: bool = false;

#[test]
fn test_reaction_diffusion_2d() {
    // The nonlinear problem originates from the FDM discretization of the following equation:
    //
    // ∂²u   ∂²u
    // ——— + ——— + λ exp(u/(1+αu)) = 0
    // ∂x²   ∂y²
    //
    // on the unit square (1.0 × 1.0) with homogeneous boundary conditions.

    // constants
    const ALPHA: f64 = 0.1;

    // allocate the Laplacian operator
    let nn = 5;
    let mut fdm = FdmLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nn, nn).unwrap();
    fdm.set_homogeneous_boundary_conditions();

    // system function
    let dim = fdm.dim();
    let calc_gg = |gg: &mut Vector, l: f64, u: &Vector, lap: &mut FdmLaplacian2d| {
        let (aa, _) = lap.coefficient_matrix().unwrap();
        aa.mat_vec_mul(gg, 1.0, u).unwrap();
        for m in 0..dim {
            let dm = 1.0 + ALPHA * u[m];
            gg[m] += l * f64::exp(u[m] / dm);
        }
        Ok(())
    };

    // Jacobian function
    let prescribed = fdm.prescribed_flags().clone();
    let calc_ggu = |ggu: &mut CooMatrix, l: f64, u: &Vector, lap: &mut FdmLaplacian2d| {
        ggu.reset();
        for m in 0..dim {
            if !prescribed[m] {
                lap.loop_over_coef_mat_row(m, |n, amn| {
                    if !prescribed[n] {
                        ggu.put(m, n, amn).unwrap();
                    }
                });
            } else {
                ggu.put(m, m, 1.0).unwrap();
            }
            let dm = 1.0 + ALPHA * u[m];
            let gm = l * f64::exp(u[m] / dm);
            let hm = 1.0 / dm - ALPHA * u[m] / (dm * dm);
            ggu.put(m, m, gm * hm).unwrap();
        }
        Ok(())
    };

    // allocate nonlinear problem
    let mut system = System::new(dim, calc_gg).unwrap();

    // max number of non-zeros in the Jacobian (disregarding prescribed)
    let band = 5;
    let nnz = dim + dim * band; // 1 diagonal matrix + banded (laplacian) matrix
    let sym = Sym::No;

    // set the Jacobian function
    system.set_calc_ggu(Some(nnz), sym, calc_ggu).unwrap();

    // check Jacobian matrix
    if CHECK_JACOBIAN {
        let mut u0 = Vector::new(dim);
        u0.fill(1.0);
        let l0 = 1.0;
        let mut ggu = CooMatrix::new(dim, dim, nnz, sym).unwrap();
        calc_ggu(&mut ggu, l0, &u0, &mut fdm).unwrap();
        let ana = ggu.as_dense();
        let num = num_jacobian(dim, l0, &u0, 1.0, &mut fdm, calc_gg).unwrap();
        if nn <= 5 {
            println!("ana =\n{:.5}", ana);
            println!("num =\n{:.5}", num);
        }
        mat_approx_eq(&ana, &num, 1e-10);
    }
}
