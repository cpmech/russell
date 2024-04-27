use russell_lab::*;

fn main() -> Result<(), StrError> {
    // Example from:
    // https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/zgesv_example.html

    #[rustfmt::skip]
    let mut a = ComplexMatrix::from(&[
        [cpx!(-1.34, 2.55), cpx!( 0.28, 3.17), cpx!(-6.39,-2.20), cpx!( 0.72,-0.92)],
        [cpx!(-0.17,-1.41), cpx!( 3.31,-0.15), cpx!(-0.15, 1.34), cpx!( 1.29, 1.38)],
        [cpx!(-3.29,-2.39), cpx!(-1.91, 4.42), cpx!(-0.14,-1.35), cpx!( 1.72, 1.35)],
        [cpx!( 2.41, 0.39), cpx!(-0.56, 1.47), cpx!(-0.83,-0.69), cpx!(-1.96, 0.67)],
    ]);

    let mut b = ComplexVector::from(&[
        cpx!(26.26, 51.78),
        cpx!(6.43, -8.68),
        cpx!(-5.75, 25.31),
        cpx!(1.16, 2.57),
    ]);

    // solve b := x := A⁻¹ b
    complex_solve_lin_sys(&mut b, &mut a).unwrap();

    // print results
    println!("a (after) =\n{:.3}", a);
    println!("b (after) =\n{:.3}", b);

    // expected results
    let correct = ComplexVector::from(&[
        cpx!(1.0, 1.0),   //
        cpx!(2.0, -3.0),  //
        cpx!(-4.0, -5.0), //
        cpx!(0.0, 6.0),   //
    ]);
    println!("expected =\n{:.3}", correct);
    complex_vec_approx_eq(&b, &correct, 1e-13);
    Ok(())
}
