use russell_lab::*;

fn main() -> Result<(), StrError> {
    // TODO

    // matrix a
    // #[rustfmt::skip]
    // let mut a_up = Matrix::from(&[
    //     [3.0, 0.0, -3.0, 0.0],
    //     [0.0, 3.0,  1.0, 2.0],
    //     [0.0, 0.0,  4.0, 1.0],
    //     [0.0, 0.0,  0.0, 3.0],
    // ]);
    // #[rustfmt::skip]
    // let mut a_lo = Matrix::from(&[
    //     [ 3.0, 0.0, 0.0, 0.0],
    //     [ 0.0, 3.0, 0.0, 0.0],
    //     [-3.0, 1.0, 4.0, 0.0],
    //     [ 0.0, 2.0, 1.0, 3.0],
    // ]);

    // // perform Cholesky factorization
    // let m = a_up.nrow();
    // let mut l = Matrix::new(m, m);
    // mat_cholesky(&mut l, &a_up)?;
    // TODO

    // let mut l_lt = Matrix::new(m, m);
    // for i in 0..m {
    //     for j in 0..m {
    //         for k in 0..m {
    //             l_lt.add(i, j, l.get(i, k) * l.get(j, k));
    //         }
    //     }
    // }

    // println!("{}", l_lt);

    Ok(())
}
