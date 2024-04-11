use russell_lab::{mat_cholesky, Matrix, StrError};

fn main() -> Result<(), StrError> {
    // set matrix
    let sym = 0.0;
    #[rustfmt::skip]
    let mut a = Matrix::from(&[
        [  4.0,   sym,   sym],
        [ 12.0,  37.0,   sym],
        [-16.0, -43.0,  98.0],
    ]);

    // perform factorization
    mat_cholesky(&mut a, false)?;

    // define alias (for convenience)
    let l = &a;

    // compare with solution
    let l_correct = "┌          ┐\n\
                     │  2  0  0 │\n\
                     │  6  1  0 │\n\
                     │ -8  5  3 │\n\
                     └          ┘";
    assert_eq!(format!("{}", l), l_correct);

    // check:  l ⋅ lᵀ = a
    let m = a.nrow();
    let mut l_lt = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                l_lt.add(i, j, l.get(i, k) * l.get(j, k));
            }
        }
    }
    let l_lt_correct = "┌             ┐\n\
                        │   4  12 -16 │\n\
                        │  12  37 -43 │\n\
                        │ -16 -43  98 │\n\
                        └             ┘";
    assert_eq!(format!("{}", l_lt), l_lt_correct);
    Ok(())
}
