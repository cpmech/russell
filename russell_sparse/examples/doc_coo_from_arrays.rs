use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as CSC matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let nrow = 5;
    let ncol = 5;
    let row_indices = vec![0, /*dup*/ 0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4];
    let col_indices = vec![0, /*dup*/ 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4];
    let values = vec![
        1.0, /*dup*/ 1.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0,
    ];
    let sym = None;
    let coo = CooMatrix::from(nrow, ncol, row_indices, col_indices, values, sym)?;

    // covert to dense
    let a = coo.as_dense();
    let correct = "┌                ┐\n\
                   │  2  3  0  0  0 │\n\
                   │  3  0  4  0  6 │\n\
                   │  0 -1 -3  2  0 │\n\
                   │  0  0  1  0  0 │\n\
                   │  0  4  2  0  1 │\n\
                   └                ┘";
    assert_eq!(format!("{}", a), correct);
    Ok(())
}
