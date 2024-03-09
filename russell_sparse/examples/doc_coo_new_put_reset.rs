use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate the coefficient matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let mut coo = CooMatrix::new(5, 5, 13, Sym::No)?;
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(1, 0, 3.0)?;
    coo.put(0, 1, 3.0)?;
    coo.put(2, 1, -1.0)?;
    coo.put(4, 1, 4.0)?;
    coo.put(1, 2, 4.0)?;
    coo.put(2, 2, -3.0)?;
    coo.put(3, 2, 1.0)?;
    coo.put(4, 2, 2.0)?;
    coo.put(2, 3, 2.0)?;
    coo.put(1, 4, 6.0)?;
    coo.put(4, 4, 1.0)?;

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

    // reset
    coo.reset();

    // covert to dense
    let a = coo.as_dense();
    let correct = "┌           ┐\n\
                   │ 0 0 0 0 0 │\n\
                   │ 0 0 0 0 0 │\n\
                   │ 0 0 0 0 0 │\n\
                   │ 0 0 0 0 0 │\n\
                   │ 0 0 0 0 0 │\n\
                   └           ┘";
    assert_eq!(format!("{}", a), correct);

    // put again doubled values
    coo.put(0, 0, 2.0)?; // << duplicate
    coo.put(0, 0, 2.0)?; // << duplicate
    coo.put(1, 0, 6.0)?;
    coo.put(0, 1, 6.0)?;
    coo.put(2, 1, -2.0)?;
    coo.put(4, 1, 8.0)?;
    coo.put(1, 2, 8.0)?;
    coo.put(2, 2, -6.0)?;
    coo.put(3, 2, 2.0)?;
    coo.put(4, 2, 4.0)?;
    coo.put(2, 3, 4.0)?;
    coo.put(1, 4, 12.0)?;
    coo.put(4, 4, 2.0)?;

    // covert to dense
    let a = coo.as_dense();
    let correct = "┌                ┐\n\
                   │  4  6  0  0  0 │\n\
                   │  6  0  8  0 12 │\n\
                   │  0 -2 -6  4  0 │\n\
                   │  0  0  2  0  0 │\n\
                   │  0  8  4  0  2 │\n\
                   └                ┘";
    assert_eq!(format!("{}", a), correct);
    Ok(())
}
