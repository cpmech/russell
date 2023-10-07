// TODO

//     // set matrix
//     let mut a_full = Matrix::from(&[
//         [  4.0,  12.0, -16.0],
//         [ 12.0,  37.0, -43.0],
//         [-16.0, -43.0,  98.0],
//     ]);
//
//     // perform factorization
//     mat_cholesky(&mut a_full, false)?;
//
//     // compare with solution
//     // (note that upper part remains unchanged)
//     let l_correct = "┌             ┐\n\
//                      │   2  12 -16 │\n\
//                      │   6   1 -43 │\n\
//                      │  -8   5   3 │\n\
//                      └             ┘";
//     assert_eq!(format!("{}", l), l_correct);
//
//     // check if l⋅lᵀ == a
//     let l = &a_full; // alias
//     let m = a_full.nrow();
//     let mut l_lt = Matrix::new(m, m);
//     for i in 0..m {
//         for j in 0..m {
//             for k in 0..m {
//                 l_lt.add(i, j, l.get(i, k) * l.get(j, k));
//             }
//         }
//     }
//     let l_lt_correct = "┌             ┐\n\
//                         │   4  12 -16 │\n\
//                         │  12  37 -43 │\n\
//                         │ -16 -43  98 │\n\
//                         └             ┘";
//     assert_eq!(format!("{}", l), l_correct);

//     // set matrix
//     let a = Matrix::from(&[
//         [  4.0,  12.0, -16.0],
//         [ 12.0,  37.0, -43.0],
//         [-16.0, -43.0,  98.0],
//     ]);
//
//     // perform factorization
//     let m = a.nrow();
//     let mut l = Matrix::new(m, m);
//     mat_cholesky(&mut l, &a)?;
//
//     // compare with solution
//     let l_correct = "┌          ┐\n\
//                      │  2  0  0 │\n\
//                      │  6  1  0 │\n\
//                      │ -8  5  3 │\n\
//                      └          ┘";
//     assert_eq!(format!("{}", l), l_correct);

use russell_lab::*;

fn main() -> Result<(), StrError> {
    Ok(())
}
