use super::SparseTriplet;
use crate::StrError;
use russell_lab::{format_nanoseconds, vec_norm, vec_update, Norm, Stopwatch, Vector};
use russell_openblas::{idamax, to_i32};
use std::fmt;

/// Verifies the linear system a ⋅ x = rhs
pub struct VerifyLinSys {
    pub max_abs_a: f64,      // max abs a
    pub max_abs_ax: f64,     // max abs a ⋅ x
    pub max_abs_diff: f64,   // max abs diff = a ⋅ x - rhs
    pub relative_error: f64, // max_abs_diff / (max_abs_a + 1)
    pub time_check: u128,    // elapsed time spent in the `new` method
}

impl VerifyLinSys {
    /// Creates a new verification dataset
    ///
    /// ```text
    /// diff : = |a ⋅ x - rhs|
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::{SparseTriplet, VerifyLinSys, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 4 non-zeros
    ///     let (neq, nnz) = (3, 4);
    ///     let mut trip = SparseTriplet::new(neq, nnz)?;
    ///     trip.put(0, 0, 1.0)?;
    ///     trip.put(0, 2, 4.0)?;
    ///     trip.put(1, 1, 2.0)?;
    ///     trip.put(2, 2, 3.0)?;
    ///
    ///     // check matrix
    ///     let mut a = Matrix::new(neq, neq);
    ///     trip.to_matrix(&mut a)?;
    ///     let correct_a = "┌       ┐\n\
    ///                      │ 1 0 4 │\n\
    ///                      │ 0 2 0 │\n\
    ///                      │ 0 0 3 │\n\
    ///                      └       ┘";
    ///     assert_eq!(format!("{}", a), correct_a);
    ///
    ///     // verify lin-sys
    ///     let x = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let rhs = Vector::from(&[5.0, 2.0, 3.0]);
    ///     let verify = VerifyLinSys::new(&trip, &x, &rhs, false)?;
    ///     assert_eq!(verify.max_abs_a, 4.0);
    ///     assert_eq!(verify.max_abs_ax, 5.0);
    ///     assert_eq!(verify.max_abs_diff, 0.0);
    ///     assert_eq!(verify.relative_error, 0.0);
    ///     assert!(verify.time_check > 0);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(trip: &SparseTriplet, x: &Vector, rhs: &Vector, triangular: bool) -> Result<Self, StrError> {
        if x.dim() != trip.neq || rhs.dim() != trip.neq {
            return Err("vector dimensions are incompatible");
        }
        // start stopwatch
        let mut sw = Stopwatch::new("");

        // compute max_abs_a
        let nnz = to_i32(trip.pos);
        let idx = idamax(nnz, &trip.values_aij, 1);
        let max_abs_a = f64::abs(trip.values_aij[idx as usize]);

        // compute max_abs_ax
        let mut ax = trip.mat_vec_mul(&x, triangular).unwrap(); // already checked
        let max_abs_ax = vec_norm(&ax, Norm::Max);

        // compute max_abs_diff
        vec_update(&mut ax, -1.0, &rhs).unwrap(); // ax := ax - rhs
        let max_abs_diff = vec_norm(&ax, Norm::Max);

        // compute relative_error
        let relative_error = max_abs_diff / (max_abs_a + 1.0);

        // stop stopwatch
        let time_check = sw.stop();

        // results
        Ok(VerifyLinSys {
            max_abs_a,
            max_abs_ax,
            max_abs_diff,
            relative_error,
            time_check,
        })
    }
}

impl fmt::Display for VerifyLinSys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\x20\x20\x20\x20\"maxAbsA\": {},\n\
             \x20\x20\x20\x20\"maxAbsAx\": {},\n\
             \x20\x20\x20\x20\"maxAbsDiff\": {:e},\n\
             \x20\x20\x20\x20\"relativeError\": {:e},\n\
             \x20\x20\x20\x20\"timeCheckNs\": {},\n\
             \x20\x20\x20\x20\"timeCheckStr\": \"{}\"",
            self.max_abs_a,
            self.max_abs_ax,
            self.max_abs_diff,
            self.relative_error,
            self.time_check,
            format_nanoseconds(self.time_check),
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{SparseTriplet, VerifyLinSys};
    use russell_lab::Vector;

    #[test]
    fn new_fails_on_wrong_vectors() {
        let trip = SparseTriplet::new(1, 1).unwrap();
        let x = Vector::new(2);
        let rhs = Vector::new(3);
        let x_wrong = Vector::new(3);
        let rhs_wrong = Vector::new(2);
        assert_eq!(
            VerifyLinSys::new(&trip, &x_wrong, &rhs, false).err(),
            Some("vector dimensions are incompatible")
        );
        assert_eq!(
            VerifyLinSys::new(&trip, &x, &rhs_wrong, false).err(),
            Some("vector dimensions are incompatible")
        );
    }

    #[test]
    fn new_works() {
        // | 1  3 -2 |
        // | 3  5  6 |
        // | 2  4  3 |
        let mut trip = SparseTriplet::new(3, 9).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(0, 1, 3.0).unwrap();
        trip.put(0, 2, -2.0).unwrap();
        trip.put(1, 0, 3.0).unwrap();
        trip.put(1, 1, 5.0).unwrap();
        trip.put(1, 2, 6.0).unwrap();
        trip.put(2, 0, 2.0).unwrap();
        trip.put(2, 1, 4.0).unwrap();
        trip.put(2, 2, 3.0).unwrap();
        let x = Vector::from(&[-15.0, 8.0, 2.0]);
        let rhs = Vector::from(&[5.0, 7.0, 8.0]);
        let verify = VerifyLinSys::new(&trip, &x, &rhs, false).unwrap();
        assert_eq!(verify.max_abs_a, 6.0);
        assert_eq!(verify.max_abs_ax, 8.0);
        assert_eq!(verify.max_abs_diff, 0.0);
        assert_eq!(verify.relative_error, 0.0);
        assert!(verify.time_check > 0);
    }

    #[test]
    fn display_trait_works() {
        let mut trip = SparseTriplet::new(2, 2).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(1, 1, 1.0).unwrap();
        let x = Vector::from(&[1.0, 1.0]);
        let rhs = Vector::from(&[1.0, 1.0]);
        let mut verify = VerifyLinSys::new(&trip, &x, &rhs, false).unwrap();
        verify.time_check = 0;
        let correct: &str = "\x20\x20\x20\x20\"maxAbsA\": 1,\n\
                             \x20\x20\x20\x20\"maxAbsAx\": 1,\n\
                             \x20\x20\x20\x20\"maxAbsDiff\": 0e0,\n\
                             \x20\x20\x20\x20\"relativeError\": 0e0,\n\
                             \x20\x20\x20\x20\"timeCheckNs\": 0,\n\
                             \x20\x20\x20\x20\"timeCheckStr\": \"0ns\"";
        assert_eq!(format!("{}", verify), correct);
    }
}
