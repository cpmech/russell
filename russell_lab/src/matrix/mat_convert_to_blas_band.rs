use super::Matrix;
use crate::StrError;

/// Converts a general (dense) matrix to the BLAS band matrix format
///
/// Only the band elements of a general banded matrix are stored in the BLAS band format.
/// The storage mode packs the matrix elements by columns such that each diagonal of the
/// matrix appears as a row in the packed array.
///
/// # Output
///
/// * `band` -- the BLAS band storage matrix. It must be an `(ml + mu + 1, n)` matrix
///
/// **Warning:** The output matrix `band` is not zeroed; thus, any existing value at the "out of band"
/// positions will be preserved.
///
/// # Input
///
/// * `dense` -- the general `(m, n)` matrix
/// * `ml` -- the lower diagonal dimension (does not count the diagonal)
/// * `mu` -- the upper diagonal dimension (does not count the diagonal)
///
/// See example below.
///
/// ```text
/// Input:
///                     =3
///                 ↓---mu--↓
///           ┌                                ┐
///           | 11  12  13  14   ·   ·   ·   · |
///         → │ 21  22  23  24  25   ·   ·   · |
/// ml = 2  → │ 31  32  33  34  35  36   ·   · |
///           │  ·  42  43  44  45  46  47   · |
///   dense = │  ·   ·  53  54  55  56  57  58 |
///           │  ·   ·   ·  64  65  66  67  68 |
///           │  ·   ·   ·   ·  75  76  77  78 |
///           │  ·   ·   ·   ·   ·  86  87  88 |
///           │  ·   ·   ·   ·   ·   ·  97  98 |
///           └                                ┘
/// ```
///
/// Imagine that you "push" the left-hand side down to make a "more rectangle" array, resulting in:
///
/// ```text
/// Output:
///         ┌                                ┐
///         │  ·   ·   ·  14  25  36  47  58 |
///         │  ·   ·  13  24  35  46  57  68 |
///  band = │  ·  12  23  34  45  56  67  78 |
///         │ 11  22  33  44  55  66  77  88 |
///         │ 21  32  43  54  65  76  87  98 |
///         │ 31  42  53  64  75  86  97   · |
///         └                                ┘
/// ```
///
/// # References
///
/// * <https://www.ibm.com/docs/en/essl/6.1?topic=representation-blas-general-band-storage-mode#am5gr_upbsm>
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let ____ = 0.0;
///     #[rustfmt::skip]
///     let dense = Matrix::from(&[
///         [11.0, 12.0, 13.0, ____, ____, ____, ____],
///         [21.0, 22.0, 23.0, 24.0, ____, ____, ____],
///         [____, 32.0, 33.0, 34.0, 35.0, ____, ____],
///         [____, ____, 43.0, 44.0, 45.0, 46.0, ____],
///         [____, ____, ____, 54.0, 55.0, 56.0, 57.0],
///     ]);
///     #[rustfmt::skip]
///     let band_correct = Matrix::from(&[
///         [____, ____, 13.0, 24.0, 35.0, 46.0, 57.0],
///         [____, 12.0, 23.0, 34.0, 45.0, 56.0, ____],
///         [11.0, 22.0, 33.0, 44.0, 55.0, ____, ____],
///         [21.0, 32.0, 43.0, 54.0, ____, ____, ____],
///     ]);
///     let n = dense.dims().1;
///     let (ml, mu) = (1, 2);
///     let mut band = Matrix::new(ml + mu + 1, n);
///     mat_convert_to_blas_band(&mut band, &dense, ml, mu).unwrap();
///     assert_eq!(band.as_data(), band_correct.as_data());
///     Ok(())
/// }
/// ```
pub fn mat_convert_to_blas_band(band: &mut Matrix, dense: &Matrix, ml: usize, mu: usize) -> Result<(), StrError> {
    let (m, n) = dense.dims();
    let (mb, nb) = band.dims();
    if mb != ml + mu + 1 || nb != n {
        return Err("the resulting matrix must be ml + mu + 1 by n");
    }
    for j in 0..n {
        let a = if j > mu { j - mu } else { 0 };
        let b = if j + ml + 1 < m { j + ml + 1 } else { m };
        for i in a..b {
            band.set(i + mu - j, j, dense.get(i, j));
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::mat_convert_to_blas_band;
    use crate::Matrix;

    #[test]
    fn mat_convert_to_blas_band_captures_errors() {
        let dense = Matrix::from(&[[1.0, 2.0], [3.0, 4.0]]);
        let n = dense.dims().1;
        let (ml, mu) = (1, 1);
        let mut band_wrong = Matrix::new(ml + mu + 0, n);
        assert_eq!(
            mat_convert_to_blas_band(&mut band_wrong, &dense, ml, mu).err(),
            Some("the resulting matrix must be ml + mu + 1 by n")
        );
        let mut band_wrong = Matrix::new(ml + mu + 0, n + 1);
        assert_eq!(
            mat_convert_to_blas_band(&mut band_wrong, &dense, ml, mu).err(),
            Some("the resulting matrix must be ml + mu + 1 by n")
        );
    }

    #[test]
    fn mat_convert_to_blas_band_works_m_gt_n() {
        #[rustfmt::skip]
        let dense = Matrix::from(&[
            [11.0, 12.0, 13.0, 14.0,  0.0,  0.0,  0.0,  0.0],
            [21.0, 22.0, 23.0, 24.0, 25.0,  0.0,  0.0,  0.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0,  0.0,  0.0],
            [ 0.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,  0.0],
            [ 0.0,  0.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            [ 0.0,  0.0,  0.0, 64.0, 65.0, 66.0, 67.0, 68.0],
            [ 0.0,  0.0,  0.0,  0.0, 75.0, 76.0, 77.0, 78.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0, 86.0, 87.0, 88.0],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 97.0, 98.0],
        ]);
        #[rustfmt::skip]
        let band_correct = Matrix::from(&[
            [ 0.0,  0.0,  0.0, 14.0, 25.0, 36.0, 47.0, 58.0],
            [ 0.0,  0.0, 13.0, 24.0, 35.0, 46.0, 57.0, 68.0],
            [ 0.0, 12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0],
            [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0],
            [21.0, 32.0, 43.0, 54.0, 65.0, 76.0, 87.0, 98.0],
            [31.0, 42.0, 53.0, 64.0, 75.0, 86.0, 97.0,  0.0],
        ]);
        let n = dense.dims().1;
        let (ml, mu) = (2, 3);
        let mut band = Matrix::new(ml + mu + 1, n);
        mat_convert_to_blas_band(&mut band, &dense, ml, mu).unwrap();
        assert_eq!(band.as_data(), band_correct.as_data());
    }

    #[test]
    fn mat_convert_to_blas_band_works_n_gt_m() {
        #[rustfmt::skip]
        let dense = Matrix::from(&[
            [11.0, 12.0, 13.0, 14.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [21.0, 22.0, 23.0, 24.0, 25.0,  0.0,  0.0,  0.0,  0.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0,  0.0,  0.0,  0.0],
            [ 0.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,  0.0,  0.0],
            [ 0.0,  0.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,  0.0],
            [ 0.0,  0.0,  0.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0],
            [ 0.0,  0.0,  0.0,  0.0, 75.0, 76.0, 77.0, 78.0, 79.0],
        ]);
        #[rustfmt::skip]
        let band_correct = Matrix::from(&[
            [ 0.0,  0.0,  0.0, 14.0, 25.0, 36.0, 47.0, 58.0, 69.0],
            [ 0.0,  0.0, 13.0, 24.0, 35.0, 46.0, 57.0, 68.0, 79.0],
            [ 0.0, 12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0,  0.0],
            [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0,  0.0,  0.0],
            [21.0, 32.0, 43.0, 54.0, 65.0, 76.0,  0.0,  0.0,  0.0],
            [31.0, 42.0, 53.0, 64.0, 75.0,  0.0,  0.0,  0.0,  0.0],
        ]);
        let n = dense.dims().1;
        let (ml, mu) = (2, 3);
        let mut band = Matrix::new(ml + mu + 1, n);
        mat_convert_to_blas_band(&mut band, &dense, ml, mu).unwrap();
        assert_eq!(band.as_data(), band_correct.as_data());
    }
}
