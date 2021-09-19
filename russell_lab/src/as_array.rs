/// Defines a trait to handle 1D arrays
///
/// # Example
///
/// ```
/// use russell_lab::AsArray1D;
///
/// fn sum<'a, T, U>(array: &'a T) -> f64
/// where
///     T: AsArray1D<'a, U>,
///     U: 'a + Into<f64>,
/// {
///     let mut res = 0.0;
///     let m = array.size();
///     for i in 0..m {
///         res += array.at(i).into();
///     }
///     res
/// }
///
/// // heap-allocated 1D array (vector)
/// let x = vec![1.0, 2.0, 3.0];
/// assert_eq!(sum(&x), 6.0);
///
/// // heap-allocated 1D array (slice)
/// let y: &[f64] = &[10.0, 20.0, 30.0];
/// assert_eq!(sum(&y), 60.0);
///
/// // stack-allocated (fixed-size) 2D array
/// let z = [100.0, 200.0, 300.0];
/// assert_eq!(sum(&z), 600.0);
/// ```
///
/// # Review
///
/// ## Arrays
///
/// Arrays have type `[T; N]` and are a fixed-size list of elements of the same type.
/// Arrays are allocated on the **stack**.
/// Subscripts start at zero.
///
/// ```text
/// let x = [1.0, 2.0, 3.0]; // a: [f64; 3]
/// ```
///
/// Shorthand for initializing each element to 0.0:
///
/// ```text
/// let x = [0.0; 20]; // a: [f64; 20]
/// ```
///
/// ## Vectors
///
/// Vectors are dynamic or "growable" arrays of type `Vec<T>`.
/// Vectors are allocated on the **heap**.
/// We can create vectors with the vec! macro:
///
/// ```text
/// let v = vec![1.0, 2.0, 3.0]; // v: Vec<f64>
/// ```
///
/// ## Slices
///
/// Slices have type `&[T]` and are a reference to (or "view" into) an array.
/// Slices allow efficient access to a portion of an array without copying.
/// A slice is not created directly, but from an existing variable.
/// Slices have a length, can be mutable or not, and in behave like arrays:
///
/// ```text
/// let a = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let middle = &a[1..4]; // A slice of a: just the elements 1.0, 2.0, and 3.0
/// ```
pub trait AsArray1D<'a, U: 'a> {
    /// Returns the size of the array
    fn size(&self) -> usize;

    /// Returns the value at index i
    fn at(&self, i: usize) -> U;
}

/// Defines a heap-allocated 1D array (vector)
impl<'a, U: 'a> AsArray1D<'a, U> for Vec<U>
where
    U: 'a + Copy,
{
    fn size(&self) -> usize {
        self.len()
    }
    fn at(&self, i: usize) -> U {
        self[i]
    }
}

/// Defines a heap-allocated 1D array (slice)
impl<'a, U> AsArray1D<'a, U> for &'a [U]
where
    U: 'a + Copy,
{
    fn size(&self) -> usize {
        self.len()
    }
    fn at(&self, i: usize) -> U {
        self[i]
    }
}

/// Defines a stack-allocated (fixed-size) 1D array
impl<'a, U, const M: usize> AsArray1D<'a, U> for [U; M]
where
    U: 'a + Copy,
{
    fn size(&self) -> usize {
        self.len()
    }
    fn at(&self, i: usize) -> U {
        self[i]
    }
}

/// Defines a trait to handle 2D arrays
///
/// # Example
///
/// ```
/// use russell_lab::AsArray2D;
///
/// fn sum<'a, T, U>(array: &'a T) -> f64
/// where
///     T: AsArray2D<'a, U>,
///     U: 'a + Into<f64>,
/// {
///     let mut res = 0.0;
///     let (m, n) = array.size();
///     for i in 0..m {
///         for j in 0..n {
///             res += array.at(i, j).into();
///         }
///     }
///     res
/// }
///
/// // heap-allocated 2D array (vector of vectors)
/// const IGNORED: f64 = 123.456;
/// let a = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
///     vec![5.0, 6.0],
/// ];
/// assert_eq!(sum(&a), 21.0);
///
/// // heap-allocated 2D array (aka slice of slices)
/// let b: &[&[f64]] = &[
///     &[10.0, 20.0],
///     &[30.0, 40.0, IGNORED],
///     &[50.0, 60.0, IGNORED, IGNORED],
/// ];
/// assert_eq!(sum(&b), 210.0);
///
/// // stack-allocated (fixed-size) 2D array
/// let c = [
///     [100.0, 200.0],
///     [300.0, 400.0],
///     [500.0, 600.0],
/// ];
/// assert_eq!(sum(&c), 2100.0);
/// ```
pub trait AsArray2D<'a, U: 'a> {
    /// Returns the (m,n) size of the array
    fn size(&self) -> (usize, usize);

    /// Returns the value at (i,j) indices
    fn at(&self, i: usize, j: usize) -> U;
}

/// Defines a heap-allocated 2D array (vector of vectors)
///
/// # Notes
///
/// * The number of columns is defined by the first row
/// * The next rows must have at least the same number of columns as the first row
impl<'a, U: 'a> AsArray2D<'a, U> for Vec<Vec<U>>
where
    U: 'a + Copy,
{
    fn size(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }
    fn at(&self, i: usize, j: usize) -> U {
        self[i][j]
    }
}

/// Defines a heap-allocated 2D array (slice of slices)
///
/// # Notes
///
/// * The number of columns is defined by the first row
/// * The next rows must have at least the same number of columns as the first row
impl<'a, U> AsArray2D<'a, U> for &'a [&'a [U]]
where
    U: 'a + Copy,
{
    fn size(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }
    fn at(&self, i: usize, j: usize) -> U {
        self[i][j]
    }
}

/// Defines a stack-allocated (fixed-size) 2D array
impl<'a, U, const M: usize, const N: usize> AsArray2D<'a, U> for [[U; N]; M]
where
    U: 'a + Copy,
{
    fn size(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }
    fn at(&self, i: usize, j: usize) -> U {
        self[i][j]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{AsArray1D, AsArray2D};
    use std::fmt::Write;

    fn array_1d_str<'a, T, U>(array: &'a T) -> String
    where
        T: AsArray1D<'a, U>,
        U: 'a + std::fmt::Display,
    {
        let mut buf = String::new();
        let m = array.size();
        for i in 0..m {
            write!(&mut buf, "{},", array.at(i)).unwrap();
        }
        write!(&mut buf, "\n").unwrap();
        buf
    }

    fn array_2d_str<'a, T, U>(array: &'a T) -> String
    where
        T: AsArray2D<'a, U>,
        U: 'a + std::fmt::Display,
    {
        let mut buf = String::new();
        let (m, n) = array.size();
        for i in 0..m {
            for j in 0..n {
                write!(&mut buf, "{},", array.at(i, j)).unwrap();
            }
            write!(&mut buf, "\n").unwrap();
        }
        buf
    }

    #[test]
    fn as_array_1d_works() {
        // heap-allocated 1D array (vector)
        let x_data = vec![1.0, 2.0, 3.0];
        assert_eq!(array_1d_str(&x_data), "1,2,3,\n");

        // heap-allocated 1D array (slice)
        let y_data: &[f64] = &[10.0, 20.0, 30.0];
        assert_eq!(array_1d_str(&y_data), "10,20,30,\n");

        // stack-allocated (fixed-size) 2D array
        let z_data = [100.0, 200.0, 300.0];
        assert_eq!(array_1d_str(&z_data), "100,200,300,\n");
    }

    #[test]
    fn as_array_2d_works() {
        // heap-allocated 2D array (vector of vectors)
        const IGNORED: f64 = 123.456;
        let a_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, IGNORED, IGNORED, IGNORED],
            vec![5.0, 6.0],
        ];
        assert_eq!(
            array_2d_str(&a_data),
            "1,2,\n\
             3,4,\n\
             5,6,\n"
        );

        // heap-allocated 2D array (aka slice of slices)
        let b_data: &[&[f64]] = &[&[10.0, 20.0], &[30.0, 40.0, IGNORED], &[50.0, 60.0, IGNORED, IGNORED]];
        assert_eq!(
            array_2d_str(&b_data),
            "10,20,\n\
             30,40,\n\
             50,60,\n"
        );

        // stack-allocated (fixed-size) 2D array
        let c_data = [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]];
        assert_eq!(
            array_2d_str(&c_data),
            "100,200,\n\
             300,400,\n\
             500,600,\n"
        );
    }
}
