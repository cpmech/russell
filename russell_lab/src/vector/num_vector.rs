use crate::{AsArray1D, StrError};
use num_traits::{cast, Num, NumCast};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::{self, Write};
use std::ops::{Index, IndexMut};

/// Implements a vector with numeric components for linear algebra
///
/// # Remarks
///
/// * NumVector implements the Index and IntoIterator traits (mutable or not),
///   thus, we can access components by indices or loop over the components
/// * NumVector has also methods to access the underlying data (mutable or not);
///   e.g., using `as_data()` and `as_mut_data()`.
/// * For faster computations, we recommend using the set of functions that
///   operate on Vectors and Matrices; e.g., `vec_add`, `vec_inner`, `vec_outer`,
///   `vec_copy`, `mat_vec_mul`, and others.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_add, NumVector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // create vector
///     let mut u = NumVector::<f64>::from(&[4.0, 9.0, 16.0, 25.0]);
///     assert_eq!(
///         format!("{}", u),
///         "┌    ┐\n\
///          │  4 │\n\
///          │  9 │\n\
///          │ 16 │\n\
///          │ 25 │\n\
///          └    ┘"
///     );
///
///     // create vector filled with zeros
///     let n = u.dim();
///     let v = NumVector::<f64>::filled(n, 10.0);
///     assert_eq!(
///         format!("{}", v),
///         "┌    ┐\n\
///          │ 10 │\n\
///          │ 10 │\n\
///          │ 10 │\n\
///          │ 10 │\n\
///          └    ┘"
///     );
///
///     // create a copy and change its components
///     let mut w = u.clone();
///     w.map(|x| f64::sqrt(x));
///     w[0] *= -1.0;
///     w[1] *= -1.0;
///     w[2] *= -1.0;
///     w[3] *= -1.0;
///     assert_eq!(
///         format!("{}", w),
///         "┌    ┐\n\
///          │ -2 │\n\
///          │ -3 │\n\
///          │ -4 │\n\
///          │ -5 │\n\
///          └    ┘"
///     );
///
///     // change the components
///     for x in &mut u {
///         *x = f64::sqrt(*x);
///     }
///
///     // add vectors
///     let mut z = NumVector::<f64>::new(n);
///     vec_add(&mut z, 1.0, &u, 1.0, &w)?;
///     println!("{}", z);
///     assert_eq!(
///         format!("{}", z),
///         "┌   ┐\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          │ 0 │\n\
///          └   ┘"
///     );
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    #[serde(bound(deserialize = "Vec<T>: Deserialize<'de>"))]
    data: Vec<T>,
}

impl<T> NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Creates a new (zeroed) vector
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::new(3);
    /// let correct = "┌   ┐\n\
    ///                │ 0 │\n\
    ///                │ 0 │\n\
    ///                │ 0 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn new(dim: usize) -> Self {
        NumVector {
            data: vec![T::zero(); dim],
        }
    }

    /// Creates new vector completely filled with the same value
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::filled(3, 4.0);
    /// let correct = "┌   ┐\n\
    ///                │ 4 │\n\
    ///                │ 4 │\n\
    ///                │ 4 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn filled(dim: usize, value: T) -> Self {
        NumVector { data: vec![value; dim] }
    }

    /// Creates a vector from data
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    ///
    /// // heap-allocated 1D array (vector)
    /// let u_data = vec![1.0, 2.0, 3.0];
    /// let u = NumVector::<f64>::from(&u_data);
    /// assert_eq!(
    ///     format!("{}", &u),
    ///     "┌   ┐\n\
    ///      │ 1 │\n\
    ///      │ 2 │\n\
    ///      │ 3 │\n\
    ///      └   ┘"
    /// );
    ///
    /// // heap-allocated 1D array (slice)
    /// let v_data: &[f64] = &[10.0, 20.0, 30.0];
    /// let v = NumVector::<f64>::from(&v_data);
    /// assert_eq!(
    ///     format!("{}", &v),
    ///     "┌    ┐\n\
    ///      │ 10 │\n\
    ///      │ 20 │\n\
    ///      │ 30 │\n\
    ///      └    ┘"
    /// );
    ///
    /// // stack-allocated (fixed-size) 2D array
    /// let w_data = [100.0, 200.0, 300.0];
    /// let w = NumVector::<f64>::from(&w_data);
    /// assert_eq!(
    ///     format!("{}", &w),
    ///     "┌     ┐\n\
    ///      │ 100 │\n\
    ///      │ 200 │\n\
    ///      │ 300 │\n\
    ///      └     ┘"
    /// );
    /// ```
    pub fn from<'a, S, U>(array: &'a S) -> Self
    where
        S: AsArray1D<'a, U>,
        U: 'a + Into<T>,
    {
        let dim = array.size();
        let mut data = vec![T::zero(); dim];
        for i in 0..dim {
            data[i] = array.at(i).into();
        }
        NumVector { data }
    }

    /// Returns a new vector that is initialized from a callback function (map)
    ///
    /// The function maps the index to the value, e.g., `|i| (i as f64)`
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::initialized(3, |i| (1 + 2 * i) as f64);
    /// assert_eq!(
    ///     format!("{}", u),
    ///     "┌   ┐\n\
    ///      │ 1 │\n\
    ///      │ 3 │\n\
    ///      │ 5 │\n\
    ///      └   ┘"
    /// );
    /// ```
    pub fn initialized<F>(dim: usize, function: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let data: Vec<T> = (0..dim).into_iter().map(function).collect();
        NumVector { data }
    }

    /// Returns evenly spaced numbers over a specified closed interval
    ///
    /// # Panics
    ///
    /// This function may panic if `count` cannot be cast as the number type of `start` and `stop`.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{NumVector, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let x = NumVector::<f64>::linspace(2.0, 3.0, 5)?;
    ///     let correct = "┌      ┐\n\
    ///                    │    2 │\n\
    ///                    │ 2.25 │\n\
    ///                    │  2.5 │\n\
    ///                    │ 2.75 │\n\
    ///                    │    3 │\n\
    ///                    └      ┘";
    ///     assert_eq!(format!("{}", x), correct);
    ///     let indices = NumVector::<usize>::linspace(0, 10, 4)?;
    ///     assert_eq!(*indices.as_data(), [0, 3, 6, 9]);
    ///     Ok(())
    /// }
    /// ```
    pub fn linspace(start: T, stop: T, count: usize) -> Result<Self, StrError> {
        let mut res = NumVector::new(count);
        if count == 0 {
            return Ok(res);
        }
        res.data[0] = start;
        if count == 1 {
            return Ok(res);
        }
        res.data[count - 1] = stop;
        if count == 2 {
            return Ok(res);
        }
        let den = cast::<usize, T>(count - 1).unwrap();
        let step = (stop - start) / den;
        for i in 1..count {
            let p = cast::<usize, T>(i).unwrap();
            res.data[i] = start + p * step;
        }
        Ok(res)
    }

    /// Returns a mapped linear-space; evenly spaced numbers modified by a function
    ///
    /// # Panics
    ///
    /// This function may panic if `count` cannot be cast as the number type of `start` and `stop`.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{NumVector, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let x = NumVector::<f64>::mapped_linspace(0.0, 4.0, 5, |v| v * v)?;
    ///     assert_eq!(
    ///         format!("{}", x),
    ///         "┌    ┐\n\
    ///          │  0 │\n\
    ///          │  1 │\n\
    ///          │  4 │\n\
    ///          │  9 │\n\
    ///          │ 16 │\n\
    ///          └    ┘",
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn mapped_linspace<F>(start: T, stop: T, count: usize, mut function: F) -> Result<Self, StrError>
    where
        F: FnMut(T) -> T,
    {
        let mut res = NumVector::new(count);
        if count == 0 {
            return Ok(res);
        }
        res.data[0] = function(start);
        if count == 1 {
            return Ok(res);
        }
        res.data[count - 1] = function(stop);
        if count == 2 {
            return Ok(res);
        }
        let den = cast::<usize, T>(count - 1).unwrap();
        let step = (stop - start) / den;
        for i in 1..count {
            let p = cast::<usize, T>(i).unwrap();
            res.data[i] = function(start + p * step);
        }
        Ok(res)
    }

    /// Returns the dimension (size) of this vector
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.dim(), 3);
    /// ```
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Fills this vector with a given value
    ///
    /// ```text
    /// u[i] := value
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::new(3);
    /// u.fill(8.8);
    /// let correct = "┌     ┐\n\
    ///                │ 8.8 │\n\
    ///                │ 8.8 │\n\
    ///                │ 8.8 │\n\
    ///                └     ┘";
    /// assert_eq!(format!("{}", u), correct);
    pub fn fill(&mut self, value: T) {
        self.data.iter_mut().map(|x| *x = value).count();
    }

    /// Returns an access to the underlying data
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.as_data(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn as_data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable access to the underlying data
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// let data = u.as_mut_data();
    /// data[1] = 2.2;
    /// assert_eq!(data, &[1.0, 2.2, 3.0]);
    /// ```
    #[inline]
    pub fn as_mut_data(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Returns the i-th component
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let u = NumVector::<f64>::from(&[1.0, 2.0]);
    /// assert_eq!(u.get(1), 2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the index is out-of-bounds.
    #[inline]
    pub fn get(&self, i: usize) -> T {
        assert!(i < self.data.len());
        self.data[i]
    }

    /// Change the i-th component
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0]);
    /// u.set(1, -2.0);
    /// let correct = "┌    ┐\n\
    ///                │  1 │\n\
    ///                │ -2 │\n\
    ///                └    ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the index is out-of-bounds.
    #[inline]
    pub fn set(&mut self, i: usize, value: T) {
        assert!(i < self.data.len());
        self.data[i] = value;
    }

    /// Copy another vector into this one
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0]);
    /// u.set_vector(&[-3.0, -4.0]);
    /// let correct = "┌    ┐\n\
    ///                │ -3 │\n\
    ///                │ -4 │\n\
    ///                └    ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    ///
    /// # Panics
    ///
    /// This function may panic if the other vector has a different length than this one
    pub fn set_vector(&mut self, other: &[T]) {
        assert_eq!(other.len(), self.data.len());
        self.data.copy_from_slice(other);
    }

    /// Applies a function over all components of this vector
    ///
    /// ```text
    /// u := map(function(ui))
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// u.map(|x| x * x);
    /// let correct = "┌   ┐\n\
    ///                │ 1 │\n\
    ///                │ 4 │\n\
    ///                │ 9 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn map<F>(&mut self, function: F)
    where
        F: Fn(T) -> T,
    {
        for elem in self.data.iter_mut() {
            *elem = function(*elem);
        }
    }

    /// Applies a function (with index) over all components of this vector
    ///
    /// ```text
    /// u := map(function(i, ui))
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// u.map_with_index(|i, x| x * x + (i as f64));
    /// let correct = "┌    ┐\n\
    ///                │  1 │\n\
    ///                │  5 │\n\
    ///                │ 11 │\n\
    ///                └    ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn map_with_index<F>(&mut self, function: F)
    where
        F: Fn(usize, T) -> T,
    {
        for (index, elem) in self.data.iter_mut().enumerate() {
            *elem = function(index, *elem);
        }
    }

    /// Returns a mapped version of this vector
    ///
    /// # Examples
    ///
    /// ```
    /// # use russell_lab::NumVector;
    /// let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
    /// let v = u.get_mapped(|v| 4.0 - v);
    /// u.set(1, 100.0);
    /// assert_eq!(
    ///     format!("{}", u),
    ///     "┌     ┐\n\
    ///      │   1 │\n\
    ///      │ 100 │\n\
    ///      │   3 │\n\
    ///      └     ┘",
    /// );
    /// assert_eq!(
    ///     format!("{}", v),
    ///     "┌   ┐\n\
    ///      │ 3 │\n\
    ///      │ 2 │\n\
    ///      │ 1 │\n\
    ///      └   ┘",
    /// );
    /// ```
    pub fn get_mapped<F>(&self, mut function: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        let mut data = self.data.to_vec();
        for elem in data.iter_mut() {
            *elem = function(*elem);
        }
        NumVector { data }
    }
}

impl<T> fmt::Display for NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize + fmt::Display,
{
    /// Generates a string representation of the NumVector
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::NumVector;
    /// let u = NumVector::<f64>::from(&[4.0, 3.0, 1.0, 0.0, -4.04]);
    /// assert_eq!(
    ///     format!("{}", u),
    ///     "┌       ┐\n\
    ///      │     4 │\n\
    ///      │     3 │\n\
    ///      │     1 │\n\
    ///      │     0 │\n\
    ///      │ -4.04 │\n\
    ///      └       ┘"
    /// );
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // handle empty vector
        if self.dim() == 0 {
            write!(f, "[]").unwrap();
            return Ok(());
        }
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.data.len() {
            let val = self.data[i];
            match f.precision() {
                Some(v) => write!(&mut buf, "{:.1$}", val, v).unwrap(),
                None => write!(&mut buf, "{}", val).unwrap(),
            }
            width = cmp::max(buf.chars().count(), width);
            buf.clear();
        }
        // draw vector
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width + 1).unwrap();
        for i in 0..self.data.len() {
            if i > 0 {
                write!(f, " │\n").unwrap();
            }
            write!(f, "│").unwrap();
            let val = self.data[i];
            match f.precision() {
                Some(v) => write!(f, "{:>1$.2$}", val, width, v).unwrap(),
                None => write!(f, "{:>1$}", val, width).unwrap(),
            }
        }
        write!(f, " │\n").unwrap();
        write!(f, "└{:1$}┘", " ", width + 1).unwrap();
        Ok(())
    }
}

/// Allows to access NumVector components using indices
///
/// # Examples
///
/// ```
/// use russell_lab::NumVector;
/// let u = NumVector::<f64>::from(&[-3.0, 1.2, 2.0]);
/// assert_eq!(u[0], -3.0);
/// assert_eq!(u[1],  1.2);
/// assert_eq!(u[2],  2.0);
/// ```
///
/// # Panics
///
/// The index function may panic if the index is out-of-bounds.
impl<T> Index<usize> for NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/// Allows to change NumVector components using indices
///
/// # Examples
///
/// ```
/// use russell_lab::NumVector;
/// let mut u = NumVector::<f64>::from(&[-3.0, 1.2, 2.0]);
/// u[0] -= 10.0;
/// u[1] += 10.0;
/// u[2] += 20.0;
/// assert_eq!(u[0], -13.0);
/// assert_eq!(u[1],  11.2);
/// assert_eq!(u[2],  22.0);
/// ```
///
/// # Panics
///
/// The index function may panic if the index is out-of-bounds.
impl<T> IndexMut<usize> for NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Allows to iterate over NumVector components (move version)
///
/// # Examples
///
/// ```
/// use russell_lab::NumVector;
/// let u = NumVector::<f64>::from(&[10.0, 20.0, 30.0]);
/// for (i, v) in u.into_iter().enumerate() {
///     assert_eq!(v, (10 * (i + 1)) as f64);
/// }
/// ```
impl<T> IntoIterator for NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Allows to iterate over NumVector components (borrow version)
///
/// # Examples
///
/// ```
/// use russell_lab::NumVector;
/// let u = NumVector::<f64>::from(&[10.0, 20.0, 30.0]);
/// let mut x = 10.0;
/// for v in &u {
///     assert_eq!(*v, x);
///     x += 10.0;
/// }
/// ```
impl<'a, T> IntoIterator for &'a NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Allows to iterate over NumVector components (mutable version)
///
/// # Examples
///
/// ```
/// use russell_lab::NumVector;
/// let mut u = NumVector::<f64>::from(&[10.0, 20.0, 30.0]);
/// let mut x = 100.0;
/// for v in &mut u {
///     *v *= 10.0;
///     assert_eq!(*v, x);
///     x += 100.0;
/// }
/// ```
impl<'a, T> IntoIterator for &'a mut NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

/// Allows accessing NumVector as an Array1D
impl<'a, T: 'a> AsArray1D<'a, T> for NumVector<T>
where
    T: Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    #[inline]
    fn size(&self) -> usize {
        self.dim()
    }
    #[inline]
    fn at(&self, i: usize) -> T {
        self[i]
    }
    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumVector;
    use crate::{vec_approx_eq, AsArray1D};
    use std::fmt::Write;

    fn pow2(x: f64) -> f64 {
        x * x
    }

    fn pow3(x: f64) -> f64 {
        x * x * x
    }

    fn pow3_plus_i(i: usize, x: f64) -> f64 {
        x * x * x + (i as f64)
    }

    #[test]
    fn new_vector_works() {
        let u = NumVector::<f64>::new(3);
        assert_eq!(u.data, &[0.0, 0.0, 0.0])
    }

    #[test]
    fn filled_works() {
        let u = NumVector::<f64>::filled(3, 5.0);
        assert_eq!(u.data, &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn from_works() {
        // heap-allocated 1D array (vector)
        let x_data = vec![1.0, 2.0, 3.0];
        let x = NumVector::<f64>::from(&x_data);
        assert_eq!(x.data, &[1.0, 2.0, 3.0]);

        // heap-allocated 1D array (slice)
        let y_data: &[f64] = &[10.0, 20.0, 30.0];
        let y = NumVector::<f64>::from(&y_data);
        assert_eq!(y.data, &[10.0, 20.0, 30.0]);

        // stack-allocated (fixed-size) 2D array
        let z_data = [100.0, 200.0, 300.0];
        let z = NumVector::<f64>::from(&z_data);
        assert_eq!(z.data, &[100.0, 200.0, 300.0]);
    }

    #[test]
    fn initialized_works() {
        let u = NumVector::<f64>::initialized(3, |i| i as f64);
        assert_eq!(u.data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn linspace_works() {
        let x = NumVector::<f64>::linspace(0.0, 1.0, 11).unwrap();
        let correct = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        vec_approx_eq(&x, correct, 1e-15);

        let x = NumVector::<f64>::linspace(2.0, 3.0, 0).unwrap();
        assert_eq!(x.data.len(), 0);

        let x = NumVector::<f64>::linspace(2.0, 3.0, 1).unwrap();
        assert_eq!(x.data.len(), 1);
        assert_eq!(x.data[0], 2.0);

        let x = NumVector::<f64>::linspace(2.0, 3.0, 2).unwrap();
        assert_eq!(x.data.len(), 2);
        assert_eq!(x.data[0], 2.0);
        assert_eq!(x.data[1], 3.0);

        let i = NumVector::<usize>::linspace(0, 10, 0).unwrap();
        assert_eq!(i.data, [] as [usize; 0]);
        let i = NumVector::<usize>::linspace(0, 10, 1).unwrap();
        assert_eq!(i.data, [0]);
        let i = NumVector::<usize>::linspace(0, 10, 2).unwrap();
        assert_eq!(i.data, [0, 10]);
        let i = NumVector::<usize>::linspace(0, 10, 3).unwrap();
        assert_eq!(i.data, [0, 5, 10]);
        let i = NumVector::<usize>::linspace(0, 10, 4).unwrap();
        assert_eq!(i.data, [0, 3, 6, 9]);
    }

    #[test]
    fn mapped_linspace_works() {
        let x = NumVector::<f64>::mapped_linspace(0.0, 4.0, 5, pow2).unwrap();
        assert_eq!(x.data, &[0.0, 1.0, 4.0, 9.0, 16.0]);

        let x = NumVector::<f64>::mapped_linspace(-1.0, 1.0, 5, f64::abs).unwrap();
        assert_eq!(x.data, &[1.0, 0.5, 0.0, 0.5, 1.0]);

        let x = NumVector::<f64>::mapped_linspace(2.0, 3.0, 0, pow3).unwrap();
        assert_eq!(x.data.len(), 0);

        let x = NumVector::<f64>::mapped_linspace(2.0, 3.0, 1, pow3).unwrap();
        assert_eq!(x.data.len(), 1);
        assert_eq!(x.data[0], 8.0);

        let x = NumVector::<f64>::mapped_linspace(2.0, 3.0, 2, pow3).unwrap();
        assert_eq!(x.data.len(), 2);
        assert_eq!(x.data[0], 8.0);
        assert_eq!(x.data[1], 27.0);

        let i = NumVector::<usize>::mapped_linspace(0, 10, 4, |v| v * 2).unwrap();
        assert_eq!(i.data, [0, 6, 12, 18]);
    }

    #[test]
    fn fill_works() {
        let mut u = NumVector::<f64>::from(&[6.0, 9.0, 12.0]);
        u.fill(7.7);
        let correct = &[7.7, 7.7, 7.7];
        assert_eq!(u.data, correct);
    }

    #[test]
    fn as_data_works() {
        let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        assert_eq!(u.as_data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_mut_data_works() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        let data = u.as_mut_data();
        data[1] = 2.2;
        assert_eq!(data, &[1.0, 2.2, 3.0]);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_wrong_index() {
        let u = NumVector::<f64>::new(1);
        u.get(1);
    }

    #[test]
    fn get_works() {
        let u = NumVector::<f64>::from(&[1.0, 2.0]);
        assert_eq!(u.get(0), 1.0);
        assert_eq!(u.get(1), 2.0);
    }

    #[test]
    #[should_panic]
    fn set_panics_on_wrong_index() {
        let mut u = NumVector::<f64>::new(1);
        u.set(1, 1.0);
    }

    #[test]
    fn set_works() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0]);
        u.set(0, -1.0);
        u.set(1, -2.0);
        assert_eq!(u.data, &[-1.0, -2.0]);
    }

    #[test]
    #[should_panic]
    fn set_vector_panics_on_wrong_len() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0]);
        u.set_vector(&[8.0, 9.0, 10.0]);
    }

    #[test]
    fn set_vector_works() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0]);
        u.set_vector(&[8.0, 9.0]);
        assert_eq!(u.data, &[8.0, 9.0]);
    }

    #[test]
    fn map_works() {
        let mut u = NumVector::<f64>::from(&[-1.0, -2.0, -3.0]);
        u.map(pow3);
        let correct = &[-1.0, -8.0, -27.0];
        assert_eq!(u.data, correct);
    }

    #[test]
    fn map_with_index_works() {
        let mut u = NumVector::<f64>::from(&[-1.0, -2.0, -3.0]);
        u.map_with_index(pow3_plus_i);
        let correct = &[-1.0, -7.0, -25.0];
        assert_eq!(u.data, correct);
    }

    #[test]
    fn get_mapped_works() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        let v = u.get_mapped(pow2);
        u.set(0, 0.11);
        u.set(1, 0.22);
        u.set(2, 0.33);
        assert_eq!(u.data, &[0.11, 0.22, 0.33]);
        assert_eq!(v.data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn display_works() {
        let x0 = NumVector::<f64>::new(0);
        assert_eq!(format!("{}", x0), "[]");
        let mut x = NumVector::<f64>::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        assert_eq!(
            format!("{}", x),
            "┌   ┐\n\
             │ 1 │\n\
             │ 2 │\n\
             │ 3 │\n\
             └   ┘"
        );
    }

    #[test]
    fn display_precision_works() {
        let u = NumVector::<f64>::from(&[1.012444, 2.034123, 3.05678]);
        assert_eq!(
            format!("{:.2}", u),
            "┌      ┐\n\
             │ 1.01 │\n\
             │ 2.03 │\n\
             │ 3.06 │\n\
             └      ┘"
        );
    }

    #[test]
    fn debug_works() {
        let u = NumVector::<f64>::new(1);
        assert_eq!(format!("{:?}", u), "NumVector { data: [0.0] }");
    }

    #[test]
    fn index_works() {
        let mut x = NumVector::<f64>::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn index_mut_works() {
        let mut x = NumVector::<f64>::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        x[0] += 10.0;
        x[1] += 20.0;
        x[2] += 30.0;
        assert_eq!(x.data, &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn into_iterator_works() {
        let mut x = NumVector::<f64>::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        // borrow x
        let mut i = 0_usize;
        for val in &x {
            assert_eq!(*val, (i + 1) as f64);
            i += 1;
        }
        // mut borrow x
        for val in &mut x {
            *val += 10.0;
        }
        assert_eq!(x.data, &[11.0, 12.0, 13.0]);
        // move x
        for (i, val) in x.into_iter().enumerate() {
            assert_eq!(val, (i + 11) as f64);
        }
    }

    #[test]
    fn into_iterator_bound_works() {
        fn vector_to_string<'a, T, U>(vector: &'a T) -> String
        where
            &'a T: IntoIterator<Item = U>,
            U: 'a + std::fmt::Display,
        {
            let mut buf = String::new();
            for (i, val) in vector.into_iter().enumerate() {
                write!(&mut buf, "({}:{}), ", i, val).unwrap();
            }
            buf
        }
        let mut x = NumVector::<f64>::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        let res = vector_to_string(&x);
        assert_eq!(res, "(0:1), (1:2), (2:3), ");
    }

    #[test]
    fn clone_and_serialize_work() {
        let mut u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        let u_copy = u.clone();
        u.set(0, 0.11);
        u.set(1, 0.22);
        u.set(2, 0.33);
        assert_eq!(u.data, &[0.11, 0.22, 0.33]);
        assert_eq!(u_copy.data, &[1.0, 2.0, 3.0]);

        let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        let mut cloned = u.clone();
        cloned[0] = -1.0;
        assert_eq!(
            format!("{}", u),
            "┌   ┐\n\
             │ 1 │\n\
             │ 2 │\n\
             │ 3 │\n\
             └   ┘"
        );
        assert_eq!(
            format!("{}", cloned),
            "┌    ┐\n\
             │ -1 │\n\
             │  2 │\n\
             │  3 │\n\
             └    ┘"
        );

        // serialize to json
        let json = serde_json::to_string(&u).unwrap();
        assert_eq!(json, r#"{"data":[1.0,2.0,3.0]}"#);

        // deserialize from json
        let from_json: NumVector<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(
            format!("{}", from_json),
            "┌   ┐\n\
             │ 1 │\n\
             │ 2 │\n\
             │ 3 │\n\
             └   ┘"
        );
    }

    fn array_1d_test<'a, T, U>(array: &'a T) -> String
    where
        T: AsArray1D<'a, U>,
        U: 'a + std::fmt::Display,
    {
        let mut buf = String::new();
        for i in 0..array.size() {
            write!(&mut buf, "{}", array.at(i)).unwrap();
        }
        buf
    }

    #[test]
    fn as_array_1d_works() {
        let u = NumVector::<i32>::from(&[1, 2, 3]);
        assert_eq!(array_1d_test(&u), "123");
    }
}
