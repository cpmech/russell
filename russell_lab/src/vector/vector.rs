use crate::{AsArray1D, EnumVectorNorm};
use russell_openblas::*;
use std::cmp;
use std::fmt::{self, Write};
use std::ops::{Index, IndexMut};

/// Holds vector components and associated functions
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// Creates a new (zeroed) vector
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::new(3);
    /// let correct = "┌   ┐\n\
    ///                │ 0 │\n\
    ///                │ 0 │\n\
    ///                │ 0 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn new(dim: usize) -> Self {
        Vector { data: vec![0.0; dim] }
    }

    /// Creates new vector completely filled with the same value
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::filled(3, 4.0);
    /// let correct = "┌   ┐\n\
    ///                │ 4 │\n\
    ///                │ 4 │\n\
    ///                │ 4 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn filled(dim: usize, value: f64) -> Self {
        Vector { data: vec![value; dim] }
    }

    /// Creates a vector from data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// // heap-allocated 1D array (vector)
    /// let u_data = vec![1.0, 2.0, 3.0];
    /// let u = Vector::from(&u_data);
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
    /// let v = Vector::from(&v_data);
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
    /// let w = Vector::from(&w_data);
    /// assert_eq!(
    ///     format!("{}", &w),
    ///     "┌     ┐\n\
    ///      │ 100 │\n\
    ///      │ 200 │\n\
    ///      │ 300 │\n\
    ///      └     ┘"
    /// );
    /// ```
    pub fn from<'a, T, U>(array: &'a T) -> Self
    where
        T: AsArray1D<'a, U>,
        U: 'a + Into<f64>,
    {
        let dim = array.size();
        let mut data = vec![0.0; dim];
        for i in 0..dim {
            data[i] = array.at(i).into();
        }
        Vector { data }
    }

    /// Returns evenly spaced numbers over a specified closed interval
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let x = Vector::linspace(2.0, 3.0, 5);
    /// let correct = "┌      ┐\n\
    ///                │    2 │\n\
    ///                │ 2.25 │\n\
    ///                │  2.5 │\n\
    ///                │ 2.75 │\n\
    ///                │    3 │\n\
    ///                └      ┘";
    /// assert_eq!(format!("{}", x), correct);
    /// ```
    pub fn linspace(start: f64, stop: f64, count: usize) -> Self {
        let mut res = Vector::new(count);
        if count == 0 {
            return res;
        }
        res.data[0] = start;
        if count == 1 {
            return res;
        }
        res.data[count - 1] = stop;
        if count == 2 {
            return res;
        }
        let step = (stop - start) / ((count - 1) as f64);
        for i in 1..count {
            res.data[i] = start + (i as f64) * step;
        }
        res
    }

    /// Returns the dimension (size) of this vector
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.dim(), 3);
    /// ```
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Scales this vector
    ///
    /// ```text
    /// u := alpha * u
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    /// u.scale(0.5);
    /// let correct = "┌     ┐\n\
    ///                │ 0.5 │\n\
    ///                │   1 │\n\
    ///                │ 1.5 │\n\
    ///                └     ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn scale(&mut self, alpha: f64) {
        let n_i32: i32 = to_i32(self.data.len());
        dscal(n_i32, alpha, &mut self.data, 1);
    }

    /// Fills this vector with a given value
    ///
    /// ```text
    /// u[i] := value
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::new(3);
    /// u.fill(8.8);
    /// let correct = "┌     ┐\n\
    ///                │ 8.8 │\n\
    ///                │ 8.8 │\n\
    ///                │ 8.8 │\n\
    ///                └     ┘";
    /// assert_eq!(format!("{}", u), correct);
    pub fn fill(&mut self, value: f64) {
        self.data.iter_mut().map(|x| *x = value).count();
    }

    /// Returns an access to the underlying data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.as_data(), &[1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn as_data(&self) -> &Vec<f64> {
        &self.data
    }

    /// Returns a mutable access to the underlying data
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    /// let data = u.as_mut_data();
    /// data[1] = 2.2;
    /// assert_eq!(data, &[1.0, 2.2, 3.0]);
    /// ```
    #[inline]
    pub fn as_mut_data(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }

    /// Returns the i-th component
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::from(&[1.0, 2.0]);
    /// assert_eq!(u.get(1), 2.0);
    /// ```
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        assert!(i < self.data.len());
        self.data[i]
    }

    /// Change the i-th component
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0]);
    /// u.set(1, -2.0);
    /// let correct = "┌    ┐\n\
    ///                │  1 │\n\
    ///                │ -2 │\n\
    ///                └    ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, value: f64) {
        assert!(i < self.data.len());
        self.data[i] = value;
    }

    /// Executes the += operation on the i-th component
    ///
    /// ```text
    /// u_i += value
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0]);
    /// u.plus_equal(1, 0.22);
    /// let correct = "┌      ┐\n\
    ///                │ 1.00 │\n\
    ///                │ 2.22 │\n\
    ///                └      ┘";
    /// assert_eq!(format!("{:.2}", u), correct);
    /// ```
    #[inline]
    pub fn plus_equal(&mut self, i: usize, value: f64) {
        self.data[i] += value;
    }

    /// Applies a function over all components of this vector
    ///
    /// ```text
    /// u := apply(function(ui))
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    /// u.apply(|x| x * x);
    /// let correct = "┌   ┐\n\
    ///                │ 1 │\n\
    ///                │ 4 │\n\
    ///                │ 9 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn apply<F>(&mut self, function: F)
    where
        F: Fn(f64) -> f64,
    {
        for elem in self.data.iter_mut() {
            *elem = function(*elem);
        }
    }

    /// Applies a function (with index) over all components of this vector
    ///
    /// ```text
    /// u := apply(function(i, ui))
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    /// u.apply_with_index(|i, x| x * x + (i as f64));
    /// let correct = "┌    ┐\n\
    ///                │  1 │\n\
    ///                │  5 │\n\
    ///                │ 11 │\n\
    ///                └    ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn apply_with_index<F>(&mut self, function: F)
    where
        F: Fn(usize, f64) -> f64,
    {
        for (index, elem) in self.data.iter_mut().enumerate() {
            *elem = function(index, *elem);
        }
    }

    /// Returns a copy of this vector
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
    /// let u_copy = u.get_copy();
    /// u.set(1, 5.0);
    /// let u_correct = "┌   ┐\n\
    ///                  │ 1 │\n\
    ///                  │ 5 │\n\
    ///                  │ 3 │\n\
    ///                  └   ┘";
    /// let u_copy_correct = "┌   ┐\n\
    ///                       │ 1 │\n\
    ///                       │ 2 │\n\
    ///                       │ 3 │\n\
    ///                       └   ┘";
    /// assert_eq!(format!("{}", u), u_correct);
    /// assert_eq!(format!("{}", u_copy), u_copy_correct);
    /// ```
    pub fn get_copy(&self) -> Self {
        Vector {
            data: self.data.to_vec(),
        }
    }

    /// Returns the vector norm
    ///
    /// Computes one of:
    ///
    /// ```text
    /// One:  1-norm (taxicab or sum of abs values)
    ///
    ///       ‖u‖_1 := sum_i |uᵢ|
    ///
    /// Euc:  Euclidean-norm
    ///
    ///       ‖u‖_2 = sqrt(Σ_i uᵢ⋅uᵢ)
    ///
    /// Max:  max-norm (inf-norm)
    ///
    ///       ‖u‖_max = max_i ( |uᵢ| ) == ‖u‖_∞
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// let u = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0]);
    /// assert_eq!(u.norm(EnumVectorNorm::One), 11.0);
    /// assert_eq!(u.norm(EnumVectorNorm::Euc), 5.0);
    /// assert_eq!(u.norm(EnumVectorNorm::Max), 3.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self, kind: EnumVectorNorm) -> f64 {
        let n = to_i32(self.data.len());
        match kind {
            EnumVectorNorm::One => dasum(n, &self.data, 1),
            EnumVectorNorm::Euc => dnrm2(n, &self.data, 1),
            EnumVectorNorm::Max => {
                let idx = idamax(n, &self.data, 1);
                f64::abs(self.data[idx as usize])
            }
        }
    }
}

impl fmt::Display for Vector {
    /// Generates a string representation of the Vector
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::Vector;
    /// let u = Vector::from(&[4.0, 3.0, 1.0, 0.0, -4.04]);
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
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.data.len() {
            let val = self.data[i];
            match f.precision() {
                Some(v) => write!(&mut buf, "{:.1$}", val, v)?,
                None => write!(&mut buf, "{}", val)?,
            }
            width = cmp::max(buf.chars().count(), width);
            buf.clear();
        }
        // draw vector
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width + 1)?;
        for i in 0..self.data.len() {
            if i > 0 {
                write!(f, " │\n")?;
            }
            write!(f, "│")?;
            let val = self.data[i];
            match f.precision() {
                Some(v) => write!(f, "{:>1$.2$}", val, width, v)?,
                None => write!(f, "{:>1$}", val, width)?,
            }
        }
        write!(f, " │\n")?;
        write!(f, "└{:1$}┘", " ", width + 1)?;
        Ok(())
    }
}

/// Allows to access Vector components using indices
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// let u = Vector::from(&[-3.0, 1.2, 2.0]);
/// assert_eq!(u[0], -3.0);
/// assert_eq!(u[1],  1.2);
/// assert_eq!(u[2],  2.0);
/// ```
impl Index<usize> for Vector {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/// Allows to change Vector components using indices
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// let mut u = Vector::from(&[-3.0, 1.2, 2.0]);
/// u[0] -= 10.0;
/// u[1] += 10.0;
/// u[2] += 20.0;
/// assert_eq!(u[0], -13.0);
/// assert_eq!(u[1],  11.2);
/// assert_eq!(u[2],  22.0);
/// ```
impl IndexMut<usize> for Vector {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Allows to iterate over Vector components (move version)
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// let u = Vector::from(&[10.0, 20.0, 30.0]);
/// for (i, v) in u.into_iter().enumerate() {
///     assert_eq!(v, (10 * (i + 1)) as f64);
/// }
/// ```
impl IntoIterator for Vector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Allows to iterate over Vector components (borrow version)
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// let u = Vector::from(&[10.0, 20.0, 30.0]);
/// let mut x = 10.0;
/// for v in &u {
///     assert_eq!(*v, x);
///     x += 10.0;
/// }
/// ```
impl<'a> IntoIterator for &'a Vector {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Allows to iterate over Vector components (mutable version)
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// let mut u = Vector::from(&[10.0, 20.0, 30.0]);
/// let mut x = 100.0;
/// for v in &mut u {
///     *v *= 10.0;
///     assert_eq!(*v, x);
///     x += 100.0;
/// }
/// ```
impl<'a> IntoIterator for &'a mut Vector {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_vector_works() {
        let u = Vector::new(3);
        assert_eq!(u.data, &[0.0, 0.0, 0.0])
    }

    #[test]
    fn filled_works() {
        let u = Vector::filled(3, 5.0);
        assert_eq!(u.data, &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn from_works() {
        // heap-allocated 1D array (vector)
        let x_data = vec![1.0, 2.0, 3.0];
        let x = Vector::from(&x_data);
        assert_eq!(x.data, &[1.0, 2.0, 3.0]);

        // heap-allocated 1D array (slice)
        let y_data: &[f64] = &[10.0, 20.0, 30.0];
        let y = Vector::from(&y_data);
        assert_eq!(y.data, &[10.0, 20.0, 30.0]);

        // stack-allocated (fixed-size) 2D array
        let z_data = [100.0, 200.0, 300.0];
        let z = Vector::from(&z_data);
        assert_eq!(z.data, &[100.0, 200.0, 300.0]);
    }

    #[test]
    fn linspace_works() {
        let x = Vector::linspace(0.0, 1.0, 11);
        let correct = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        assert_vec_approx_eq!(x.data, correct, 1e-15);
    }

    #[test]
    fn linspace_0_works() {
        let x = Vector::linspace(2.0, 3.0, 0);
        assert_eq!(x.data.len(), 0);
    }

    #[test]
    fn linspace_1_works() {
        let x = Vector::linspace(2.0, 3.0, 1);
        assert_eq!(x.data.len(), 1);
        assert_eq!(x.data[0], 2.0);
    }

    #[test]
    fn linspace_2_works() {
        let x = Vector::linspace(2.0, 3.0, 2);
        assert_eq!(x.data.len(), 2);
        assert_eq!(x.data[0], 2.0);
        assert_eq!(x.data[1], 3.0);
    }

    #[test]
    fn scale_works() {
        let mut u = Vector::from(&[6.0, 9.0, 12.0]);
        u.scale(1.0 / 3.0);
        let correct = &[2.0, 3.0, 4.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn fill_works() {
        let mut u = Vector::from(&[6.0, 9.0, 12.0]);
        u.fill(7.7);
        let correct = &[7.7, 7.7, 7.7];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn as_data_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        assert_eq!(u.as_data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn as_mut_data_works() {
        let mut u = Vector::from(&[1.0, 2.0, 3.0]);
        let data = u.as_mut_data();
        data[1] = 2.2;
        assert_eq!(data, &[1.0, 2.2, 3.0]);
    }

    #[test]
    #[should_panic]
    fn get_panics_on_wrong_index() {
        let u = Vector::new(1);
        u.get(1);
    }

    #[test]
    fn get_works() {
        let u = Vector::from(&[1.0, 2.0]);
        assert_eq!(u.get(0), 1.0);
        assert_eq!(u.get(1), 2.0);
    }

    #[test]
    #[should_panic]
    fn set_panics_on_wrong_index() {
        let mut u = Vector::new(1);
        u.set(1, 1.0);
    }

    #[test]
    fn set_works() {
        let mut u = Vector::from(&[1.0, 2.0]);
        u.set(0, -1.0);
        u.set(1, -2.0);
        assert_eq!(u.data, &[-1.0, -2.0]);
    }

    #[test]
    fn plus_equal_works() {
        let mut u = Vector::from(&[1.0, 2.0]);
        u.plus_equal(0, 0.11);
        u.plus_equal(1, 0.22);
        assert_eq!(u.data, &[1.11, 2.22]);
    }

    #[test]
    fn apply_works() {
        let mut u = Vector::from(&[-1.0, -2.0, -3.0]);
        u.apply(|x| x * x * x);
        let correct = &[-1.0, -8.0, -27.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn apply_with_index_works() {
        let mut u = Vector::from(&[-1.0, -2.0, -3.0]);
        u.apply_with_index(|i, x| x * x * x + (i as f64));
        let correct = &[-1.0, -7.0, -25.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn get_copy_works() {
        #[rustfmt::skip]
        let mut u = Vector::from( &[1.0, 2.0, 3.0]);
        let u_copy = u.get_copy();
        u.set(0, 0.11);
        u.set(1, 0.22);
        u.set(2, 0.33);
        assert_eq!(u.data, &[0.11, 0.22, 0.33]);
        assert_eq!(u_copy.data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn norm_works() {
        let u = Vector::from(&[-3.0, 2.0, 1.0, 1.0, 1.0]);
        assert_eq!(u.norm(EnumVectorNorm::One), 8.0);
        assert_eq!(u.norm(EnumVectorNorm::Euc), 4.0);
        assert_eq!(u.norm(EnumVectorNorm::Max), 3.0);
    }

    #[test]
    fn display_works() {
        let mut x = Vector::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        let correct: &str = "┌   ┐\n\
                             │ 1 │\n\
                             │ 2 │\n\
                             │ 3 │\n\
                             └   ┘";
        assert_eq!(format!("{}", x), correct);
    }

    #[test]
    fn display_precision_works() {
        let u = Vector::from(&[1.012444, 2.034123, 3.05678]);
        let correct: &str = "┌      ┐\n\
                             │ 1.01 │\n\
                             │ 2.03 │\n\
                             │ 3.06 │\n\
                             └      ┘";
        assert_eq!(format!("{:.2}", u), correct);
    }

    #[test]
    fn index_works() {
        let mut x = Vector::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn index_mut_works() {
        let mut x = Vector::new(3);
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
        let mut x = Vector::new(3);
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
        let mut x = Vector::new(3);
        x.data[0] = 1.0;
        x.data[1] = 2.0;
        x.data[2] = 3.0;
        let res = vector_to_string(&x);
        assert_eq!(res, "(0:1), (1:2), (2:3), ");
    }
}
