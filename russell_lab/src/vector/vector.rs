use russell_openblas::*;
use std::cmp;
use std::convert::TryInto;
use std::fmt::{self, Write};

pub struct Vector {
    pub(crate) data: Vec<f64>,
}

impl Vector {
    /// Creates a new (zeroed) vector
    ///
    /// # Examples
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
        Vector {
            data: vec![0.0; dim],
        }
    }

    /// Creates a vector from data
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::from(&[1.0, 2.0, 3.0]);
    /// let correct = "┌   ┐\n\
    ///                │ 1 │\n\
    ///                │ 2 │\n\
    ///                │ 3 │\n\
    ///                └   ┘";
    /// assert_eq!(format!("{}", u), correct);
    /// ```
    pub fn from(data: &[f64]) -> Self {
        Vector {
            data: Vec::from(data),
        }
    }

    /// Returns the dimension (size) of this vector
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    /// let u = Vector::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.dim(), 3);
    /// ```
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Scales this vector
    ///
    /// ```text
    /// u := alpha * u
    /// ```
    ///
    /// # Examples
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
    ///
    pub fn scale(&mut self, alpha: f64) {
        let n: i32 = self.data.len().try_into().unwrap();
        dscal(n, alpha, &mut self.data, 1);
    }

    /// Applies a function over all components of this vector
    ///
    /// ```text
    /// u := apply(function(ui))
    /// ```
    ///
    /// # Examples
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
    /// # Examples
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
}

impl fmt::Display for Vector {
    /// Implements the Display trait
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..self.data.len() {
            let val = self.data[i];
            write!(&mut buf, "{}", val)?;
            width = cmp::max(buf.chars().count(), width);
            buf.clear();
        }
        width += 1;
        write!(f, "┌{:1$}┐\n", " ", width + 1)?;
        for i in 0..self.data.len() {
            if i > 0 {
                write!(f, " │\n")?;
            }
            write!(f, "│")?;
            let val = self.data[i];
            write!(f, "{:>1$}", val, width)?;
        }
        write!(f, " │\n")?;
        write!(f, "└{:1$}┘", " ", width + 1)?;
        Ok(())
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
        let correct = &[0.0, 0.0, 0.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn from_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let correct = &[1.0, 2.0, 3.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn display_trait_works() {
        #[rustfmt::skip]
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let correct = "┌   ┐\n\
                            │ 1 │\n\
                            │ 2 │\n\
                            │ 3 │\n\
                            └   ┘";
        assert_eq!(format!("{}", u), correct);
    }

    #[test]
    fn scale_works() {
        let mut u = Vector::from(&[6.0, 9.0, 12.0]);
        u.scale(1.0 / 3.0);
        let correct = &[2.0, 3.0, 4.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }

    #[test]
    fn apply_works() {
        let mut u = Vector::from(&[-1.0, -2.0, -3.0]);
        u.apply_with_index(|i, x| x * x * x + (i as f64));
        let correct = &[-1.0, -7.0, -25.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }
}
