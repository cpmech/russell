use std::cmp;
use std::fmt::{self, Write};

pub struct Vector {
    pub(super) data: Vec<f64>,
}

impl Vector {
    pub fn new(dim: usize) -> Self {
        Vector {
            data: vec![0.0; dim],
        }
    }

    pub fn from(data: &[f64]) -> Self {
        Vector {
            data: Vec::from(data),
        }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
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

mod add_vectors;
mod add_vectors_simd;
pub use crate::vector::add_vectors::*;
use crate::vector::add_vectors_simd::*;

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
}
