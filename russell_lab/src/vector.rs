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
}
