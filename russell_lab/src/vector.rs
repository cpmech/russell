pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub fn new(dim: usize) -> Self {
        Vector {
            data: vec![0.0; dim],
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
}
