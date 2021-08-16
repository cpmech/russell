use std::convert::TryInto;

use super::*;
use russell_openblas::*;

pub fn inner(u: &Vector, v: &Vector) -> f64 {
    let n = if u.data.len() < v.data.len() {
        u.data.len()
    } else {
        v.data.len()
    };
    ddot(n.try_into().unwrap(), &u.data, 1, &v.data, 1)
}

// pub fn outer(A: &mut Matrix, u: &Vecgtor, v: &Vector) {}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let y = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(inner(&x, &y), -1070.0);
    }
}
