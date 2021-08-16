use super::*;
use russell_openblas::*;
use std::convert::TryInto;

pub fn inner(u: &Vector, v: &Vector) -> f64 {
    let n = if u.data.len() < v.data.len() {
        u.data.len()
    } else {
        v.data.len()
    };
    ddot(n.try_into().unwrap(), &u.data, 1, &v.data, 1)
}

pub fn outer(a: &mut Matrix, u: &Vector, v: &Vector) {
    let m = u.data.len();
    let n = v.data.len();
    if a.nrow != m {
        panic!("matrix must have the same number of rows than the length of vector u");
    }
    if a.ncol != n {
        panic!("matrix must have the same number of columns than the length of vector v");
    }
    let lda = m;
    dger(
        m.try_into().unwrap(),
        n.try_into().unwrap(),
        1.0,
        &u.data,
        1,
        &v.data,
        1,
        &mut a.data,
        lda.try_into().unwrap(),
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn inner_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let y = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(inner(&x, &y), -1070.0);
    }

    #[test]
    fn outer_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, &u, &v);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[ 5.0, -2.0, 0.0, 1.0],
            &[10.0, -4.0, 0.0, 2.0],
            &[15.0, -6.0, 0.0, 3.0],
        ]);
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    fn outer_alt_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, &u, &v);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[1.0, 1.0, -2.0],
            &[2.0, 2.0, -4.0],
            &[3.0, 3.0, -6.0],
            &[4.0, 4.0, -8.0],
        ]);
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }
}
