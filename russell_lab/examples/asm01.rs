use russell_lab::NumMatrix;

#[inline(never)]
fn set_approach(a: &mut NumMatrix<f64>) {
    a.set(0, 0, 1.0);
    a.set(0, 1, 2.0);
    a.set(1, 0, 3.0);
    a.set(1, 1, 4.0);
}

#[inline(never)]
fn index_approach(a: &mut NumMatrix<f64>) {
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 2.0;
    a[(1, 0)] = 3.0;
    a[(1, 1)] = 4.0;
}

fn main() {
    let mut a = NumMatrix::new(2, 2);
    set_approach(&mut a);
    index_approach(&mut a);
}
