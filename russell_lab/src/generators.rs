use crate::Matrix;

/// Generates 2d points (meshgrid)
///
/// # Input
///
/// * `xmin`, `xmax` -- range along x
/// * `ymin`, `ymax` -- range along y
/// * `nx` -- is the number of points along x (must be `>= 2`)
/// * `ny` -- is the number of points along y (must be `>= 2`)
///
/// # Output
///
/// * `x`, `y` -- (`ny` by `nx`) matrices
///
/// # Example
///
/// ```
/// use russell_lab::generate2d;
/// let (nx, ny) = (5, 3);
/// let (x, y) = generate2d(-1.0, 1.0, -2.0, 2.0, nx, ny);
/// assert_eq!(
///     format!("{}", x),
///     "┌                          ┐\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      └                          ┘"
/// );
/// assert_eq!(
///     format!("{}", y),
///     "┌                ┐\n\
///      │ -2 -2 -2 -2 -2 │\n\
///      │  0  0  0  0  0 │\n\
///      │  2  2  2  2  2 │\n\
///      └                ┘"
/// );
/// ```
pub fn generate2d(xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> (Matrix, Matrix) {
    let mut x = Matrix::new(ny, nx);
    let mut y = Matrix::new(ny, nx);
    if nx == 0 || ny == 0 {
        return (x, y);
    }
    let dx = if nx == 1 {
        xmin
    } else {
        (xmax - xmin) / ((nx - 1) as f64)
    };
    let dy = if ny == 1 {
        ymin
    } else {
        (ymax - ymin) / ((ny - 1) as f64)
    };
    for i in 0..ny {
        let v = ymin + (i as f64) * dy;
        for j in 0..nx {
            let u = xmin + (j as f64) * dx;
            x.set(i, j, u);
            y.set(i, j, v);
        }
    }
    (x, y)
}

/// Generates 3d points (function over meshgrid)
///
/// # Input
///
/// * `xmin`, `xmax` -- range along x
/// * `ymin`, `ymax` -- range along y
/// * `nx` -- is the number of points along x (must be `>= 2`)
/// * `ny` -- is the number of points along y (must be `>= 2`)
/// * `calc_z` -- is a function of (xij, yij) that calculates zij
///
/// # Output
///
/// * `x`, `y`, `z` -- (`ny` by `nx`) matrices
///
/// # Example
///
/// ```
/// use russell_lab::generate3d;
/// let (nx, ny) = (5, 3);
/// let (x, y, z) = generate3d(-1.0, 1.0, -2.0, 2.0, nx, ny, |x, y| x * x + y * y);
/// assert_eq!(
///     format!("{}", x),
///     "┌                          ┐\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      │   -1 -0.5    0  0.5    1 │\n\
///      └                          ┘"
/// );
/// assert_eq!(
///     format!("{}", y),
///     "┌                ┐\n\
///      │ -2 -2 -2 -2 -2 │\n\
///      │  0  0  0  0  0 │\n\
///      │  2  2  2  2  2 │\n\
///      └                ┘"
/// );
/// assert_eq!(
///     format!("{}", z),
///     "┌                          ┐\n\
///      │    5 4.25    4 4.25    5 │\n\
///      │    1 0.25    0 0.25    1 │\n\
///      │    5 4.25    4 4.25    5 │\n\
///      └                          ┘"
/// );
/// ```
pub fn generate3d<F>(
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
    calc_z: F,
) -> (Matrix, Matrix, Matrix)
where
    F: Fn(f64, f64) -> f64,
{
    let mut x = Matrix::new(ny, nx);
    let mut y = Matrix::new(ny, nx);
    let mut z = Matrix::new(ny, nx);
    if nx == 0 || ny == 0 {
        return (x, y, z);
    }
    let dx = if nx == 1 {
        xmin
    } else {
        (xmax - xmin) / ((nx - 1) as f64)
    };
    let dy = if ny == 1 {
        ymin
    } else {
        (ymax - ymin) / ((ny - 1) as f64)
    };
    for i in 0..ny {
        let v = ymin + (i as f64) * dy;
        for j in 0..nx {
            let u = xmin + (j as f64) * dx;
            x.set(i, j, u);
            y.set(i, j, v);
            z.set(i, j, calc_z(u, v));
        }
    }
    (x, y, z)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{generate2d, generate3d};
    use russell_openblas::col_major;

    #[test]
    fn generate2d_edge_cases_work() {
        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 0, 0);
        assert_eq!(x.dims(), (0, 0));
        assert_eq!(y.dims(), (0, 0));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);

        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 1, 1);
        assert_eq!(x.dims(), (1, 1));
        assert_eq!(y.dims(), (1, 1));
        assert_eq!(x.as_data(), &[-1.0]);
        assert_eq!(y.as_data(), &[-3.0]);
    }

    #[test]
    fn generate2d_works() {
        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 0, 2);
        assert_eq!(x.dims(), (2, 0));
        assert_eq!(y.dims(), (2, 0));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);

        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 2, 0);
        assert_eq!(x.dims(), (0, 2));
        assert_eq!(y.dims(), (0, 2));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);

        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 1, 2);
        assert_eq!(x.dims(), (2, 1));
        assert_eq!(y.dims(), (2, 1));
        assert_eq!(x.as_data(), &[-1.0, -1.0]);
        assert_eq!(y.as_data(), &[-3.0, 3.0]);

        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 2, 1);
        assert_eq!(x.dims(), (1, 2));
        assert_eq!(y.dims(), (1, 2));
        assert_eq!(x.as_data(), &[-1.0, 1.0]);
        assert_eq!(y.as_data(), &[-3.0, -3.0]);

        let (x, y) = generate2d(-1.0, 1.0, -3.0, 3.0, 2, 3);
        assert_eq!(x.dims(), (3, 2));
        assert_eq!(y.dims(), (3, 2));
        assert_eq!(x.as_data(), &col_major(3, 2, &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]));
        assert_eq!(y.as_data(), &col_major(3, 2, &[-3.0, -3.0, 0.0, 0.0, 3.0, 3.0]));
    }

    fn calc_z(x: f64, y: f64) -> f64 {
        x + y
    }

    #[test]
    fn generate3d_edge_cases_work() {
        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 0, 0, calc_z);
        assert_eq!(x.dims(), (0, 0));
        assert_eq!(y.dims(), (0, 0));
        assert_eq!(z.dims(), (0, 0));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);
        assert_eq!(z.as_data(), &[] as &[f64]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 1, 1, calc_z);
        assert_eq!(x.dims(), (1, 1));
        assert_eq!(y.dims(), (1, 1));
        assert_eq!(z.dims(), (1, 1));
        assert_eq!(x.as_data(), &[-1.0]);
        assert_eq!(y.as_data(), &[-3.0]);
        assert_eq!(z.as_data(), &[-4.0]);
    }

    #[test]
    fn generate3d_works() {
        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 0, 2, calc_z);
        assert_eq!(x.dims(), (2, 0));
        assert_eq!(y.dims(), (2, 0));
        assert_eq!(z.dims(), (2, 0));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);
        assert_eq!(z.as_data(), &[] as &[f64]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 0, calc_z);
        assert_eq!(x.dims(), (0, 2));
        assert_eq!(y.dims(), (0, 2));
        assert_eq!(z.dims(), (0, 2));
        assert_eq!(x.as_data(), &[] as &[f64]);
        assert_eq!(y.as_data(), &[] as &[f64]);
        assert_eq!(z.as_data(), &[] as &[f64]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 1, 2, calc_z);
        assert_eq!(x.dims(), (2, 1));
        assert_eq!(y.dims(), (2, 1));
        assert_eq!(z.dims(), (2, 1));
        assert_eq!(x.as_data(), &[-1.0, -1.0]);
        assert_eq!(y.as_data(), &[-3.0, 3.0]);
        assert_eq!(z.as_data(), &[-4.0, 2.0]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 1, calc_z);
        assert_eq!(x.dims(), (1, 2));
        assert_eq!(y.dims(), (1, 2));
        assert_eq!(z.dims(), (1, 2));
        assert_eq!(x.as_data(), &[-1.0, 1.0]);
        assert_eq!(y.as_data(), &[-3.0, -3.0]);
        assert_eq!(z.as_data(), &[-4.0, -2.0]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 3, calc_z);
        assert_eq!(x.dims(), (3, 2));
        assert_eq!(y.dims(), (3, 2));
        assert_eq!(z.dims(), (3, 2));
        assert_eq!(x.as_data(), &col_major(3, 2, &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]));
        assert_eq!(y.as_data(), &col_major(3, 2, &[-3.0, -3.0, 0.0, 0.0, 3.0, 3.0]));
        assert_eq!(z.as_data(), &col_major(3, 2, &[-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]));
    }
}
