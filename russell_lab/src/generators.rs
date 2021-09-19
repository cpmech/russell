use crate::Matrix;

/// Generates 3d points
///
/// # Input
///
/// * `xmin`, `xmax` -- range along x
/// * `ymin`, `ymax` -- range along y
/// * `nx` -- is the number of points along x (must be `>= 2`)
/// * `ny` -- is the number of points along y (must be `>= 2`)
/// * `calc_z` -- is a function of (x[i][j], y[i][j]) that calculates z[i][j]
///
/// # Output
///
/// * `x`, `y`, `z` -- (ny, nx) matrices
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
            x[i][j] = u;
            y[i][j] = v;
            z[i][j] = calc_z(u, v);
        }
    }
    (x, y, z)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate3d_works() {
        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 0, 2, |x, y| x + y);
        assert_eq!(x.dims(), (2, 0));
        assert_eq!(y.dims(), (2, 0));
        assert_eq!(z.dims(), (2, 0));
        assert_eq!(x.as_data(), &[]);
        assert_eq!(y.as_data(), &[]);
        assert_eq!(z.as_data(), &[]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 0, |x, y| x + y);
        assert_eq!(x.dims(), (0, 2));
        assert_eq!(y.dims(), (0, 2));
        assert_eq!(z.dims(), (0, 2));
        assert_eq!(x.as_data(), &[]);
        assert_eq!(y.as_data(), &[]);
        assert_eq!(z.as_data(), &[]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 1, 2, |x, y| x + y);
        assert_eq!(x.dims(), (2, 1));
        assert_eq!(y.dims(), (2, 1));
        assert_eq!(z.dims(), (2, 1));
        assert_eq!(x.as_data(), &[-1.0, -1.0]);
        assert_eq!(y.as_data(), &[-3.0, 3.0]);
        assert_eq!(z.as_data(), &[-4.0, 2.0]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 1, |x, y| x + y);
        assert_eq!(x.dims(), (1, 2));
        assert_eq!(y.dims(), (1, 2));
        assert_eq!(z.dims(), (1, 2));
        assert_eq!(x.as_data(), &[-1.0, 1.0]);
        assert_eq!(y.as_data(), &[-3.0, -3.0]);
        assert_eq!(z.as_data(), &[-4.0, -2.0]);

        let (x, y, z) = generate3d(-1.0, 1.0, -3.0, 3.0, 2, 3, |x, y| x + y);
        assert_eq!(x.dims(), (3, 2));
        assert_eq!(y.dims(), (3, 2));
        assert_eq!(z.dims(), (3, 2));
        assert_eq!(x.as_data(), &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        assert_eq!(y.as_data(), &[-3.0, -3.0, 0.0, 0.0, 3.0, 3.0]);
        assert_eq!(z.as_data(), &[-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]);
    }
}
