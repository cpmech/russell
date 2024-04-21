use russell_lab::{mat_write_vismatrix, Matrix, StrError};

fn main() -> Result<(), StrError> {
    // the code below is the JavaScript one by George Reith from
    // https://codegolf.stackexchange.com/questions/16587/print-a-smiley-face
    let c = |y, w, h| (0.5 * w * f64::sqrt(1.0 - y * y / (h * h)) + 0.5) as i32 | 0;
    let r = 22.0;
    let dim = 2 * (r as usize) - 1;
    let mut matrix = Matrix::new(dim - 1, dim);
    let p = r / 2.0;
    let q = p / 5.0;
    let mut s = String::new();
    let mut y = 1.0 - p;
    let mut i = 0;
    while y < p {
        let mut j = 0;
        let mut x = 1.0 - r;
        while x < r {
            let d = c(y, r * 2.0, p) as f64;
            let e = c(y + q, r / 5.0, q) as f64;
            let f = e - p;
            let g = e + p;
            let h = c(y, r * 1.3, r / 3.0) as f64;
            s += if x >= d || x <= -d || (x > -g && x < f) || (x < g && x > -f) || (y > q && (x > -h && x < h)) {
                " "
            } else {
                matrix.set(2 * i + 0, j, 99.0 + r + x);
                matrix.set(2 * i + 1, j, 99.0 + r + x);
                "#"
            };
            j += 1;
            x += 1.0;
        }
        s += "\n";
        i += 1;
        y += 1.0;
    }
    println!("{}", s);
    if r < 10.0 {
        println!("{}", matrix);
    }
    mat_write_vismatrix("/tmp/russell_lab/matrix_visualization.smat", &matrix, 1e-15)?;
    Ok(())
}
