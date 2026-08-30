use super::Tensor2;
use crate::Rep;
use russell_lab::StrError;

/// Performs the polar decomposition A = Q · H using the quaternion-based algorithm
/// by Higham & Noferini (2016)
///
/// For a real 3×3 matrix A, this computes the polar decomposition `A = Q · H`,
/// where `Q` is orthogonal and `H` is symmetric positive semidefinite.
///
/// This is a direct (non-iterative) method based on the connection between
/// orthogonal 3×3 matrices and quaternions. In Brannon's notation, `Q` is the
/// rotation `R` and `H` is the right stretch `U`.
///
/// # Reference
///
/// N. J. Higham and V. Noferini, "An algorithm to compute the polar decomposition
/// of a 3×3 matrix", Num. Algorithms, 73(2):349–369, 2016.
///
/// # Output
///
/// * `q` -- the orthogonal polar factor Q; must be [Rep::General]
/// * `h` -- the symmetric positive semidefinite factor H; must be [Rep::Symmetric]
///
/// # Input
///
/// * `a` -- the matrix A; must be [Rep::General]
///
/// # Errors
///
/// Returns an error if the required [Rep] enums are incorrect.
pub(crate) fn polar_quaternion_higham(q: &mut Tensor2, h: &mut Tensor2, a: &Tensor2) -> Result<(), StrError> {
    if a.rep() != Rep::General {
        return Err("a must be Rep::General");
    }
    if q.rep() != Rep::General {
        return Err("q must be Rep::General");
    }
    if h.rep() != Rep::Symmetric {
        return Err("h must be Rep::Symmetric");
    }

    let mut aa = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            aa[i][j] = a.get_std(i, j);
        }
    }

    let (qq, hh) = polar_quaternion_raw(&aa);

    q.set_std_matrix(&qq).unwrap();
    h.set_std_matrix(&hh).unwrap();
    Ok(())
}

/// Port of `polar_quaternion.m` (Higham & Noferini, 2016).
///
/// Works on raw 3×3 arrays; returns `(Q, H)` with `A = Q · H`.
fn polar_quaternion_raw(aa_in: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    // Frobenius norm and scaling to unit norm
    let mut n = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            n += aa_in[i][j] * aa_in[i][j];
        }
    }
    n = n.sqrt();
    let mut a = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = aa_in[i][j] / n;
        }
    }

    let mut subspa = false;

    // b = 1 - 4 * sum of squares of the nine 2x2 minors of A
    let mut b = 0.0;
    let mut m1;
    m1 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    b += m1 * m1;
    m1 = a[1][0] * a[2][2] - a[1][2] * a[2][0];
    b += m1 * m1;
    m1 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    b += m1 * m1;
    m1 = a[0][0] * a[2][1] - a[0][1] * a[2][0];
    b += m1 * m1;
    m1 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    b += m1 * m1;
    m1 = a[0][1] * a[2][2] - a[0][2] * a[2][1];
    b += m1 * m1;
    m1 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    b += m1 * m1;
    m1 = a[0][0] * a[1][2] - a[0][2] * a[1][0];
    b += m1 * m1;
    m1 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    b += m1 * m1;
    b = b * (-4.0) + 1.0;

    let mut d: f64;
    let mut dd: f64;
    let mut nit: usize = 0;
    let mut quick = true;
    if (b - 1.0 + 1e-4) > 0.0 {
        quick = false;
        // LU (full pivoting)
        let mut r = 0usize;
        let mut c = 0usize;
        let mut aa2 = a;
        dd = 1.0;
        if a[1][0].abs() > a[0][0].abs() {
            r = 1;
        }
        if a[2][0].abs() > aa2[r][c].abs() {
            r = 2;
        }
        if a[0][1].abs() > aa2[r][c].abs() {
            r = 0;
            c = 1;
        }
        if a[1][1].abs() > aa2[r][c].abs() {
            r = 1;
            c = 1;
        }
        if a[2][1].abs() > aa2[r][c].abs() {
            r = 2;
            c = 1;
        }
        if a[0][2].abs() > aa2[r][c].abs() {
            r = 0;
            c = 2;
        }
        if a[1][2].abs() > aa2[r][c].abs() {
            r = 1;
            c = 2;
        }
        if a[2][2].abs() > aa2[r][c].abs() {
            r = 2;
            c = 2;
        }
        if r > 0 {
            aa2.swap(0, r);
            dd = -1.0;
        }
        if c > 0 {
            for k in 0..3 {
                aa2[k].swap(0, c);
            }
            dd = -dd;
        }
        let mut u = [0.0f64; 3];
        u[0] = aa2[0][0];
        let m1 = aa2[0][1] / aa2[0][0];
        let m2 = aa2[0][2] / aa2[0][0];
        let aa22 = [
            [aa2[1][1] - aa2[1][0] * m1, aa2[1][2] - aa2[1][0] * m2],
            [aa2[2][1] - aa2[2][0] * m1, aa2[2][2] - aa2[2][0] * m2],
        ];
        r = 0;
        c = 0;
        if aa22[1][0].abs() > aa22[0][0].abs() {
            r = 1;
        }
        if aa22[0][1].abs() > aa22[r][c].abs() {
            r = 0;
            c = 1;
        }
        if aa22[1][1].abs() > aa22[r][c].abs() {
            r = 1;
            c = 1;
        }
        if r == 1 {
            dd = -dd;
        }
        if c > 0 {
            dd = -dd;
        }
        u[1] = aa22[r][c];
        if u[1] == 0.0 {
            u[2] = 0.0;
        } else {
            u[2] = aa22[1 - r][1 - c] - aa22[r][1 - c] * aa22[1 - r][c] / u[1];
        }
        d = dd;
        dd = dd * u[0] * u[1] * u[2];
        if u[0] < 0.0 {
            d = -d;
        }
        if u[1] < 0.0 {
            d = -d;
        }
        if u[2] < 0.0 {
            d = -d;
        }
        let au = u[1].abs();
        if au > 6.607e-8 {
            let nitf = 16.8 + 2.0 * au.log10();
            nit = (15.0 / nitf).ceil() as usize;
        } else {
            subspa = true;
        }
    } else {
        // LU (partial pivoting)
        let aa2: [[f64; 3]; 3];
        if a[1][0].abs() > a[2][0].abs() {
            if a[0][0].abs() > a[1][0].abs() {
                aa2 = a;
                dd = 1.0;
            } else {
                aa2 = [a[1], a[0], a[2]];
                dd = -1.0;
            }
        } else {
            if a[0][0].abs() > a[2][0].abs() {
                aa2 = a;
                dd = 1.0;
            } else {
                aa2 = [a[2], a[1], a[0]];
                dd = -1.0;
            }
        }
        d = dd;
        let mut u = [0.0f64; 3];
        u[0] = aa2[0][0];
        if u[0] < 0.0 {
            d = -d;
        }
        let m1 = aa2[0][1] / aa2[0][0];
        let m2 = aa2[0][2] / aa2[0][0];
        let aa22 = [
            [aa2[1][1] - aa2[1][0] * m1, aa2[1][2] - aa2[1][0] * m2],
            [aa2[2][1] - aa2[2][0] * m1, aa2[2][2] - aa2[2][0] * m2],
        ];
        if aa22[0][0].abs() < aa22[1][0].abs() {
            u[1] = aa22[1][0];
            u[2] = aa22[0][1] - aa22[0][0] * aa22[1][1] / aa22[1][0];
            dd = -dd;
            d = -d;
            if u[1] < 0.0 {
                d = -d;
            }
            if u[2] < 0.0 {
                d = -d;
            }
        } else if aa22[0][0] == 0.0 {
            u[1] = 0.0;
            u[2] = 0.0;
        } else {
            u[1] = aa22[0][0];
            u[2] = aa22[1][1] - aa22[1][0] * aa22[0][1] / aa22[0][0];
            if u[1] < 0.0 {
                d = -d;
            }
            if u[2] < 0.0 {
                d = -d;
            }
        }
        dd = dd * u[0] * u[1] * u[2];
    }

    if d == 0.0 {
        d = 1.0;
    }
    dd = 8.0 * d * dd;
    let t = a[0][0] + a[1][1] + a[2][2];

    // Build the symmetric 4x4 quaternion matrix B
    let mut b4 = [[0.0f64; 4]; 4];
    b4[0][0] = t;
    b4[0][1] = a[1][2] - a[2][1];
    b4[0][2] = a[2][0] - a[0][2];
    b4[0][3] = a[0][1] - a[1][0];
    b4[1][1] = 2.0 * a[0][0] - t;
    b4[1][2] = a[0][1] + a[1][0];
    b4[1][3] = a[0][2] + a[2][0];
    b4[2][2] = 2.0 * a[1][1] - t;
    b4[2][3] = a[1][2] + a[2][1];
    b4[3][3] = 2.0 * a[2][2] - t;
    for i in 0..4 {
        for j in 0..4 {
            b4[i][j] *= d;
        }
    }
    b4[1][0] = b4[0][1];
    b4[2][0] = b4[0][2];
    b4[3][0] = b4[0][3];
    b4[2][1] = b4[1][2];
    b4[3][1] = b4[1][3];
    b4[3][2] = b4[2][3];

    // Largest eigenvalue x of B
    let x: f64;
    if b >= -0.3332 {
        let delta0 = 1.0 + 3.0 * b;
        let delta1 = -1.0 + (27.0 / 16.0) * dd * dd + 9.0 * b;
        let phi = ((delta1 / delta0) / delta0.sqrt()).clamp(-1.0, 1.0);
        let ss = (4.0 / 3.0) * (1.0 + (phi.acos() / 3.0).cos() * delta0.sqrt());
        let s = ss.sqrt() / 2.0;
        x = s + 0.5 * (0.0f64.max(-ss + 4.0 + dd / s)).sqrt();
    } else {
        let mut xx = 3.0f64.sqrt();
        let mut xold = 3.0;
        while (xold - xx) > 1e-12 {
            xold = xx;
            let px = xx * (xx * (xx * xx - 2.0) - dd) + b;
            let dpx = xx * (4.0 * xx * xx - 4.0) - dd;
            xx = xx - px / dpx;
        }
        x = xx;
    }

    // Eigenvector v (the quaternion) via LDL^T
    let mut v: [f64; 4];
    if quick {
        // LDL (quick path)
        let mut bb = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                bb[i][j] = -b4[i][j];
            }
        }
        for i in 0..4 {
            bb[i][i] += x;
        }
        let mut p = [0usize, 1, 2, 3];
        let mut l = [[0.0f64; 4]; 4];
        l[0][0] = 1.0;
        l[1][1] = 1.0;
        l[2][2] = 1.0;
        l[3][3] = 1.0;
        let mut d4 = [0.0f64; 4];

        // First step
        let mut r = 3usize;
        if bb[3][3] < bb[2][2] {
            r = 2;
        }
        if bb[r][r] < bb[1][1] {
            r = 1;
        }
        if bb[r][r] > bb[0][0] {
            p.swap(0, r);
            bb.swap(0, r);
            for k in 0..4 {
                bb[k].swap(0, r);
            }
        }
        d4[0] = bb[0][0];
        l[1][0] = bb[1][0] / d4[0];
        l[2][0] = bb[2][0] / d4[0];
        l[3][0] = bb[3][0] / d4[0];
        bb[1][1] -= l[1][0] * bb[0][1];
        bb[2][1] -= l[2][0] * bb[0][1];
        bb[1][2] = bb[2][1];
        bb[3][1] -= l[3][0] * bb[0][1];
        bb[1][3] = bb[3][1];
        bb[2][2] -= l[2][0] * bb[0][2];
        bb[3][2] -= l[3][0] * bb[0][2];
        bb[2][3] = bb[3][2];
        bb[3][3] -= l[3][0] * bb[0][3];

        // Second step
        r = 3;
        if bb[3][3] < bb[2][2] {
            r = 2;
        }
        if bb[r][r] > bb[1][1] {
            p.swap(1, r);
            bb.swap(1, r);
            for k in 0..4 {
                bb[k].swap(1, r);
            }
            l.swap(1, r);
            for k in 0..4 {
                l[k].swap(1, r);
            }
        }
        d4[1] = bb[1][1];
        l[2][1] = bb[2][1] / d4[1];
        l[3][1] = bb[3][1] / d4[1];
        bb[2][2] -= l[2][1] * bb[1][2];
        bb[3][2] -= l[3][1] * bb[1][2];
        bb[2][3] = bb[3][2];
        bb[3][3] -= l[3][1] * bb[1][3];

        // Third step
        if bb[2][2] < bb[3][3] {
            d4[2] = bb[3][3];
            bb.swap(2, 3);
            for k in 0..4 {
                bb[k].swap(2, 3);
            }
            l.swap(2, 3);
            for k in 0..4 {
                l[k].swap(2, 3);
            }
            p.swap(2, 3);
        } else {
            d4[2] = bb[2][2];
        }
        l[3][2] = bb[3][2] / d4[2];
        v = [
            l[1][0] * l[3][1] + l[2][0] * l[3][2] - l[1][0] * l[3][2] * l[2][1] - l[3][0],
            l[3][2] * l[2][1] - l[3][1],
            -l[3][2],
            1.0,
        ];
        let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
        for i in 0..4 {
            v[i] /= nv;
        }
        let v_old = v;
        for i in 0..4 {
            v[p[i]] = v_old[i];
        }
    } else {
        // LDL (full path)
        let mut bb = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                bb[i][j] = -b4[i][j];
            }
        }
        for i in 0..4 {
            bb[i][i] += x;
        }
        let mut p = [0usize, 1, 2, 3];
        let mut l = [[0.0f64; 4]; 4];
        l[0][0] = 1.0;
        l[1][1] = 1.0;
        l[2][2] = 1.0;
        l[3][3] = 1.0;
        let mut d4 = [[0.0f64; 4]; 4];

        // First step
        let mut r = 3usize;
        if bb[3][3] < bb[2][2] {
            r = 2;
        }
        if bb[r][r] < bb[1][1] {
            r = 1;
        }
        if bb[r][r] > bb[0][0] {
            p.swap(0, r);
            bb.swap(0, r);
            for k in 0..4 {
                bb[k].swap(0, r);
            }
        }
        d4[0][0] = bb[0][0];
        l[1][0] = bb[1][0] / d4[0][0];
        l[2][0] = bb[2][0] / d4[0][0];
        l[3][0] = bb[3][0] / d4[0][0];
        bb[1][1] -= l[1][0] * bb[0][1];
        bb[2][1] -= l[2][0] * bb[0][1];
        bb[1][2] = bb[2][1];
        bb[3][1] -= l[3][0] * bb[0][1];
        bb[1][3] = bb[3][1];
        bb[2][2] -= l[2][0] * bb[0][2];
        bb[3][2] -= l[3][0] * bb[0][2];
        bb[2][3] = bb[3][2];
        bb[3][3] -= l[3][0] * bb[0][3];

        // Second step
        r = 2;
        if bb[2][2] < bb[1][1] {
            r = 1;
        }
        if bb[r][r] > bb[0][0] {
            p.swap(1, r);
            bb.swap(1, r);
            for k in 0..4 {
                bb[k].swap(1, r);
            }
            l.swap(1, r);
            for k in 0..4 {
                l[k].swap(1, r);
            }
        }
        d4[1][1] = bb[1][1];
        l[2][1] = bb[2][1] / d4[1][1];
        l[3][1] = bb[3][1] / d4[1][1];
        d4[2][2] = bb[2][2] - l[2][1] * bb[1][2];
        d4[3][2] = bb[3][2] - l[3][1] * bb[1][2];
        d4[2][3] = d4[3][2];
        d4[3][3] = bb[3][3] - l[3][1] * bb[1][3];

        let dd2 = d4[2][2] * d4[3][3] - d4[2][3] * d4[2][3];
        if dd2 == 0.0 {
            // treat specially
            let mx = d4[2][2].abs().max(d4[2][3].abs()).max(d4[3][3].abs());
            if mx == 0.0 {
                v = [l[1][0] * l[3][1] - l[3][0], -l[3][1], 0.0, 1.0];
                let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                for i in 0..4 {
                    v[i] /= nv;
                }
            } else {
                let n2 = null2_sym(d4[2][2], d4[2][3], d4[3][3]);
                // v = L' \ [0; 0; n2]  (back-substitution with L^T upper triangular)
                let b = [0.0, 0.0, n2[0], n2[1]];
                v = back_sub_l_t(&l, &b);
                let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                for i in 0..4 {
                    v[i] /= nv;
                }
            }
        } else {
            let id = [[d4[3][3], -d4[2][3]], [-d4[2][3], d4[2][2]]];
            if subspa {
                // subspace path
                let mut vv = [
                    [l[1][0] * l[2][1] - l[2][0], l[1][0] * l[3][1] - l[3][0]],
                    [-l[2][1], -l[3][1]],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ];
                let mut il = [[0.0f64; 4]; 4];
                il[0][0] = 1.0;
                il[1][1] = 1.0;
                il[1][0] = -l[1][0];
                il[2][0] = vv[0][0];
                il[2][1] = vv[1][0];
                il[2][2] = vv[2][0];
                il[2][3] = vv[3][0];
                il[3][0] = vv[0][1];
                il[3][1] = vv[1][1];
                il[3][2] = vv[2][1];
                il[3][3] = vv[3][1];

                qr4x2(&mut vv);
                // vv = IL * vv
                let tmp = mat4_vec(&il, &[vv[0][0], vv[1][0], vv[2][0], vv[3][0]]);
                vv[0][0] = tmp[0];
                vv[1][0] = tmp[1];
                vv[2][0] = tmp[2];
                vv[3][0] = tmp[3];
                let tmp = mat4_vec(&il, &[vv[0][1], vv[1][1], vv[2][1], vv[3][1]]);
                vv[0][1] = tmp[0];
                vv[1][1] = tmp[1];
                vv[2][1] = tmp[2];
                vv[3][1] = tmp[3];
                // vv(1,:) /= D(1,1); vv(2,:) /= D(2,2); vv(3:4,:) = ID*vv(3:4,:)/DD
                for j in 0..2 {
                    vv[0][j] /= d4[0][0];
                    vv[1][j] /= d4[1][1];
                    let z0 = vv[2][j];
                    let z1 = vv[3][j];
                    vv[2][j] = (id[0][0] * z0 + id[0][1] * z1) / dd2;
                    vv[3][j] = (id[1][0] * z0 + id[1][1] * z1) / dd2;
                }
                // vv = IL^T * vv
                vv = mat4t_times_4x2(&il, &vv);
                // vv = IL * vv
                let tmp = mat4_vec(&il, &[vv[0][0], vv[1][0], vv[2][0], vv[3][0]]);
                vv[0][0] = tmp[0];
                vv[1][0] = tmp[1];
                vv[2][0] = tmp[2];
                vv[3][0] = tmp[3];
                let tmp = mat4_vec(&il, &[vv[0][1], vv[1][1], vv[2][1], vv[3][1]]);
                vv[0][1] = tmp[0];
                vv[1][1] = tmp[1];
                vv[2][1] = tmp[2];
                vv[3][1] = tmp[3];
                // vv(1,:) /= D(1,1); vv(2,:) /= D(2,2); vv(3:4,:) = ID*vv(3:4,:)/DD
                for j in 0..2 {
                    vv[0][j] /= d4[0][0];
                    vv[1][j] /= d4[1][1];
                    let z0 = vv[2][j];
                    let z1 = vv[3][j];
                    vv[2][j] = (id[0][0] * z0 + id[0][1] * z1) / dd2;
                    vv[3][j] = (id[1][0] * z0 + id[1][1] * z1) / dd2;
                }
                // vv = IL^T * vv
                vv = mat4t_times_4x2(&il, &vv);
                qr4x2(&mut vv);
                // H = vv' * L ; H = -H * D * H'
                // H = vv' * L  (2x4)
                let mut hh24 = [[0.0f64; 4]; 2];
                for ii in 0..2 {
                    for jj in 0..4 {
                        let mut s = 0.0;
                        for k in 0..4 {
                            s += vv[k][ii] * l[k][jj];
                        }
                        hh24[ii][jj] = s;
                    }
                }
                // H = -H * D * H'  =>  hh3 = - hh24 * d4 * hh24^T  (2x2)
                let mut tmp24 = [[0.0f64; 4]; 2];
                for ii in 0..2 {
                    for jj in 0..4 {
                        let mut s = 0.0;
                        for k in 0..4 {
                            s += hh24[ii][k] * d4[k][jj];
                        }
                        tmp24[ii][jj] = s;
                    }
                }
                let mut hh3 = [[0.0f64; 2]; 2];
                for ii in 0..2 {
                    for jj in 0..2 {
                        let mut s = 0.0;
                        for k in 0..4 {
                            s += tmp24[ii][k] * hh24[jj][k];
                        }
                        hh3[ii][jj] = -s;
                    }
                }
                let mut col: [f64; 4];
                if hh3[0][1].abs() < 1e-15 {
                    if hh3[0][0] > hh3[0][1] {
                        col = [vv[0][0], vv[1][0], vv[2][0], vv[3][0]];
                    } else {
                        col = [vv[0][1], vv[1][1], vv[2][1], vv[3][1]];
                    }
                } else {
                    let rr = (hh3[0][0] - hh3[1][1]) / (2.0 * hh3[0][1]);
                    let w = rr + hh3[0][1].signum() * (1.0 + rr * rr).sqrt();
                    col = [
                        vv[0][0] * w + vv[0][1],
                        vv[1][0] * w + vv[1][1],
                        vv[2][0] * w + vv[2][1],
                        vv[3][0] * w + vv[3][1],
                    ];
                    let ncol = (col[0] * col[0] + col[1] * col[1] + col[2] * col[2] + col[3] * col[3]).sqrt();
                    for i in 0..4 {
                        col[i] /= ncol;
                    }
                }
                v = col;
            } else {
                v = [
                    l[1][0] * l[3][1] + l[2][0] * l[3][2] - l[1][0] * l[3][2] * l[2][1] - l[3][0],
                    l[3][2] * l[2][1] - l[3][1],
                    -l[3][2],
                    1.0,
                ];
                let mut il = [[0.0f64; 4]; 4];
                il[0][0] = 1.0;
                il[1][1] = 1.0;
                il[1][0] = -l[1][0];
                il[2][0] = l[1][0] * l[2][1] - l[2][0];
                il[2][1] = -l[2][1];
                il[2][2] = 1.0;
                il[3][0] = v[0];
                il[3][1] = v[1];
                il[3][2] = v[2];
                il[3][3] = v[3];
                let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                for i in 0..4 {
                    v[i] /= nv;
                }
                for _ in 0..nit {
                    let w = mat4_vec(&il, &v);
                    v = w;
                    v[0] /= d4[0][0];
                    v[1] /= d4[1][1];
                    let z0 = v[2];
                    let z1 = v[3];
                    v[2] = (id[0][0] * z0 + id[0][1] * z1) / dd2;
                    v[3] = (id[1][0] * z0 + id[1][1] * z1) / dd2;
                    // v = IL^T * v
                    let mut w2 = [0.0f64; 4];
                    for j in 0..4 {
                        let mut s = 0.0;
                        for i in 0..4 {
                            s += il[i][j] * v[i];
                        }
                        w2[j] = s;
                    }
                    v = w2;
                    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                    for i in 0..4 {
                        v[i] /= nv;
                    }
                }
            }
        }
        let v_old = v;
        for i in 0..4 {
            v[p[i]] = v_old[i];
        }
    }

    // Polar factor Q (up to sign) from the quaternion v
    let v22 = 2.0 * v[1] * v[1];
    let v33 = 2.0 * v[2] * v[2];
    let v44 = 2.0 * v[3] * v[3];
    let v23 = 2.0 * v[1] * v[2];
    let v14 = 2.0 * v[0] * v[3];
    let v24 = 2.0 * v[1] * v[3];
    let v13 = 2.0 * v[0] * v[2];
    let v12 = 2.0 * v[0] * v[1];
    let v34 = 2.0 * v[2] * v[3];
    let mut qq = [[0.0f64; 3]; 3];
    qq[0][0] = 1.0 - v33 - v44;
    qq[0][1] = v23 + v14;
    qq[0][2] = v24 - v13;
    qq[1][0] = v23 - v14;
    qq[1][1] = 1.0 - v22 - v44;
    qq[1][2] = v12 + v34;
    qq[2][0] = v13 + v24;
    qq[2][1] = v34 - v12;
    qq[2][2] = 1.0 - v22 - v33;

    if d == -1.0 {
        for i in 0..3 {
            for j in 0..3 {
                qq[i][j] = -qq[i][j];
            }
        }
    }

    // H = Q^T * A, then un-scale
    let mut hh = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += qq[k][i] * a[k][j];
            }
            hh[i][j] = s;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            hh[i][j] = n * hh[i][j];
        }
    }
    // Symmetrize H (the MATLAB code has this as an optional step)
    for i in 0..3 {
        for j in 0..i {
            let s = 0.5 * (hh[i][j] + hh[j][i]);
            hh[i][j] = s;
            hh[j][i] = s;
        }
    }

    (qq, hh)
}

/// Null vector of a symmetric 2×2 matrix [[a, b], [b, d]] with determinant zero.
fn null2_sym(a: f64, b: f64, _d: f64) -> [f64; 2] {
    if a != 0.0 || b != 0.0 { [b, -a] } else { [1.0, 0.0] }
}

/// Solves L^T v = b for a 4×4 unit lower-triangular L (back-substitution).
fn back_sub_l_t(l: &[[f64; 4]; 4], b: &[f64; 4]) -> [f64; 4] {
    let mut v = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut s = b[i];
        for j in (i + 1)..4 {
            s -= l[j][i] * v[j];
        }
        v[i] = s;
    }
    v
}

/// Thin QR of a 4×2 matrix (modified Gram-Schmidt), in place.
fn qr4x2(m: &mut [[f64; 2]; 4]) {
    let mut n = 0.0;
    for i in 0..4 {
        n += m[i][0] * m[i][0];
    }
    n = n.sqrt();
    for i in 0..4 {
        m[i][0] /= n;
    }
    let mut dot = 0.0;
    for i in 0..4 {
        dot += m[i][0] * m[i][1];
    }
    for i in 0..4 {
        m[i][1] -= dot * m[i][0];
    }
    n = 0.0;
    for i in 0..4 {
        n += m[i][1] * m[i][1];
    }
    n = n.sqrt();
    for i in 0..4 {
        m[i][1] /= n;
    }
}

/// Matrix-vector product (4×4 times 4×1).
fn mat4_vec(m: &[[f64; 4]; 4], v: &[f64; 4]) -> [f64; 4] {
    let mut w = [0.0f64; 4];
    for i in 0..4 {
        let mut s = 0.0;
        for j in 0..4 {
            s += m[i][j] * v[j];
        }
        w[i] = s;
    }
    w
}

/// Computes IL^T * M for a 4×4 IL and a 4×2 M (returns 4×2).
fn mat4t_times_4x2(il: &[[f64; 4]; 4], m: &[[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let mut r = [[0.0f64; 2]; 4];
    for j in 0..2 {
        for i in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += il[k][i] * m[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::polar_quaternion_higham;
    use crate::test_common::{
        case51, case52, case52_rotation, check_agree, check_polar, example01, example01_rotation, example01_stretch,
        example03, example03_rotation, example03_stretch,
    };
    use crate::{Rep, Tensor2};
    use russell_lab::mat_approx_eq;

    #[test]
    fn polar_quaternion_higham_works_case51() {
        // Higham & Noferini test (5.1)
        let a = case51();
        let mut q = Tensor2::new(Rep::General);
        let mut h = Tensor2::new(Rep::Symmetric);
        polar_quaternion_higham(&mut q, &mut h, &a).unwrap();
        check_polar(&a, &q, &h, 1e-13);
    }

    #[test]
    fn polar_quaternion_higham_works_case52() {
        // Higham & Noferini test (5.2) over a range of condition numbers
        // (y = sqrt([1, 1e-4, 1e-8, 1e-12, 1e-16]))
        for y in [1.0f64, 1e-2, 1e-4, 1e-6, 1e-8] {
            let a = case52(y);
            let mut q = Tensor2::new(Rep::General);
            let mut h = Tensor2::new(Rep::Symmetric);
            polar_quaternion_higham(&mut q, &mut h, &a).unwrap();
            check_polar(&a, &q, &h, 1e-13);
        }
        // Compare Q with the exact Q1 from the paper (well-conditioned case y = 1)
        let a = case52(1.0);
        let mut q = Tensor2::new(Rep::General);
        let mut h = Tensor2::new(Rep::Symmetric);
        polar_quaternion_higham(&mut q, &mut h, &a).unwrap();
        mat_approx_eq(&q.as_std_matrix(), &case52_rotation(), 1e-13);
    }

    #[test]
    fn polar_quaternion_higham_on_brannon_cases() {
        // Brannon's example 01 (in-plane), cross-checked against her algorithm
        let a = example01();
        let mut q = Tensor2::new(Rep::General);
        let mut h = Tensor2::new(Rep::Symmetric);
        polar_quaternion_higham(&mut q, &mut h, &a).unwrap();
        check_agree(&a);
        mat_approx_eq(&q.as_std_matrix(), &example01_rotation(), 1e-13);
        mat_approx_eq(&h.as_std_matrix(), &example01_stretch(), 1e-13);

        // Brannon's example 03 (fully 3-D), cross-checked against her algorithm
        let a = example03();
        let mut q = Tensor2::new(Rep::General);
        let mut h = Tensor2::new(Rep::Symmetric);
        polar_quaternion_higham(&mut q, &mut h, &a).unwrap();
        check_agree(&a);
        mat_approx_eq(&q.as_std_matrix(), &example03_rotation(), 1e-3);
        mat_approx_eq(&h.as_std_matrix(), &example03_stretch(), 1e-3);
    }
}
