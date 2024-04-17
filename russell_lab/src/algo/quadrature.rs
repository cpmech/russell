use super::Stats;
use crate::math::{LN2, SQRT_2};
use crate::StrError;

// The quadrature function below is based on the Fortran function named
// dgauss_generic by Jacob Williams (the function is, in turn, based on SLATEC)
//
// quadrature-fortran: Adaptive Gaussian Quadrature with Modern Fortran
// <https://github.com/jacobwilliams/quadrature-fortran>
//
// Copyright (c) 2019-2021, Jacob Williams
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice, this
//   list of conditions and the following disclaimer in the documentation and/or
//   other materials provided with the distribution.
//
// * The names of its contributors may not be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// quadrature-fortran includes code from SLATEC, a public domain library.

/// Unknown definition
const KMX: usize = 5000; // !! ??

/// Unknown definition
const KML: usize = 6; // !! ??

/// Unknown definition
const MAGIC: f64 = 0.30102000_f64; // !! ??

/// Size of the workspace arrays
const N_WORK: usize = 60; // !! size of the work arrays. ?? Why 60 ??

/// log10(2)
const D1MACH5: f64 = 0.30102999566398119521373889472449; // !! machine constant

/// Multiplier for the error estimate comparison
///
/// **Note:** The original Fortran code has this constant set to 2.0. However, for the
/// `log(2 Cos(x/2))` function, the comparison fails with tolerance = 1e-10 but succeeds
/// with tolerance = 2.2e-11. The same behavior arises in the original Fortran code.
const M_ERR: f64 = 3.0;

/// Holds parameters for the quadrature functions
#[derive(Clone, Copy, Debug)]
pub struct QuadParams {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 2
    /// ```
    pub n_iteration_max: usize,

    /// Tolerance
    ///
    /// e.g., 1e-10
    pub tolerance: f64,

    /// Number of Gauss points in {6, 8, 10, 12, 14}
    pub n_gauss: usize,
}

impl QuadParams {
    /// Allocates a new instance
    pub fn new() -> Self {
        QuadParams {
            n_iteration_max: 300,
            tolerance: 1e-10,
            n_gauss: 10,
        }
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 2 {
            return Err("n_iteration_max must be ≥ 2");
        }
        if self.tolerance < 10.0 * f64::EPSILON {
            return Err("the tolerance must be ≥ 10.0 * f64::EPSILON");
        }
        let ok = match self.n_gauss {
            6 | 8 | 10 | 12 | 14 => true,
            _ => false,
        };
        if !ok {
            return Err("n_gauss must be 6, 8, 10, 12, or 14");
        }
        Ok(())
    }
}

/// Implements numerical integration using adaptive Gaussian quadrature
pub struct Quadrature {
    aa: Vec<f64>, // with len() +1 so we can use Fortran's one-based indexing
    hh: Vec<f64>,
    vl: Vec<f64>,
    gr: Vec<f64>,
    lr: Vec<i32>,
}

impl Quadrature {
    /// Allocates a new instance
    pub fn new() -> Self {
        Quadrature {
            aa: vec![0.0; N_WORK + 1], // +1 so we can use Fortran's one-based indexing
            hh: vec![0.0; N_WORK + 1],
            vl: vec![0.0; N_WORK + 1],
            gr: vec![0.0; N_WORK + 1],
            lr: vec![0; N_WORK + 1],
        }
    }

    /// Integrates a function f(x) using numerical quadrature
    ///
    /// Approximates:
    ///
    /// ```text
    ///        b
    ///       ⌠
    /// I  =  │  f(x) dx
    ///       ⌡
    ///      a
    /// ```
    ///
    /// # Input
    ///
    /// * `a` -- the lower bound
    /// * `b` -- the upper bound
    /// * `params` -- optional control parameters
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Output
    ///
    /// Returns `(ii, stats)` where:
    ///
    /// * `ans` -- the result `I` of the integration: `I = ∫_a^b f(x) dx`
    /// * `stats` -- some statistics about the computations, including the estimated error
    pub fn integrate<F, A>(
        &mut self,
        a: f64,
        b: f64,
        params: Option<QuadParams>,
        args: &mut A,
        mut f: F,
    ) -> Result<(f64, Stats), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // check
        if f64::abs(b - a) < 10.0 * f64::EPSILON {
            return Err("the lower and upper bounds must be different from each other");
        }

        // parameters
        let par = match params {
            Some(p) => p,
            None => QuadParams::new(),
        };
        par.validate()?;

        // allocate stats struct
        let mut stats = Stats::new();

        // initialization
        let mut ans = 0.0;
        let mut err = 0.0;
        let k = f64::MANTISSA_DIGITS as f64;
        let mut n_ib = D1MACH5 * k / MAGIC;
        let n_bit = n_ib as usize;
        let n_lmx = usize::min(60, (n_bit * 5) / 8);
        let mut lmx = n_lmx;

        // check
        if b != 0.0 {
            if f64::copysign(1.0, b) * a > 0.0 {
                let c = f64::abs(1.0 - a / b);
                if c <= 0.1 {
                    //
                    // Important: the following (removed) branching is not possible:
                    //
                    // if c <= 0.0 {
                    //     return Ok((ans, stats));
                    // }
                    //
                    // because c is never negative and a/b cannot be 1.0 (yielding c == 0.0)
                    // Note that a and b have already been checked above, validating:
                    //
                    // f64::abs(b - a) >= 10.0 * f64::EPSILON
                    //
                    n_ib = 0.5 - f64::ln(c) / LN2;
                    let nib = n_ib as usize;
                    lmx = usize::min(n_lmx, n_bit - nib - 7);
                    if lmx < 1 {
                        return Err("the lower and upper bounds must not be so close one from another");
                    }
                }
            }
        }

        let tol = f64::max(f64::abs(par.tolerance), f64::powf(2.0, 5.0 - n_bit as f64)) / 2.0;
        let mut eps = tol;
        self.hh[1] = (b - a) / 4.0;
        self.aa[1] = a;
        self.lr[1] = 1;
        let mut l = 1;

        let mut est = gauss(
            par.n_gauss,
            self.aa[l] + 2.0 * self.hh[l],
            2.0 * self.hh[l],
            args,
            &mut f,
        )?;
        stats.n_function += par.n_gauss;

        let mut k = 8;
        let mut area = f64::abs(est);
        let mut ef = 0.5;
        let mut mxl = 0;

        // compute refined estimates, estimate the error, etc.
        let mut converged = false;
        for _ in 0..par.n_iteration_max {
            stats.n_iterations += 1;

            let gl = gauss(par.n_gauss, self.aa[l] + self.hh[l], self.hh[l], args, &mut f)?;
            stats.n_function += par.n_gauss;

            self.gr[l] = gauss(par.n_gauss, self.aa[l] + 3.0 * self.hh[l], self.hh[l], args, &mut f)?;
            stats.n_function += par.n_gauss;

            k += 16;
            area += f64::abs(gl) + f64::abs(self.gr[l]) - f64::abs(est);
            let glr = gl + self.gr[l];
            let ee = f64::abs(est - glr) * ef;
            let ae = f64::max(eps * area, tol * f64::abs(glr));

            if ee - ae > 0.0 {
                // consider the left half of this level
                if k > KMX {
                    lmx = KML;
                }
                if l >= lmx {
                    mxl = 1;
                } else {
                    l += 1;
                    eps *= 0.5;
                    ef /= SQRT_2;
                    self.hh[l] = self.hh[l - 1] * 0.5;
                    self.lr[l] = -1;
                    self.aa[l] = self.aa[l - 1];
                    est = gl;
                    continue;
                }
            }

            err += est - glr;
            if self.lr[l] > 0 {
                // return one level
                ans = glr;
                loop {
                    if l <= 1 {
                        converged = true;
                        break;
                    }
                    l -= 1;
                    eps *= 2.0;
                    ef *= SQRT_2;
                    if self.lr[l] <= 0 {
                        self.vl[l] = self.vl[l + 1] + ans;
                        est = self.gr[l - 1];
                        self.lr[l] = 1;
                        self.aa[l] = self.aa[l] + 4.0 * self.hh[l];
                        break;
                    }
                    ans += self.vl[l + 1];
                }
                if converged {
                    break;
                }
            } else {
                // proceed to right half at this level
                self.vl[l] = glr;
                est = self.gr[l - 1];
                self.lr[l] = 1;
                self.aa[l] += 4.0 * self.hh[l];
                // no need for cycle/continue here ...
            }
        }

        // check
        if !converged {
            return Err("integrate failed to converge");
        }

        // done
        stats.error_estimate = err;
        stats.stop_sw_total();
        if (mxl != 0) && (f64::abs(err) > M_ERR * tol * area) {
            Err("cannot achieve the desired tolerance")
        } else {
            Ok((ans, stats))
        }
    }
}

/// Performs a Gaussian quadrature
fn gauss<F, A>(npoint: usize, x: f64, h: f64, args: &mut A, f: &mut F) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let mut sum = 0.0;
    match npoint {
        6 => {
            for i in 0..3 {
                sum += G6_W[i] * (f(x - G6_A[i] * h, args)? + f(x + G6_A[i] * h, args)?);
            }
        }
        8 => {
            for i in 0..4 {
                sum += G8_W[i] * (f(x - G8_A[i] * h, args)? + f(x + G8_A[i] * h, args)?);
            }
        }
        10 => {
            for i in 0..5 {
                sum += G10_W[i] * (f(x - G10_A[i] * h, args)? + f(x + G10_A[i] * h, args)?);
            }
        }
        12 => {
            for i in 0..6 {
                sum += G12_W[i] * (f(x - G12_A[i] * h, args)? + f(x + G12_A[i] * h, args)?);
            }
        }
        14 => {
            for i in 0..7 {
                sum += G14_W[i] * (f(x - G14_A[i] * h, args)? + f(x + G14_A[i] * h, args)?);
            }
        }
        _ => return Err("n_gauss must be 6, 8, 10, 12, or 14"),
    }
    Ok(h * sum)
}

// Gauss points: abscissae `A` and weights `W` (just half of the array, due to symmetry)

const G6_A: [f64; 3] = [
    6.6120938646626448154108857124811038374900817871093750000000000000E-01,
    2.3861918608319690471297747080825502052903175354003906250000000000E-01,
    9.3246951420315205005806546978419646620750427246093750000000000000E-01,
];

const G6_W: [f64; 3] = [
    3.6076157304813860626779842277755960822105407714843750000000000000E-01,
    4.6791393457269103706153146049473434686660766601562500000000000000E-01,
    1.7132449237917035667067011672770604491233825683593750000000000000E-01,
];

const G8_A: [f64; 4] = [
    1.8343464249564980783624434934608871117234230041503906250000000000E-01,
    5.2553240991632899081764662696514278650283813476562500000000000000E-01,
    7.9666647741362672796583410672610625624656677246093750000000000000E-01,
    9.6028985649753628717206765941227786242961883544921875000000000000E-01,
];

const G8_W: [f64; 4] = [
    3.6268378337836199021282368448737543076276779174804687500000000000E-01,
    3.1370664587788726906936176419549155980348587036132812500000000000E-01,
    2.2238103445337448205165742365352343767881393432617187500000000000E-01,
    1.0122853629037625866615712766360957175493240356445312500000000000E-01,
];

const G10_A: [f64; 5] = [
    1.4887433898163121570590305964287836104631423950195312500000000000E-01,
    4.3339539412924721339948064269265159964561462402343750000000000000E-01,
    6.7940956829902443558921731892041862010955810546875000000000000000E-01,
    8.6506336668898453634568568304530344903469085693359375000000000000E-01,
    9.7390652851717174343093574861995875835418701171875000000000000000E-01,
];

const G10_W: [f64; 5] = [
    2.9552422471475287002462550844938959926366806030273437500000000000E-01,
    2.6926671930999634962944355720537714660167694091796875000000000000E-01,
    2.1908636251598204158774763072869973257184028625488281250000000000E-01,
    1.4945134915058058688863695806503528729081153869628906250000000000E-01,
    6.6671344308688137991758537737041478976607322692871093750000000000E-02,
];

const G12_A: [f64; 6] = [
    1.2523340851146891328227184203569777309894561767578125000000000000E-01,
    3.6783149899818018413455433801573235541582107543945312500000000000E-01,
    5.8731795428661748292853417297010309994220733642578125000000000000E-01,
    7.6990267419430469253427418152568861842155456542968750000000000000E-01,
    9.0411725637047490877762356831226497888565063476562500000000000000E-01,
    9.8156063424671924355635610481840558350086212158203125000000000000E-01,
];

const G12_W: [f64; 6] = [
    2.4914704581340277322887288846686715260148048400878906250000000000E-01,
    2.3349253653835480570855054338608169928193092346191406250000000000E-01,
    2.0316742672306592476516584611090365797281265258789062500000000000E-01,
    1.6007832854334622108005703466915292665362358093261718750000000000E-01,
    1.0693932599531842664308811663431697525084018707275390625000000000E-01,
    4.7175336386511827757583859010992455296218395233154296875000000000E-02,
];

const G14_A: [f64; 7] = [
    1.0805494870734366763542766420869156718254089355468750000000000000E-01,
    3.1911236892788974461865336706978268921375274658203125000000000000E-01,
    5.1524863635815409956819621584145352244377136230468750000000000000E-01,
    6.8729290481168547888302100545843131840229034423828125000000000000E-01,
    8.2720131506976501967187687114346772432327270507812500000000000000E-01,
    9.2843488366357351804225572777795605361461639404296875000000000000E-01,
    9.8628380869681231413181876632734201848506927490234375000000000000E-01,
];

const G14_W: [f64; 7] = [
    2.1526385346315779489856367945321835577487945556640625000000000000E-01,
    2.0519846372129560418962057610769988968968391418457031250000000000E-01,
    1.8553839747793782199991596826293971389532089233398437500000000000E-01,
    1.5720316715819354635996774049999658018350601196289062500000000000E-01,
    1.2151857068790318516793291792055242694914340972900390625000000000E-01,
    8.0158087159760207929259934189758496358990669250488281250000000000E-02,
    3.5119460331751860271420895287519670091569423675537109375000000000E-02,
];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{gauss, QuadParams, Quadrature};
    use crate::algo::testing::get_test_functions;
    use crate::algo::NoArgs;
    use crate::approx_eq;
    use crate::base::format_fortran;
    use crate::math::{PI, SQRT_2};

    #[test]
    fn quad_params_captures_errors() {
        let mut params = QuadParams::new();
        params.n_iteration_max = 0;
        assert_eq!(params.validate().err(), Some("n_iteration_max must be ≥ 2"));
        params.n_iteration_max = 2;
        params.tolerance = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("the tolerance must be ≥ 10.0 * f64::EPSILON")
        );
        params.n_iteration_max = 2;
        params.tolerance = 1e-7;
        params.n_gauss = 7;
        assert_eq!(params.validate().err(), Some("n_gauss must be 6, 8, 10, 12, or 14"));
    }

    #[test]
    fn gauss_captures_errors() {
        struct Args {
            count: usize,
            target: usize,
        }
        let mut f = |x, a: &mut Args| {
            if a.count == a.target {
                return Err("stop");
            }
            a.count += 1;
            Ok(x)
        };
        let args = &mut Args { count: 0, target: 0 };
        assert_eq!(
            gauss(7, 0.0, 0.0, args, &mut f).err(),
            Some("n_gauss must be 6, 8, 10, 12, or 14")
        );
        for n_gauss in [6, 8, 10, 12, 14] {
            args.count = 0;
            args.target = 0;
            assert_eq!(gauss(n_gauss, 0.0, 0.0, args, &mut f).err(), Some("stop"));
            args.count = 0;
            args.target = 1;
            assert_eq!(gauss(n_gauss, 0.0, 0.0, args, &mut f).err(), Some("stop"));
        }
    }

    #[test]
    fn quadrature_captures_errors_1() {
        let f = |x, _: &mut NoArgs| Ok(f64::sin(x * x - 1.0));
        let args = &mut 0;
        let mut quad = Quadrature::new();
        assert_eq!(
            quad.integrate(0.0, 0.0, None, args, f).err(),
            Some("the lower and upper bounds must be different from each other")
        );
        let mut params = QuadParams::new();
        params.n_iteration_max = 0;
        assert_eq!(
            quad.integrate(0.0, 1.0, Some(params), args, f).err(),
            Some("n_iteration_max must be ≥ 2")
        );
        let mut params = QuadParams::new();
        params.n_iteration_max = 2;
        params.n_gauss = 6;
        assert_eq!(
            quad.integrate(-5.0, 5.0, Some(params), args, f).err(),
            Some("integrate failed to converge")
        );
    }

    #[test]
    fn quadrature_works_1() {
        // compare with Fortran code
        // f(x) = sin(x)
        let ii_ana = 2.0;

        let mut params = QuadParams::new();
        params.tolerance = 100_000.0 * f64::EPSILON;

        let mut quad = Quadrature::new();
        let args = &mut 0;

        for (n_gauss, n_f_eval) in [(6, 42), (8, 24), (10, 30), (12, 36), (14, 42)] {
            params.n_gauss = n_gauss;

            let (ii, stats) = quad
                .integrate(0.0, PI, Some(params), args, |x, _| Ok(f64::sin(x)))
                .unwrap();

            println!("\n=================================================");
            println!("\nn_gauss = {}", n_gauss);
            println!("\nI = {}", ii);
            println!("\n{}", stats);

            approx_eq(ii, ii_ana, 1e-15);
            assert_eq!(stats.n_function, n_f_eval);
            if n_gauss > 6 {
                assert_eq!(stats.n_iterations, 1);
            }
        }
    }

    #[test]
    fn quadrature_works_2() {
        // compare with Fortran code
        // f(x) = 0.092834 sin(77.0001 + 19.87 x) in [-2.34567, 12.34567]
        let a = -2.34567;
        let b = 12.34567;
        let amp = 0.092834;
        let freq = 19.87;
        let phase = 77.0001;
        let ii_ana = (amp * (f64::cos(a * freq + phase) - f64::cos(b * freq + phase))) / freq;

        let mut params = QuadParams::new();
        params.tolerance = 100_000.0 * f64::EPSILON;

        let mut quad = Quadrature::new();
        let args = &mut 0;

        for (n_gauss, n_f_eval) in [(6, 3066), (8, 2040), (10, 1270), (12, 1476), (14, 882)] {
            params.n_gauss = n_gauss;

            let (ii, stats) = quad
                .integrate(a, b, Some(params), args, |x, _| Ok(amp * f64::sin(freq * x + phase)))
                .unwrap();

            println!("\n=================================================");
            println!("\nn_gauss = {}", n_gauss);
            println!("\nI = {}", ii);
            println!("\n{}", stats);

            approx_eq(ii, ii_ana, 1e-15);
            assert_eq!(stats.n_function, n_f_eval);
        }
    }

    #[test]
    fn quadrature_works_3() {
        // compare with Fortran code
        // f(x) = log(2 Cos(x/2))
        let ii_ana = 0.0;

        let mut params = QuadParams::new();
        params.tolerance = 100_000.0 * f64::EPSILON;

        let mut quad = Quadrature::new();
        let args = &mut 0;

        for (n_gauss, n_f_eval) in [(6, 1674), (8, 2040), (10, 2550), (12, 3060), (14, 3570)] {
            params.n_gauss = n_gauss;

            let (ii, stats) = quad
                .integrate(-PI, PI, Some(params), args, |x, _| Ok(f64::ln(2.0 * f64::cos(x / 2.0))))
                .unwrap();

            println!("I = {}", format_fortran(ii));
            println!("\n{}", stats);

            approx_eq(ii, ii_ana, 1e-10);
            assert_eq!(stats.n_function, n_f_eval);
        }
    }

    #[test]
    fn quadrature_works_4() {
        let mut quad = Quadrature::new();
        let args = &mut 0;
        for test in &get_test_functions() {
            println!("\n===================================================================");
            println!("\n{}", test.name);
            if let Some(data) = test.integral {
                let (ii, stats) = quad.integrate(data.0, data.1, None, args, test.f).unwrap();
                println!("\nI = {}", ii);
                println!("\n{}", stats);
                approx_eq(ii, data.2, test.tol_integral);
            }
        }
        println!("\n===================================================================\n");
    }

    #[test]
    fn quadrature_works_5() {
        // check that ∫_a^b f(x) dx = - ∫_b^a f(x) dx
        let mut quad = Quadrature::new();
        let args = &mut 0;
        let (ii, _) = quad.integrate(0.0, PI, None, args, |x, _| Ok(f64::sin(x))).unwrap();
        let (mii, _) = quad.integrate(PI, 0.0, None, args, |x, _| Ok(f64::sin(x))).unwrap();
        assert_eq!(ii, -mii);
    }

    #[test]
    fn quadrature_edge_cases_work() {
        let f = |x, _: &mut NoArgs| Ok(x);
        let mut quad = Quadrature::new();
        let args = &mut 0;

        //
        // c <= 0.1
        //
        let b = 1.0;
        let a = 0.9 * b;
        let (ii, stats) = quad.integrate(a, b, None, args, f).unwrap();
        println!("\nI = {}", ii);
        println!("\n{}", stats);
        approx_eq(ii, 0.095, 1e-15);

        //
        // n_bit - nib - 7 = 46 - nib = 0
        //
        let c = 1.0 / (35184372088832.0 * SQRT_2);
        let b = 1.0;
        let a = (1.0 - c) * b;
        assert_eq!(
            quad.integrate(a, b, None, args, f).err(),
            Some("the lower and upper bounds must not be so close one from another")
        );
    }
}
