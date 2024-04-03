use super::{frexp, ldexp};

/// Evaluates the modified Bessel function I0(x) for any real x
///
/// Special cases:
///
///	* `I0(0.0) = 1.0`
pub fn bessel_mod_i0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let ax = f64::abs(x);
    if ax < 15.0 {
        // rational approximation
        let y = x * x;
        return poly(&I0P, 13, y) / poly(&I0Q, 4, 225.0 - y);
    }
    // rational approximation with exp(x)/sqrt(x) factored out.
    let z = 1.0 - 15.0 / ax;
    return f64::exp(ax) * poly(&I0PP, 4, z) / (poly(&I0QQ, 5, z) * f64::sqrt(ax));
}

/// Evaluates the modified Bessel function I1(x) for any real x
pub fn bessel_mod_i1(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let ax = f64::abs(x);
    if ax < 15.0 {
        // rational approximation
        let y = x * x;
        return x * poly(&I1P, 13, y) / poly(&I1Q, 4, 225.0 - y);
    }
    // rational approximation with exp(x)/sqrt(x) factored out
    let z = 1.0 - 15.0 / ax;
    let ans = f64::exp(ax) * poly(&I1PP, 4, z) / (poly(&I1QQ, 5, z) * f64::sqrt(ax));
    if x > 0.0 {
        ans
    } else {
        -ans
    }
}

// controls the accuracy in bessel_mod_in
const ACC: f64 = 200.0;

// half DBL_MAX_EXP
const HALF_MAX_EXP: i32 = f64::MAX_EXP / 2;

/// Evaluates the modified Bessel function In(x) for any real x and n ≥ 0
pub fn bessel_mod_in(n: usize, x: f64) -> f64 {
    if n == 0 {
        return bessel_mod_i0(x);
    }
    if n == 1 {
        return bessel_mod_i1(x);
    }
    if x * x <= 8.0 * f64::MIN_POSITIVE {
        return 0.0;
    }
    let tox = 2.0 / f64::abs(x);
    let mut bip = 0.0;
    let mut bi = 1.0;
    let mut j = 2 * (n + (f64::sqrt(ACC * (n as f64)) as usize));
    let mut ans: f64 = 0.0;
    while j > 0 {
        // downward recurrence
        let bim = bip + (j as f64) * tox * bi;
        bip = bi;
        bi = bim;
        let (_, k) = frexp(bi);
        if k > HALF_MAX_EXP {
            // re-normalize to prevent overflows
            ans = ldexp(ans, -HALF_MAX_EXP);
            bi = ldexp(bi, -HALF_MAX_EXP);
            bip = ldexp(bip, -HALF_MAX_EXP);
        }
        if j == n {
            ans = bip;
        }
        j -= 1;
    }
    ans *= bessel_mod_i0(x) / bi; // normalize using I0
    if x < 0.0 && (n & 1) != 0 {
        // negative and odd
        return -ans;
    }
    ans
}

/// Evaluates the modified Bessel function K0(x) for positive real x
///
/// Special cases:
///
/// * `K0(x < 0.0) = NaN`
/// * `K0(0.0)     = Inf`
pub fn bessel_mod_k0(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        // use two rational approximations
        let z = x * x;
        let term = poly(&K0PI, 4, z) * f64::ln(x) / poly(&K0QI, 2, 1. - z);
        return poly(&K0P, 4, z) / poly(&K0Q, 2, 1. - z) - term;
    }
    // rational approximation with exp(-x) / sqrt(x) factored out
    let z = 1.0 / x;
    f64::exp(-x) * poly(&K0PP, 7, z) / (poly(&K0QQ, 7, z) * f64::sqrt(x))
}

/// Evaluates the modified Bessel function K1(x) for positive real x
///
/// Special cases:
///
/// * `K1(x < 0.0) = NaN`
/// * `K1(0.0)     = Inf`
pub fn bessel_mod_k1(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        // use two rational approximations
        let z = x * x;
        let term = poly(&K1PI, 4, z) * f64::ln(x) / poly(&K1QI, 2, 1. - z);
        return x * (poly(&K1P, 4, z) / poly(&K1Q, 2, 1. - z) + term) + 1. / x;
    }
    // rational approximation with exp(-x)/sqrt(x) factored out
    let z = 1.0 / x;
    f64::exp(-x) * poly(&K1PP, 7, z) / (poly(&K1QQ, 7, z) * f64::sqrt(x))
}

/// Evaluates the modified Bessel function Kn(x) for positive x and n ≥ 0
///
/// Special cases:
///
/// * `Kn(x < 0.0) = NaN`
/// * `Kn(0.0)     = Inf`
pub fn bessel_mod_kn(n: i32, x: f64) -> f64 {
    if n == 0 {
        return bessel_mod_k0(x);
    }
    if n == 1 {
        return bessel_mod_k1(x);
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    let tox = 2.0 / x;
    let mut bkm = bessel_mod_k0(x); // upward recurrence for all x
    let mut bk = bessel_mod_k1(x);
    for j in 1..n {
        let bkp = bkm + (j as f64) * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}

/// evaluates a polynomial for the modified Bessel functions
fn poly(cof: &[f64], n: i32, x: f64) -> f64 {
    let mut ans = cof[n as usize];
    let mut i = n - 1;
    while i >= 0 {
        ans = ans * x + cof[i as usize];
        i -= 1;
    }
    ans
}

// constants --------------------------------------------------------------------

const I0P: [f64; 14] = [
    9.999999999999997e-1,
    2.466405579426905e-1,
    1.478980363444585e-2,
    3.826993559940360e-4,
    5.395676869878828e-6,
    4.700912200921704e-8,
    2.733894920915608e-10,
    1.115830108455192e-12,
    3.301093025084127e-15,
    7.209167098020555e-18,
    1.166898488777214e-20,
    1.378948246502109e-23,
    1.124884061857506e-26,
    5.498556929587117e-30,
];

const I0Q: [f64; 5] = [
    4.463598170691436e-1,
    1.702205745042606e-3,
    2.792125684538934e-6,
    2.369902034785866e-9,
    8.965900179621208e-13,
];

const I0PP: [f64; 5] = [
    1.192273748120670e-1,
    1.947452015979746e-1,
    7.629241821600588e-2,
    8.474903580801549e-3,
    2.023821945835647e-4,
];

const I0QQ: [f64; 6] = [
    2.962898424533095e-1,
    4.866115913196384e-1,
    1.938352806477617e-1,
    2.261671093400046e-2,
    6.450448095075585e-4,
    1.529835782400450e-6,
];

const I1P: [f64; 14] = [
    5.000000000000000e-1,
    6.090824836578078e-2,
    2.407288574545340e-3,
    4.622311145544158e-5,
    5.161743818147913e-7,
    3.712362374847555e-9,
    1.833983433811517e-11,
    6.493125133990706e-14,
    1.693074927497696e-16,
    3.299609473102338e-19,
    4.813071975603122e-22,
    5.164275442089090e-25,
    3.846870021788629e-28,
    1.712948291408736e-31,
];

const I1Q: [f64; 5] = [
    4.665973211630446e-1,
    1.677754477613006e-3,
    2.583049634689725e-6,
    2.045930934253556e-9,
    7.166133240195285e-13,
];

const I1PP: [f64; 5] = [
    1.286515211317124e-1,
    1.930915272916783e-1,
    6.965689298161343e-2,
    7.345978783504595e-3,
    1.963602129240502e-4,
];

const I1QQ: [f64; 6] = [
    3.309385098860755e-1,
    4.878218424097628e-1,
    1.663088501568696e-1,
    1.473541892809522e-2,
    1.964131438571051e-4,
    -1.034524660214173e-6,
];

const K0PI: [f64; 5] = [
    1.0,
    2.346487949187396e-1,
    1.187082088663404e-2,
    2.150707366040937e-4,
    1.425433617130587e-6,
];

const K0QI: [f64; 3] = [9.847324170755358e-1, 1.518396076767770e-2, 8.362215678646257e-5];

const K0P: [f64; 5] = [
    1.159315156584126e-1,
    2.770731240515333e-1,
    2.066458134619875e-2,
    4.574734709978264e-4,
    3.454715527986737e-6,
];

const K0Q: [f64; 3] = [9.836249671709183e-1, 1.627693622304549e-2, 9.809660603621949e-5];

const K0PP: [f64; 8] = [
    1.253314137315499,
    1.475731032429900e1,
    6.123767403223466e1,
    1.121012633939949e2,
    9.285288485892228e1,
    3.198289277679660e1,
    3.595376024148513,
    6.160228690102976e-2,
];

const K0QQ: [f64; 8] = [
    1.0,
    1.189963006673403e1,
    5.027773590829784e1,
    9.496513373427093e1,
    8.318077493230258e1,
    3.181399777449301e1,
    4.443672926432041,
    1.408295601966600e-1,
];

const K1PI: [f64; 5] = [
    0.5,
    5.598072040178741e-2,
    1.818666382168295e-3,
    2.397509908859959e-5,
    1.239567816344855e-7,
];

const K1QI: [f64; 3] = [9.870202601341150e-1, 1.292092053534579e-2, 5.881933053917096e-5];

const K1P: [f64; 5] = [
    -3.079657578292062e-1,
    -8.109417631822442e-2,
    -3.477550948593604e-3,
    -5.385594871975406e-5,
    -3.110372465429008e-7,
];

const K1Q: [f64; 3] = [9.861813171751389e-1, 1.375094061153160e-2, 6.774221332947002e-5];

const K1PP: [f64; 8] = [
    1.253314137315502,
    1.457171340220454e1,
    6.063161173098803e1,
    1.147386690867892e2,
    1.040442011439181e2,
    4.356596656837691e1,
    7.265230396353690,
    3.144418558991021e-1,
];

const K1QQ: [f64; 8] = [
    1.0,
    1.125154514806458e1,
    4.427488496597630e1,
    7.616113213117645e1,
    5.863377227890893e1,
    1.850303673841586e1,
    1.857244676566022,
    2.538540887654872e-2,
];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{bessel_mod_i0, bessel_mod_i1, bessel_mod_in, bessel_mod_k0, bessel_mod_k1, bessel_mod_kn};
    use crate::approx_eq;

    #[test]
    fn bessel_mod_i0_works() {
        assert_eq!(bessel_mod_i0(0.0), 1.0);

        // Mathematica: X = {-4, 1, 16}; Table[{X[[i]], N[BesselI[0, X[[i]]], 50]}, {i, 1, 3}]
        #[rustfmt::skip]
        let mathematica = [
            (-4.0, 1e-14, 11.301921952136330496356270183217102497412616594435),
            ( 1.0, 1e-15, 1.2660658777520083355982446252147175376076703113550),
            (16.0, 1e-9, 893446.22792010501707086403097618845427806981885885),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(bessel_mod_i0(x), reference, tol);
        }
    }

    #[test]
    fn bessel_mod_i1_works() {
        assert_eq!(bessel_mod_i1(0.0), 0.0);

        // Mathematica: X = {-4, 1, 16, -15}; Table[{X[[i]], N[BesselI[1, X[[i]]], 50]}, {i, 1, 4}]
        #[rustfmt::skip]
        let mathematica = [
            ( -4.0, 1e-14, -9.7594651537044499094751925673126809000559703332530),
            (  1.0, 1e-15, 0.56515910399248502720769602760986330732889962162109),
            ( 16.0, 1e-9, 865059.43585483947141807621749529138726744748495488),
            (-15.0, 1e-9, -328124.92197020639673369815024598440792655583311133),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(bessel_mod_i1(x), reference, tol);
        }
    }

    #[test]
    fn bessel_mod_in_with_n0_n1_works() {
        assert_eq!(bessel_mod_in(0, 0.0), 1.0);
        assert_eq!(bessel_mod_in(1, 0.0), 0.0);
    }

    #[test]
    fn bessel_mod_in_with_positive_n_works() {
        // Mathematica: X = {-4, 1, 2, -5}; Table[{n, X[[i]], N[BesselI[n, X[[i]]], 50]}, {n, 2, 5}, {i, 1, 4}]
        #[rustfmt::skip]
        let mathematica = [
            (2, -4.0, 1e-50, 6.4221893752841055416186738995607620473846314278089),
            (2,  1.0, 1e-16, 0.13574766976703828118285256999499092294987106811278),
            (2,  2.0, 1e-15, 0.68894844769873820405495001581186710533136294328992),
            (2, -5.0, 1e-14, 17.505614966624236014887011895180415898253112084901),
            (3, -4.0, 1e-50, -3.3372757784203443678565186677519188526713389054441),
            (3,  1.0, 1e-17, 0.022168424924331902476285747629899615529415349169979),
            (3,  2.0, 1e-16, 0.21273995923985265527235439337593203729175227291569),
            (3, -5.0, 1e-50, -10.331150169151138387233440935615675741962384700378),
            (4, -4.0, 1e-15, 1.4162757076535889898338958979328837683776230696427),
            (4,  1.0, 1e-17, 0.0027371202210468663251380842155932297733789730929026),
            (4,  2.0, 1e-16, 0.050728569979180238237886835684070993456106124542847),
            (4, -5.0, 1e-50, 5.1082347636428699502068827724416050078982504444481),
            (5, -4.0, 1e-50, -0.50472436311316638818872687188615131591609276615874),
            (5,  1.0, 1e-50, 0.00027146315595697187518107390515377734238356442675814),
            (5,  2.0, 1e-17, 0.0098256793231317023208070506396480634673277747443042),
            (5, -5.0, 1e-15, -2.1579745473225464669024284997091077293251839892608),
        ];
        for (n, x, tol, reference) in mathematica {
            // println!("n = {}, x = {:?}", n, x);
            approx_eq(bessel_mod_in(n, x), reference, tol);
        }
    }

    #[test]
    fn bessel_mod_in_edge_cases_work() {
        //
        // x * x <= 8.0 MIN_POSITIVE
        //
        assert_eq!(bessel_mod_in(2, 0.99 * f64::sqrt(8.0 * f64::MIN_POSITIVE)), 0.0);

        //
        // k > HALF_MAX_EXP
        //
        // Mathematica: N[BesselI[2, 10^-153], 100]
        approx_eq(
            bessel_mod_in(2, 1e-153),
            1.250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-307,
            1e-322,
        );
    }

    #[test]
    fn bessel_mod_k0_works() {
        assert!(bessel_mod_k0(-1.0).is_nan());
        assert_eq!(bessel_mod_k0(0.0), f64::INFINITY);

        // Mathematica: X = {0.5, 1, 2}; Table[{X[[i]], N[BesselK[0, X[[i]]], 50]}, {i, 1, 3}]
        #[rustfmt::skip]
        let mathematica = [
            (0.5, 1e-15, 0.924419071227666),
            (1.0, 1e-15, 0.42102443824070833333562737921260903613621974822666),
            (2.0, 1e-16, 0.11389387274953343565271957493248183299832662438881),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(bessel_mod_k0(x), reference, tol);
        }
    }

    #[test]
    fn bessel_mod_k1_works() {
        assert!(bessel_mod_k1(-1.0).is_nan());
        assert_eq!(bessel_mod_k1(0.0), f64::INFINITY);

        // Mathematica: X = {0.5, 1, 2}; Table[{X[[i]], N[BesselK[0, X[[i]]], 50]}, {i, 1, 3}]
        #[rustfmt::skip]
        let mathematica = [
            (0.5, 1e-50, 1.656441120003301),
            (1.0, 1e-50, 0.60190723019723457473754000153561733926158688996811),
            (2.0, 1e-15, 0.13986588181652242728459880703541102388723458484152),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(bessel_mod_k1(x), reference, tol);
        }
    }

    #[test]
    fn bessel_mod_kn_with_n0_n1_works() {
        assert_eq!(bessel_mod_kn(0, 0.0), f64::INFINITY);
        assert_eq!(bessel_mod_kn(1, 0.0), f64::INFINITY);
    }

    #[test]
    fn bessel_mod_kn_works() {
        assert!(bessel_mod_kn(2, -1.0).is_nan());
        assert_eq!(bessel_mod_kn(2, 0.0), f64::INFINITY);

        // Mathematica: X = {1, 4, 10}; Table[{n, X[[i]], N[BesselK[n, X[[i]]], 50]}, {n, 2, 4}, {i, 1, 3}]
        #[rustfmt::skip]
        let mathematica = [
            (2,  1.0, 1e-15, 1.6248388986351774828107073822838437146593935281629),
            (2,  4.0, 1e-50, 0.017401425529487240004937285970236523466929816382787), 
            (2, 10.0, 1e-20, 0.000021509817006932768730664564423967127249206846180873),
            (3,  1.0, 1e-14, 7.1012628247379445059803695306709921978991610026196),
            (3,  4.0, 1e-17, 0.029884924416755671475321465951042591950771401371413),
            (3, 10.0, 1e-19, 0.000027252700256598692089082683891958525581349618574983),
            (4,  1.0, 1e-13, 44.232415847062844518692924566309796902054359543880),
            (4,  4.0, 1e-17, 0.062228812154620747217919484896800411393086918439907),
            (4, 10.0, 1e-19, 0.000037861437160891983984114174759142242598016617325863),
        ];
        for (n, x, tol, reference) in mathematica {
            // println!("n = {}, x = {:?}", n, x);
            approx_eq(bessel_mod_kn(n, x), reference, tol);
        }
    }
}
