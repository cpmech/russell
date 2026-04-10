//! Solver for cubic equations (a·x³ + b·x² + c·x + d = 0)
//!
//! Algorithm: Cardano method (trigonometric solution for irreducible case)
//! Reference: https://en.wikipedia.org/wiki/Cubic_equation

use std::f64::consts::PI;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum CubicError {
    InvalidLeadingCoeff(f64),
    CalculationError(&'static str),
}

impl std::fmt::Display for CubicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubicError::InvalidLeadingCoeff(a) => {
                write!(f, "cubic equation: leading coefficient a must not be zero (got a={:.e})", a)
            }
            CubicError::CalculationError(msg) => {
                write!(f, "cubic equation calculation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CubicError {}

/// Solve cubic equation: a·x³ + b·x² + c·x + d = 0
///
/// # Inputs
/// - `a`: Coefficient of x³ (must not be zero)
/// - `b`: Coefficient of x²
/// - `c`: Coefficient of x
/// - `d`: Constant term
///
/// # Outputs
/// - `Result<Vec<f64>, CubicError>`: Sorted real roots (complex roots are omitted)
///
/// # Examples
/// ```
/// use russell_lab::algo::solve_cubic;
///
/// // Solve (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6 = 0
/// let roots = solve_cubic(1.0, -6.0, 11.0, -6.0).unwrap();
/// assert!((roots[0] - 1.0).abs() < 1e-12);
/// assert!((roots[1] - 2.0).abs() < 1e-12);
/// assert!((roots[2] - 3.0).abs() < 1e-12);
/// ```
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Result<Vec<f64>, CubicError> {
    if a.abs() < EPS {
        return Err(CubicError::InvalidLeadingCoeff(a));
    }

    let p = b / a;
    let q = c / a;
    let r = d / a;

    let p_sq = p * p;
    let aa = q - p_sq / 3.0;
    let bb = r - (p * q) / 3.0 + (2.0 * p * p_sq) / 27.0;

    let half_b = bb / 2.0;
    let third_a = aa / 3.0;
    let delta = half_b * half_b + third_a.powf(3.0);

    let mut ys = Vec::with_capacity(3);

    match delta {
        d if d > EPS => {
            let sqrt_delta = delta.sqrt();
            let u = (-half_b + sqrt_delta).cbrt();
            let v = (-half_b - sqrt_delta).cbrt();
            ys.push(u + v);
        }

        d if d < -EPS => {
            let sqrt_neg_third_a = (-third_a).sqrt();
            let radicand = -half_b / sqrt_neg_third_a.powf(3.0);
            let radicand_clamped = radicand.clamp(-1.0, 1.0);
            let theta = radicand_clamped.acos();
            let factor = 2.0 * sqrt_neg_third_a;

            let y1 = factor * (theta / 3.0).cos();
            let y2 = factor * ((theta + 2.0 * PI) / 3.0).cos();
            let y3 = factor * ((theta + 4.0 * PI) / 3.0).cos();
            ys.extend_from_slice(&[y1, y2, y3]);
        }

        _ => {
            if half_b.abs() < EPS {
                ys.extend_from_slice(&[0.0, 0.0, 0.0]);
            } else {
                let u = (-half_b).cbrt();
                let y1 = 2.0 * u;
                let y2 = -u;
                ys.extend_from_slice(&[y1, y2, y2]);
            }
        }
    }

    let p3 = p / 3.0;
    let mut xs: Vec<f64> = ys.into_iter().map(|y| y - p3).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(xs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_distinct_real_roots() {
        // (x-1)(x-2)(x-3) = x³ -6x² +11x -6 = 0 → roots: 1,2,3
        let roots = solve_cubic(1.0, -6.0, 11.0, -6.0).unwrap();
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 1.0).abs() < EPS);
        assert!((roots[1] - 2.0).abs() < EPS);
        assert!((roots[2] - 3.0).abs() < EPS);
    }

    #[test]
    fn test_triple_root() {
        // (x+1)³ = x³ +3x² +3x +1 = 0 → root: -1, -1, -1
        let roots = solve_cubic(1.0, 3.0, 3.0, 1.0).unwrap();
        assert_eq!(roots.len(), 3);
        roots.iter().for_each(|r| assert!((r + 1.0).abs() < EPS));
    }

    #[test]
    fn test_invalid_leading_coeff() {
        // a=0 Triggers Error
        let err = solve_cubic(0.0, 1.0, 1.0, 1.0).unwrap_err();
        assert!(matches!(err, CubicError::InvalidLeadingCoeff(0.0)));
    }

    #[test]
    fn test_irreducible_case() {
        // x³ - x = 0 → roots: -1,0,1
        let roots = solve_cubic(1.0, 0.0, -1.0, 0.0).unwrap();
        assert_eq!(roots.len(), 3);
        assert!((roots[0] + 1.0).abs() < EPS);
        assert!((roots[1] - 0.0).abs() < EPS);
        assert!((roots[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_single_real_root() {
        // x³ + x² + x + 1 = 0 → roots: -1, i, -i
        let roots = solve_cubic(1.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(roots.len(), 1);
        assert!((roots[0] + 1.0).abs() < EPS);
    }

    #[test]
    fn test_double_root() {
        // (x-1)²(x-2) = x³ -4x² +5x -2 = 0 → roots: 1, 1, 2
        let roots = solve_cubic(1.0, -4.0, 5.0, -2.0).unwrap();
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 1.0).abs() < EPS);
        assert!((roots[1] - 1.0).abs() < EPS);
        assert!((roots[2] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_near_zero_roots() {
        // x(x-1)(x-2) = x³ -3x² +2x = 0 → roots: 0, 1, 2
        let roots = solve_cubic(1.0, -3.0, 2.0, 0.0).unwrap();
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 0.0).abs() < EPS);
        assert!((roots[1] - 1.0).abs() < EPS);
        assert!((roots[2] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_large_coefficients() {
        // (1000x - 1)(1000x - 2)(1000x - 3) = 10^9x³ - 6*10^6x² + 11*10^3x - 6 = 0
        let a = 1e9;
        let b = -6e6;
        let c = 11e3;
        let d = -6.0;
        let roots = solve_cubic(a, b, c, d).unwrap();
        assert_eq!(roots.len(), 3);
        // Verify each root satisfies the equation
        for root in &roots {
            let value = a * root.powi(3) + b * root.powi(2) + c * root + d;
            assert!(value.abs() < 1e-6, "Root {} does not satisfy the equation: value = {}", root, value);
        }
    }

    #[test]
    fn test_small_coefficients() {
        // (0.001x - 1)(0.001x - 2)(0.001x - 3) = 1e-9x³ - 6e-6x² + 11e-3x - 6 = 0
        let a = 1e-9;
        let b = -6e-6;
        let c = 11e-3;
        let d = -6.0;
        let roots = solve_cubic(a, b, c, d).unwrap();
        assert_eq!(roots.len(), 3);
        // Verify each root satisfies the equation
        for root in &roots {
            let value = a * root.powi(3) + b * root.powi(2) + c * root + d;
            assert!(value.abs() < 1e-6, "Root {} does not satisfy the equation: value = {}", root, value);
        }
    }

    #[test]
    fn test_negative_coefficients() {
        // -x³ + 6x² - 11x + 6 = 0 → roots: 1, 2, 3
        let roots = solve_cubic(-1.0, 6.0, -11.0, 6.0).unwrap();
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 1.0).abs() < EPS);
        assert!((roots[1] - 2.0).abs() < EPS);
        assert!((roots[2] - 3.0).abs() < EPS);
    }

    #[test]
    fn test_fractional_coefficients() {
        // (0.5x - 1)(0.5x - 2)(0.5x - 3) = 0.125x³ - 0.75x² + 1.375x - 0.75 = 0
        let a = 0.125;
        let b = -0.75;
        let c = 1.375;
        let d = -0.75;
        let roots = solve_cubic(a, b, c, d).unwrap();
        assert_eq!(roots.len(), 3);
        // Verify each root satisfies the equation
        for root in &roots {
            let value = a * root.powi(3) + b * root.powi(2) + c * root + d;
            assert!(value.abs() < 1e-6, "Root {} does not satisfy the equation: value = {}", root, value);
        }
    }

    #[test]
    fn test_floating_point_precision() {
        // Test with coefficients near floating point precision limits
        // (x - 1e-15)(x - 2e-15)(x - 3e-15) = x³ - 6e-15x² + 11e-30x - 6e-45 = 0
        let a = 1.0;
        let b = -6e-15;
        let c = 11e-30;
        let d = -6e-45;
        let roots = solve_cubic(a, b, c, d).unwrap();
        assert_eq!(roots.len(), 3);
        // Verify each root satisfies the equation
        for root in &roots {
            let value = a * root.powi(3) + b * root.powi(2) + c * root + d;
            assert!(value.abs() < 1e-40, "Root {} does not satisfy the equation: value = {}", root, value);
        }
    }

    #[test]
    fn test_error_handling() {
        // Test InvalidLeadingCoeff error
        let err = solve_cubic(0.0, 1.0, 1.0, 1.0).unwrap_err();
        assert!(matches!(err, CubicError::InvalidLeadingCoeff(0.0)));
        
        // Test that error messages can be formatted
        let err_msg = format!("{}", err);
        assert!(err_msg.contains("leading coefficient a must not be zero"));
        
        // Test CalculationError error formatting (even though it's not currently used)
        let calc_err = CubicError::CalculationError("test error");
        let calc_err_msg = format!("{}", calc_err);
        assert!(calc_err_msg.contains("cubic equation calculation error: test error"));
    }
}