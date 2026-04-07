use super::chebyshev_tn;
use super::chebyshev_tn_deriv1;
/// Evaluates the Chebyshev polynomial of second kind Un(x) using the trigonometric functions
///
/// ```text
///         ⎧  (-1)ⁿ sinh[(n+1)⋅acosh(-x)] / sinh[acosh(-x)]   if x < -1
/// Uₙ(x) = ⎨        sinh[(n+1)⋅acosh( x)] / sinh[acosh( x)]   if x > 1
///         ⎩        sin [(n+1)⋅acos ( x)] / sin [acos ( x)]   if |x| ≤ 1
/// ```
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
///
/// | n | Uₙ(x)               | dUₙ/dx(x)           | d²Uₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------ |:----------------|
/// | 0 | 1                   | 0                  | 0               |
/// | 1 | 2 x                 | 2                  | 0               |
/// | 2 | -1 + 4 x²           | 8 x                | 8               |
/// | 3 | -4 x + 8 x³         | -4 + 24 x²         | 48 x            |
/// | 4 | 1 - 12 x² + 16 x⁴   | -24 x + 64 x³      | -24 + 192 x²    |
/// | 5 | 6x - 32 x³ + 32 x⁵  | 6 - 96 x² + 160 x⁴ | -192 x + 640 x³ |
/// |...| ...                 | ...                | ....            |
///
/// # Examples
///
/// ![Un](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/math_chebyshev_functions_un.svg)
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::chebyshev_un(4, 0.25), 0.31250000000000000, 1e-15);
/// ```
pub fn chebyshev_un(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    
    let m = (n + 1) as f64;  
    
    if x < -1.0 {
        let acosh_val = (-x).acosh();
        let sinh_val = (m * acosh_val).sinh() / acosh_val.sinh();
        
        if n % 2 == 0 {
            return sinh_val;
        } else {
            return -sinh_val;
        }
    } else if x > 1.0 {
        let acosh_val = x.acosh();
        return (m * acosh_val).sinh() / acosh_val.sinh();
    } else {
        if (x - 1.0).abs() < 1e-12 {
            return m;
        } else if (x + 1.0).abs() < 1e-12 {
            if n % 2 == 0 {
                return m;
            } else {
                return -m;
            }
        }
        
        let acos_val = x.acos();
        return (m * acos_val).sin() / acos_val.sin();
    }
}
/// Computes the first derivative of the second kind of Chebyshev U(n, x) function
///
/// ```text
/// dUₙ(x)
/// ——————
///   dx
/// ```
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
///
/// | n | Uₙ(x)               | dUₙ/dx(x)           | d²Uₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------ |:----------------|
/// | 0 | 1                   | 0                  | 0               |
/// | 1 | 2 x                 | 2                  | 0               |
/// | 2 | -1 + 4 x²           | 8 x                | 8               |
/// | 3 | -4 x + 8 x³         | -4 + 24 x²         | 48 x            |
/// | 4 | 1 - 12 x² + 16 x⁴   | -24 x + 64 x³      | -24 + 192 x²    |
/// | 5 | 6x - 32 x³ + 32 x⁵  | 6 - 96 x² + 160 x⁴ | -192 x + 640 x³ |
/// |...| ...                 | ...                | ....            |
///
/// # Examples
///
/// ![dUn/dx](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/math_chebyshev_functions_dun.svg)
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::chebyshev_un_deriv1(4, 0.25), -5.0 , 1e-15);
/// ```
pub fn chebyshev_un_deriv1(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    
    let n_f64 = n as f64;
    let m = n_f64 + 1.0;
    
    if (x - 1.0).abs() < 1e-12 {
        return n_f64 * (n_f64 + 1.0) * (n_f64 + 2.0) / 3.0;
    }
    
    if (x + 1.0).abs() < 1e-12 {
        let deriv_1 = n_f64 * (n_f64 + 1.0) * (n_f64 + 2.0) / 3.0;
        if n % 2 == 0 {
            return -deriv_1;
        } else {
            return deriv_1;
        }
    }

    let u_n = chebyshev_un(n, x);
    let t_nplus1 = chebyshev_tn(n + 1, x);
    let denom = 1.0 - x * x;
    
    (x * u_n - m * t_nplus1) / denom
}
/// Computes the second derivative of the second kind of Chebyshev U(n, x) function
///
/// ```text
/// d²Uₙ(x)
/// ——————
///   dx²
/// ```
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
///
/// | n | Uₙ(x)               | dUₙ/dx(x)           | d²Uₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------ |:----------------|
/// | 0 | 1                   | 0                  | 0               |
/// | 1 | 2 x                 | 2                  | 0               |
/// | 2 | -1 + 4 x²           | 8 x                | 8               |
/// | 3 | -4 x + 8 x³         | -4 + 24 x²         | 48 x            |
/// | 4 | 1 - 12 x² + 16 x⁴   | -24 x + 64 x³      | -24 + 192 x²    |
/// | 5 | 6x - 32 x³ + 32 x⁵  | 6 - 96 x² + 160 x⁴ | -192 x + 640 x³ |
/// |...| ...                 | ...                | ....            |
///
/// # Examples
///
/// ![dUn/dx](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/math_chebyshev_functions_d2un.svg)
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::chebyshev_un_deriv2(4, 0.25), -12.0 , 1e-14);
/// ```
pub fn chebyshev_un_deriv2(n: usize, x: f64) -> f64 {
    if n == 0 || n == 1 {
        return 0.0;
    }
    
    let n_f64 = n as f64;
    let m = n_f64 + 1.0;

    if (x - 1.0).abs() < 1e-12 {
        return n_f64 * (n_f64 - 1.0) * (n_f64 + 1.0) * (n_f64 + 2.0) * (n_f64 + 3.0) / 15.0;
    }
    
    if (x + 1.0).abs() < 1e-12 {
        let deriv2_1 = n_f64 * (n_f64 - 1.0) * (n_f64 + 1.0) * (n_f64 + 2.0) * (n_f64 + 3.0) / 15.0;
        if n % 2 == 0 {
            return deriv2_1;
        } else {
            return -deriv2_1;
        }
    }

    let u_n = chebyshev_un(n, x);
    let u_n_prime = chebyshev_un_deriv1(n, x);
    let t_nplus1 = chebyshev_tn(n + 1, x);
    let t_nplus1_prime = chebyshev_tn_deriv1(n + 1, x);
    
    let denom = 1.0 - x * x;
    let numerator1 = (u_n + x * u_n_prime - m * t_nplus1_prime) * denom;
    let numerator2 = 2.0 * x * (x * u_n - m * t_nplus1);
    
    (numerator1 + numerator2) / (denom * denom)
}

#[cfg(test)]
mod tests {
    use super::chebyshev_un;
    use super::{chebyshev_un_deriv1, chebyshev_un_deriv2};
    use super::chebyshev_tn_deriv1;
    use crate::{approx_eq};

    #[test]
    fn chebyshev_u_works() {
        let nn = 5;
        let mut xx: Vec<_> = (0..(nn + 1))
            .into_iter()
            .map(|i| -1.5 + (i as f64) * 3.0 / (nn as f64))
            .collect();
        xx.push(-1.0);
        xx.push(1.0);
        
        for x in xx {
            // n = 0: U₀(x) = 1
            approx_eq(chebyshev_un(0, x), 1.0, 1e-14);
            
            // n = 1: U₁(x) = 2x
            approx_eq(chebyshev_un(1, x), 2.0 * x, 1e-14);
            
            // n = 2: U₂(x) = 4x² - 1
            let x2 = x * x;
            approx_eq(chebyshev_un(2, x), 4.0 * x2 - 1.0, 1e-14);
            
            // n = 3: U₃(x) = 8x³ - 4x
            let x3 = x * x2;
            approx_eq(chebyshev_un(3, x), 8.0 * x3 - 4.0 * x, 1e-14);
            
            // n = 4: U₄(x) = 16x⁴ - 12x² + 1
            let x4 = x * x3;
            approx_eq(chebyshev_un(4, x), 16.0 * x4 - 12.0 * x2 + 1.0, 1e-14);
            
            // n = 5: U₅(x) = 32x⁵ - 32x³ + 6x
            let x5 = x * x4;
            approx_eq(chebyshev_un(5, x), 32.0 * x5 - 32.0 * x3 + 6.0 * x, 1e-12);
        }
    }

    #[test]
    fn chebyshev_u_special_values_works() {
        for n in 0..10 {
            let expected = (n + 1) as f64;
            approx_eq(chebyshev_un(n, 1.0), expected, 1e-15);
        }
        
        for n in 0..10 {
            let expected = if n % 2 == 0 {
                (n + 1) as f64
            } else {
                -((n + 1) as f64)
            };
            approx_eq(chebyshev_un(n, -1.0), expected, 1e-15);
        }
        
        for n in 0..5 {
            let even_n = 2 * n;
            let expected_even = if n % 2 == 0 { 1.0 } else { -1.0 };
            approx_eq(chebyshev_un(even_n, 0.0), expected_even, 1e-15);
            
            let odd_n = 2 * n + 1;
            approx_eq(chebyshev_un(odd_n, 0.0), 0.0, 1e-15);
        }
    }

    #[test]
    fn chebyshev_u_recursion_works() {
        let points = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        
        for x in points {
            for n in 1..9 {  
                let u_n = chebyshev_un(n, x);
                let u_n_minus_1 = chebyshev_un(n - 1, x);
                let u_n_plus_1 = chebyshev_un(n + 1, x);
                
                let expected = 2.0 * x * u_n - u_n_minus_1;
                approx_eq(u_n_plus_1, expected, 1e-9);
            }
        }
    }

    #[test]
    fn chebyshev_u_relation_with_chebyshev_t_works() {
        // Tₙ'(x) = n U_{n-1}(x)
        use crate::math::chebyshev_tn_deriv1;
        
        let points = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        
        for x in points {
            for n in 1..6 {
                let t_deriv = chebyshev_tn_deriv1(n, x);
                let u_val = chebyshev_un(n - 1, x);
                let expected = (n as f64) * u_val;
                
                if x.abs() != 1.0 {
                    approx_eq(t_deriv, expected, 1e-12);
                }
            }
        }
    }

    #[test]
    fn chebyshev_u_symmetry_works() {
        let points = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
        
        for x in points {
            for n in 0..6 {
                let u_at_x = chebyshev_un(n, x);
                let u_at_minus_x = chebyshev_un(n, -x);
                let expected = if n % 2 == 0 { u_at_x } else { -u_at_x };
                
                approx_eq(u_at_minus_x, expected, 1e-14);
            }
        }
    }

    #[test]
    fn chebyshev_u_derivatives_exact_formula() {
        let points = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        
        for x in points {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;

            // n=0
            approx_eq(chebyshev_un_deriv1(0, x), 0.0, 1e-15);
            approx_eq(chebyshev_un_deriv2(0, x), 0.0, 1e-15);

            // n=1
            approx_eq(chebyshev_un_deriv1(1, x), 2.0, 1e-14);
            approx_eq(chebyshev_un_deriv2(1, x), 0.0, 1e-15);

            // n=2
            approx_eq(chebyshev_un_deriv1(2, x), 8.0 * x, 1e-14);
            approx_eq(chebyshev_un_deriv2(2, x), 8.0, 1e-14);

            // n=3
            approx_eq(chebyshev_un_deriv1(3, x), -4.0 + 24.0 * x2, 1e-13);
            approx_eq(chebyshev_un_deriv2(3, x), 48.0 * x, 1e-13);

            // n=4
            approx_eq(chebyshev_un_deriv1(4, x), -24.0 * x + 64.0 * x3, 1e-12);
            approx_eq(chebyshev_un_deriv2(4, x), -24.0 + 192.0 * x2, 1e-12);

            // n=5
            approx_eq(chebyshev_un_deriv1(5, x), 6.0 - 96.0 * x2 + 160.0 * x4, 1e-11);
            approx_eq(chebyshev_un_deriv2(5, x), -192.0 * x + 640.0 * x3, 1e-11);
        }
    }

    #[test]
    fn chebyshev_u_deriv1_relation_with_t() {
        let points = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        
        for x in points {
            for n in 1..=6 {
                let t_deriv = chebyshev_tn_deriv1(n, x);
                let u_val = chebyshev_un(n - 1, x);
                let expected = (n as f64) * u_val;
                approx_eq(t_deriv, expected, 1e-12);
            }
        }
    }

    #[test]
    fn chebyshev_u_deriv2_differential_equation() {
        let points = [-2.0, -1.5, -0.9, -0.5, 0.0, 0.5, 0.9, 1.5, 2.0];
        
        for x in points {
            for n in 0..=5 {
                let u_n = chebyshev_un(n, x);
                let u_n_prime = chebyshev_un_deriv1(n, x);
                let u_n_double_prime = chebyshev_un_deriv2(n, x);
                let n_f64 = n as f64;

                let residual = (1.0 - x * x) * u_n_double_prime 
                    - 3.0 * x * u_n_prime 
                    + n_f64 * (n_f64 + 2.0) * u_n;
                
                approx_eq(residual, 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn chebyshev_u_derivatives_boundary_works() {
        approx_eq(chebyshev_un_deriv1(1, 1.0), 2.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(2, 1.0), 8.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(3, 1.0), 20.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(4, 1.0), 40.0, 1e-15);

        approx_eq(chebyshev_un_deriv1(1, -1.0), 2.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(2, -1.0), -8.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(3, -1.0), 20.0, 1e-15);
        approx_eq(chebyshev_un_deriv1(4, -1.0), -40.0, 1e-15);

        approx_eq(chebyshev_un_deriv2(2, 1.0), 8.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(3, 1.0), 48.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(4, 1.0), 168.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(5, 1.0), 448.0, 1e-15);

        approx_eq(chebyshev_un_deriv2(2, -1.0), 8.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(3, -1.0), -48.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(4, -1.0), 168.0, 1e-15);
        approx_eq(chebyshev_un_deriv2(5, -1.0), -448.0, 1e-15);
    }
}