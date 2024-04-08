use num_complex::Complex64;

/// Calculates the imaginary unit (i) raised to the n power
///
/// Computes:
///
/// ```text
/// iⁿ = (√-1)ⁿ
/// ```
///
/// Some results with positive n:
///
/// ```text
///    odd       even        odd      even
/// i¹ = i   i²  = -1   i³  = -i   i⁴  = 1
/// i⁵ = i   i⁶  = -1   i⁷  = -i   i⁸  = 1
/// i⁹ = i   i¹⁰ = -1   i¹¹ = -i   i¹² = 1
/// ```
///
/// Some results with negative n (even n yields the same results as above; odd n yields the negative of the above results):
///
/// ```text
///     odd         even        odd       even
/// i⁻¹ = -i   i⁻²  = -1   i⁻³  = i   i⁻⁴  = 1
/// i⁻⁵ = -i   i⁻⁶  = -1   i⁻⁷  = i   i⁻⁸  = 1
/// i⁻⁹ = -i   i⁻¹⁰ = -1   i⁻¹¹ = i   i⁻¹² = 1
/// ```
pub fn complex_imag_powi(n: i32) -> Complex64 {
    if n == 0 {
        Complex64::new(1.0, 0.0)
    } else if n > 0 {
        match n % 4 {
            1 => Complex64::new(0.0, 1.0),  // i
            2 => Complex64::new(-1.0, 0.0), // -1
            3 => Complex64::new(0.0, -1.0), // -i
            _ => Complex64::new(1.0, 0.0),  // 1
        }
    } else {
        match (-n) % 4 {
            1 => Complex64::new(0.0, -1.0), // -i
            2 => Complex64::new(-1.0, 0.0), // -1
            3 => Complex64::new(0.0, 1.0),  // i
            _ => Complex64::new(1.0, 0.0),  // 1
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_imag_powi;
    use num_complex::Complex64;

    #[test]
    fn complex_imag_powi_works() {
        let mut n: i32 = -12;
        let i = Complex64::new(0.0, 1.0);
        while n < 12 {
            // println!("n = {:>3}", n);
            assert_eq!(complex_imag_powi(n), i.powi(n));
            n += 1;
        }
    }

    /*
    #[test]
    #[rustfmt::skip]
    fn num_complex_check() {
        let o = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let i2  = i * i;
        let i3  = i * i2;
        let i4  = i * i3;
        let i5  = i * i4;
        let i6  = i * i5;
        let i7  = i * i6;
        let i8  = i * i7;
        let i9  = i * i8;
        let i10 = i * i9;
        let i11 = i * i10;
        let i12 = i * i11;
        let a = o / i  ; let b =i.powi( -1); println!("i⁻¹  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i2 ; let b =i.powi( -2); println!("i⁻²  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i3 ; let b =i.powi( -3); println!("i⁻³  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i4 ; let b =i.powi( -4); println!("i⁻⁴  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        println!();
        let a = o / i5 ; let b =i.powi( -5); println!("i⁻⁵  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i6 ; let b =i.powi( -6); println!("i⁻⁶  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i7 ; let b =i.powi( -7); println!("i⁻⁷  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i8 ; let b =i.powi( -8); println!("i⁻⁸  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        println!();
        let a = o / i9 ; let b =i.powi( -9); println!("i⁻⁹  = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i10; let b =i.powi(-10); println!("i⁻¹⁰ = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i11; let b =i.powi(-11); println!("i⁻¹¹ = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
        let a = o / i12; let b =i.powi(-12); println!("i⁻¹² = {:>6}  =?  {:>6}", a, b); assert_eq!(a, b);
    }
    */
}
