/// Asserts that two numbers are NaN at the same time or equal to each other (including ±Inf)
///
/// Two values are considered "alike" when:
///
/// * Both are `NaN`, **or**
/// * They are numerically equal **and** share the same sign (so `0.0` and `-0.0` are **not** alike).
///
/// # Panics
///
/// Panics with `"values are not alike"` when the two values are not alike, for example:
///
/// * One is `NaN` and the other is not.
/// * The magnitudes differ (e.g., `1.0` vs `2.0`).
/// * The values are `0.0` and `-0.0` (same magnitude, different signs).
/// * One is `+Inf` and the other is `-Inf`.
///
/// # Examples
///
/// ```
/// use russell_lab::assert_alike;
///
/// assert_alike(f64::NAN, f64::NAN);            // both NaN — OK
/// assert_alike(f64::INFINITY, f64::INFINITY);  // both +Inf — OK
/// assert_alike(f64::NEG_INFINITY, f64::NEG_INFINITY); // both -Inf — OK
/// assert_alike(1.0, 1.0);                      // equal finite values — OK
/// ```
pub fn assert_alike(a: f64, b: f64) {
    let alike = if f64::is_nan(a) && f64::is_nan(b) {
        true
    } else if a == b {
        a.is_sign_negative() == b.is_sign_negative()
    } else {
        false
    };
    if !alike {
        panic!("values are not alike");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::assert_alike;

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_nan_inf() {
        assert_alike(f64::NAN, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_nan_neg_inf() {
        assert_alike(f64::NAN, f64::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_nan_val() {
        assert_alike(f64::NAN, 2.5);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_inf_nan() {
        assert_alike(f64::INFINITY, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_inf_neg_inf() {
        assert_alike(f64::INFINITY, f64::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_inf_val() {
        assert_alike(f64::INFINITY, 2.5);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_neg_inf_nan() {
        assert_alike(f64::NEG_INFINITY, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_neg_inf_inf() {
        assert_alike(f64::NEG_INFINITY, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_neg_inf_val() {
        assert_alike(f64::NEG_INFINITY, 2.5);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_val_nan() {
        assert_alike(2.5, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_val_inf() {
        assert_alike(2.5, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "values are not alike")]
    fn assert_alike_panics_val_neg_inf() {
        assert_alike(2.5, f64::NEG_INFINITY);
    }

    #[test]
    fn assert_alike_ok_cases() {
        assert_alike(f64::NAN, f64::NAN);
        assert_alike(f64::INFINITY, f64::INFINITY);
        assert_alike(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert_alike(2.5, 2.5);
    }
}
