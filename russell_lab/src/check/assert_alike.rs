/// Asserts that two numbers are NaN at the same time or equal to each other (including ±Inf)
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
