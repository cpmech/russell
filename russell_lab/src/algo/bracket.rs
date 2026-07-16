use std::fmt;

/// Holds the results of a root finding or minimum bracketing algorithm
///
/// The root yields `f(xo) = 0.0`. The root is bracketed by a pair of points,
/// `a` and `b`, such that the function has opposite sign at those two points,
/// i.e., `f(a) × f(b) < 0`.
///
/// The (local) minimum yields `f(xo) = min{f(x)} in [a, b]`. The (local) minimum is
/// bracketed  by a triple of points `a`, `xo`, and `c`, such that `f(xo) < f(a)`
/// and `f(xo) < f(b)`, with `a < xo < b`.
#[derive(Clone, Copy, Debug)]
pub struct Bracket {
    /// Holds the lower bound
    pub a: f64,

    /// Holds the upper bound
    pub b: f64,

    /// Holds the function evaluated at the lower bound
    pub fa: f64,

    /// Holds the function evaluated at the upper bound
    pub fb: f64,

    /// Holds the r**o**ot or **o**ptimal coordinate
    pub xo: f64,

    /// Holds the function evaluated at the root or optimal coordinate
    pub fxo: f64,
}

impl fmt::Display for Bracket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lower bound:                   a = {}\n\
             root/optimum:                 xo = {}\n\
             upper bound:                   b = {}\n\
             function @ a:               f(a) = {}\n\
             function @ root/optimum:   f(xo) = {}\n\
             function @ b:               f(b) = {}",
            self.a, self.xo, self.b, self.fa, self.fxo, self.fb
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Bracket;

    #[test]
    fn bracket_display_works() {
        let bracket = Bracket {
            a: 1.0,
            xo: 2.0,
            b: 3.0,
            fa: 4.0,
            fxo: 5.0,
            fb: 6.0,
        };
        assert_eq!(
            format!("{}", bracket),
            "lower bound:                   a = 1\n\
             root/optimum:                 xo = 2\n\
             upper bound:                   b = 3\n\
             function @ a:               f(a) = 4\n\
             function @ root/optimum:   f(xo) = 5\n\
             function @ b:               f(b) = 6",
        );
    }
}
