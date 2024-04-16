#![allow(unused)]

use super::{Params, Stats};
use crate::StrError;

/// Integrates a function f(x) using numerical quadrature
///
/// Approximates:
///
/// ```text
///        ub
///       ⌠
/// I  =  │  f(x) dx
///       ⌡
///     lb
/// ```
///
/// # Input
///
/// * `lb` -- the lower bound
/// * `ub` -- the upper bound
/// * `params` -- optional control parameters
/// * `args` -- extra arguments for the callback function
/// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
///
/// # Output
///
/// Returns `(ii, stats)` where:
///
/// * `ii` -- the result `I` of the integration: `I = ∫_lb^ub f(x) dx`
/// * `stats` -- some statistics about the computations, including the estimated error
pub fn quadrature<F, A>(
    lb: f64,
    ub: f64,
    params: Option<Params>,
    args: &mut A,
    mut f: F,
) -> Result<(f64, Stats), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    Err("TODO")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::quadrature;
    use crate::algo::NoArgs;

    #[test]
    #[should_panic]
    fn quadrature_works_1() {
        let args = &mut 0;
        let _ii = quadrature(0.0, 1.0, None, args, |_, _| Ok(0.0)).unwrap();
    }
}
