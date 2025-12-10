use russell_lab::Vector;

/// Calculates a vector v(r) given one scalar parameter r
///
/// The callback function is `f(v, r)`, where `v` is the output vector and `r` is the input parameter.
pub type FnVec1Param1 = Box<dyn Fn(&mut Vector, f64) + Send + Sync>;

/// Calculates a vector v(r,s) given two scalar parameter r and s
///
/// The function is `f(v, r, s)`, where `v` is the output vector and `r`, `s` are the input parameters.
pub type FnVec1Param2 = Box<dyn Fn(&mut Vector, f64, f64) + Send + Sync>;

/// Calculates two vectors u(r,s) and v(r,s) given two scalar parameters r and s
///
/// The function is `f(u, v, r, s)`, where `u` and `v` are the output vectors and `r` and `s` are the input parameters.
pub type FnVec2Param2 = Box<dyn Fn(&mut Vector, &mut Vector, f64, f64) + Send + Sync>;

/// Calculates three vectors u(r,s), v(r,s) and w(r,s) given two scalar parameters r and s
///
/// The function is `f(u, v, w, r, s)`, where `u`, `v` and `w` are the output vectors and `r` and `s` are the input parameters.
pub type FnVec3Param2 = Box<dyn Fn(&mut Vector, &mut Vector, &mut Vector, f64, f64) + Send + Sync>;
