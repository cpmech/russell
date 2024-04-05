#include <math.h>

double c_ln_gamma(double x) { return lgamma(x); }
double c_frexp(double x, int *exp) { return frexp(x, exp); }
double c_ldexp(double frac, int exp) { return ldexp(frac, exp); }
