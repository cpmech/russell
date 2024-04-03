#include <math.h>

double c_erf(double x) { return erf(x); }
double c_erfc(double x) { return erfc(x); }
double c_gamma(double x) { return tgamma(x); }
double c_frexp(double x, int *exp) { return frexp(x, exp); }
double c_ldexp(double frac, int exp) { return ldexp(frac, exp); }
