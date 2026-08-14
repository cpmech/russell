// Gauss-Jordan elimination with full pivoting.
//
// Solves the linear systems A·X = B and replaces A with its inverse A⁻¹ and
// B with the corresponding solution vectors X.
//
// A is stored in row-major order as an (n×n) matrix.
// B is stored in row-major order as an (n×m) matrix. If m = 0, B may be NULL,
// in which case only the inverse of A is computed.
//
// Reference:
//   Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P. (2007)
//   "Numerical Recipes: The Art of Scientific Computing", 3rd Edition,
//   Cambridge University Press. (Section 2.1, "Gauss-Jordan Elimination")
//
// Returns:
//   0 on success
//   1 if A is singular
//   2 if a memory allocation failed
#include <inttypes.h>
#include <stdlib.h>

int32_t num_recipes_gaussj(double *a, int32_t n, double *b, int32_t m) {
    int32_t i, icol = 0, irow = 0, j, k, l, ll;
    double big, dum, pivinv;

    // integer arrays used for bookkeeping on the pivoting
    int32_t *indxc = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *indxr = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *ipiv = (int32_t *)calloc((size_t)n, sizeof(int32_t)); // zeroed
    if (indxc == NULL || indxr == NULL || ipiv == NULL) {
        free(indxc);
        free(indxr);
        free(ipiv);
        return 2;
    }

    // main loop over the columns to be reduced
    for (i = 0; i < n; i++) {
        big = 0.0;

        // search for the pivot: the largest |element| over the remaining submatrix
        for (j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        double value = a[j * n + k];
                        if (value < 0.0) value = -value;
                        if (value >= big) {
                            big = value;
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ++(ipiv[icol]);

        // interchange rows, if needed, to put the pivot on the diagonal. The
        // columns are not physically interchanged, only relabeled: indxc[i] is
        // the column reduced at this step, while indxr[i] is the row in which
        // that pivot element was originally located. The inverse is unscrambled
        // by columns at the end.
        if (irow != icol) {
            for (l = 0; l < n; l++) {
                double tmp = a[irow * n + l];
                a[irow * n + l] = a[icol * n + l];
                a[icol * n + l] = tmp;
            }
            for (l = 0; l < m; l++) {
                double tmp = b[irow * m + l];
                b[irow * m + l] = b[icol * m + l];
                b[icol * m + l] = tmp;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (a[icol * n + icol] == 0.0) {
            free(indxc);
            free(indxr);
            free(ipiv);
            return 1;
        }

        // divide the pivot row by the pivot element
        pivinv = 1.0 / a[icol * n + icol];
        a[icol * n + icol] = 1.0;
        for (l = 0; l < n; l++) {
            a[icol * n + l] *= pivinv;
        }
        for (l = 0; l < m; l++) {
            b[icol * m + l] *= pivinv;
        }

        // reduce the rows, except for the pivot one
        for (ll = 0; ll < n; ll++) {
            if (ll != icol) {
                dum = a[ll * n + icol];
                a[ll * n + icol] = 0.0;
                for (l = 0; l < n; l++) {
                    a[ll * n + l] -= a[icol * n + l] * dum;
                }
                for (l = 0; l < m; l++) {
                    b[ll * m + l] -= b[icol * m + l] * dum;
                }
            }
        }
    }

    // unscramble the inverse in view of the column interchanges, by swapping
    // pairs of columns in the reverse order that the permutation was built up
    for (l = n - 1; l >= 0; l--) {
        if (indxr[l] != indxc[l]) {
            for (k = 0; k < n; k++) {
                double tmp = a[k * n + indxr[l]];
                a[k * n + indxr[l]] = a[k * n + indxc[l]];
                a[k * n + indxc[l]] = tmp;
            }
        }
    }

    free(indxc);
    free(indxr);
    free(ipiv);
    return 0;
}
