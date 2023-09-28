#include <inttypes.h>
#include <stdlib.h>

#include "slu_ddefs.h"

#include "constants.h"

/// @brief Holds the data for SuperLU
struct InterfaceSuperLU {
    superlu_options_t options; // (input)
    // The structure defines the input parameters to control
    // how the LU decomposition will be performed and how the
    // system will be solved.

    SuperMatrix super_mat_a; // (input/output) (A->nrow, A->ncol)
    // Matrix A in A*X=B, of dimension (A->nrow, A->ncol). The number
    // of the linear equations is A->nrow. Currently, the type of A can be:
    // Stype = SLU_NC or SLU_NR, Dtype = SLU_D, Mtype = SLU_GE.
    // In the future, more general A may be handled.
    //
    // On entry, If options->Fact = FACTORED and equilibration_done is not 'N',
    // then A must have been equilibrated by the scaling factors in
    // R and/or C.
    // On exit, A is not modified if options->Equil = NO, or if
    // options->Equil = YES but equilibration_done = 'N' on exit.
    // Otherwise, if options->Equil = YES and equilibration_done is not 'N',
    // A is scaled as follows:
    // If A->Stype = SLU_NC:
    //     equilibration_done = 'R':  A := diag(R) * A
    //     equilibration_done = 'C':  A := A * diag(C)
    //     equilibration_done = 'B':  A := diag(R) * A * diag(C).
    // If A->Stype = SLU_NR:
    //     equilibration_done = 'R':  transpose(A) := diag(R) * transpose(A)
    //     equilibration_done = 'C':  transpose(A) := transpose(A) * diag(C)
    //     equilibration_done = 'B':  transpose(A) := diag(R) * transpose(A) * diag(C).

    int *permutation_col; // (input/output) (A->ncol)
    // If A->Stype = SLU_NC, Column permutation vector of size A->ncol,
    // which defines the permutation matrix Pc; perm_c[i] = j means
    // column i of A is in position j in A*Pc.
    // On exit, perm_c may be overwritten by the product of the input
    // perm_c and a permutation that post-orders the elimination tree
    // of Pc'*A'*A*Pc; perm_c is not changed if the elimination tree
    // is already in postorder.
    //
    // If A->Stype = SLU_NR, column permutation vector of size A->nrow,
    // which describes permutation of columns of transpose(A)
    // (rows of A) as described above.

    int *permutation_row; // (input/output) (size A->nrow)
    // If A->Stype = SLU_NC, row permutation vector of size A->nrow,
    // which defines the permutation matrix Pr, and is determined
    // by partial pivoting.  perm_r[i] = j means row i of A is in
    // position j in Pr*A.
    //
    // If A->Stype = SLU_NR, permutation vector of size A->ncol, which
    // determines permutation of rows of transpose(A)
    // (columns of A) as described above.
    //
    // If options->Fact = SamePattern_SameRowPerm, the pivoting routine
    // will try to use the input perm_r, unless a certain threshold
    // criterion is violated. In that case, perm_r is overwritten by a
    // new permutation determined by partial pivoting or diagonal
    // threshold pivoting.
    // Otherwise, perm_r is output argument.

    int *elimination_tree; // (input/output) (A->ncol)
    // Elimination tree of Pc'*A'*A*Pc.
    // If options->Fact != FACTORED and options->Fact != DOFACT,
    // elimination_tree is an input argument, otherwise it is an output argument.
    // Note: elimination_tree is a vector of parent pointers for a forest whose
    // vertices are the integers 0 to A->ncol-1; elimination_tree[root]==A->ncol.

    char equilibration_done[1]; // (input/output)
    // Specifies the form of equilibration that was done.
    // = 'N': No equilibration.
    // = 'R': Row equilibration, i.e., A was premultiplied by diag(R).
    // = 'C': Column equilibration, i.e., A was post-multiplied by diag(C).
    // = 'B': Both row and column equilibration, i.e., A was replaced
    //         by diag(R)*A*diag(C).
    // If options->Fact = FACTORED, equilibration_done is an input argument,
    // otherwise it is an output argument.

    double *row_scale_factors; // (input/output) (A->nrow)
    // The row scale factors for A or transpose(A).
    // If equilibration_done = 'R' or 'B', A (if A->Stype = SLU_NC) or transpose(A)
    //     (if A->Stype = SLU_NR) is multiplied on the left by diag(R).
    // If equilibration_done = 'N' or 'C', R is not accessed.
    // If options->Fact = FACTORED, R is an input argument,
    //     otherwise, R is output.
    // If options->zFact = FACTORED and equilibration_done = 'R' or 'B', each element
    //     of R must be positive.

    double *col_scale_factors; // (input/output) (A->ncol)
    // The column scale factors for A or transpose(A).
    // If equilibration_done = 'C' or 'B', A (if A->Stype = SLU_NC) or transpose(A)
    //     (if A->Stype = SLU_NR) is multiplied on the right by diag(C).
    // If equilibration_done = 'N' or 'R', C is not accessed.
    // If options->Fact = FACTORED, C is an input argument,
    //     otherwise, C is output.
    // If options->Fact = FACTORED and equilibration_done = 'C' or 'B', each element
    //     of C must be positive.

    SuperMatrix super_mat_l; // (output)
    // The factor L from the factorization
    //     Pr*A*Pc=L*U              (if A->Stype SLU_= NC) or
    //     Pr*transpose(A)*Pc=L*U   (if A->Stype = SLU_NR).
    // Uses compressed row subscripts storage for super-nodes, i.e.,
    // L has types: Stype = SLU_SC, Dtype = SLU_D, Mtype = SLU_TRLU.

    SuperMatrix super_mat_u; // (output)
    // The factor U from the factorization
    //     Pr*A*Pc=L*U              (if A->Stype = SLU_NC) or
    //     Pr*transpose(A)*Pc=L*U   (if A->Stype = SLU_NR).
    // Uses column-wise storage scheme, i.e., U has types:
    // Stype = SLU_NC, Dtype = SLU_D, Mtype = SLU_TRU.

    void *workspace; // (workspace/output), size (size_of_workspace) (in bytes)
    // User supplied workspace, should be large enough
    // to hold data structures for factors L and U.
    // On exit, if fact is not 'F', L and U point to this array.

    int size_of_workspace; // (input)
    // Specifies the size of work array in bytes.
    // = 0:  allocate space internally by system malloc;
    // > 0:  use user-supplied work array of length size_of_workspace in bytes,
    //       returns error if space runs out.
    // = -1: the routine guesses the amount of space needed without
    //       performing the factorization, and returns it in
    //       mem_usage->total_needed; no other side effects.
    //       See argument 'mem_usage' for memory usage statistics.

    SuperMatrix super_mat_b; // (input/output)
    // B has types: Stype = SLU_DN, Dtype = SLU_D, Mtype = SLU_GE.
    // On entry, the right hand side matrix.
    // If B->ncol = 0, only LU decomposition is performed, the triangular
    //                 solve is skipped.
    // On exit, if equilibration_done = 'N', B is not modified; otherwise
    // if A->Stype = SLU_NC:
    //     if options->Trans = NOTRANS and equilibration_done = 'R' or 'B',
    //         B is overwritten by diag(R)*B;
    //     if options->Trans = TRANS or CONJ and equilibration_done = 'C' of 'B',
    //         B is overwritten by diag(C)*B;
    // if A->Stype = SLU_NR:
    //     if options->Trans = NOTRANS and equilibration_done = 'C' or 'B',
    //         B is overwritten by diag(C)*B;
    //     if options->Trans = TRANS or CONJ and equilibration_done = 'R' of 'B',
    //         B is overwritten by diag(R)*B.

    SuperMatrix super_mat_x; // (output)
    // X has types: Stype = SLU_DN, Dtype = SLU_D, Mtype = SLU_GE.
    // If info = 0 or info = A->ncol+1, X contains the solution matrix
    // to the original system of equations. Note that A and B are modified
    // on exit if equilibration_done is not 'N', and the solution to the equilibrated
    // system is inv(diag(C))*X if options->Trans = NOTRANS and
    // equilibration_done = 'C' or 'B', or inv(diag(R))*X if options->Trans = 'T' or 'C'
    // and equilibration_done = 'R' or 'B'.

    double recip_pivot_growth; // (output)
    // The reciprocal pivot growth factor max_j( norm(A_j)/norm(U_j) ).
    // The infinity norm is used. If recip_pivot_growth is much less
    // than 1, the stability of the LU factorization could be poor.

    double reciprocal_condition_number; // (output)
    // The estimate of the reciprocal condition number of the matrix A
    // after equilibration (if done). If rcond is less than the machine
    // precision (in particular, if rcond = 0), the matrix is singular
    // to working precision. This condition is indicated by a return
    // code of info > 0.

    double *forward_error; // (output) (B->ncol)
    // The estimated forward error bound for each solution vector
    // X(j) (the j-th column of the solution matrix X).
    // If XTRUE is the true solution corresponding to X(j), forward_error(j)
    // is an estimated upper bound for the magnitude of the largest
    // element in (X(j) - XTRUE) divided by the magnitude of the
    // largest element in X(j).  The estimate is as reliable as
    // the estimate for RCOND, and is almost always a slight
    // overestimate of the true error.
    // If options->IterRefine = NOREFINE, forward_error = 1.0.

    double *backward_error; // (output), (B->ncol)
    // The component-wise relative backward error of each solution
    // vector X(j) (i.e., the smallest relative change in
    // any element of A or B that makes X(j) an exact solution).
    // If options->IterRefine = NOREFINE, backward_error = 1.0.

    GlobalLU_t global_lu; // (input/output)
    // If options->Fact == SamePattern_SameRowPerm, it is an input;
    //     The matrix A will be factorized assuming that a
    //     factorization of a matrix with the same sparsity pattern
    //     and similar numerical values was performed prior to this one.
    //     Therefore, this factorization will reuse both row and column
    //     scaling factors R and C, both row and column permutation
    //     vectors perm_r and perm_c, and the L & U data structures
    //     set up from the previous factorization.
    // Otherwise, it is an output.

    mem_usage_t mem_usage; // (output)
    // Record the memory usage statistics, consisting of following fields:
    // - for_lu (float)
    //   The amount of space used in bytes for L\\U data structures.
    // - total_needed (float)
    //   The amount of space needed in bytes to perform factorization.
    // - expansions (int)
    //   The number of memory expansions during the LU factorization.

    SuperLUStat_t stat; // (output)
    // Record the statistics on runtime and floating-point operation count.
    // See slu_util.h for the definition of 'SuperLUStat_t'.

    int info; // (output)
    // = 0: successful exit
    // < 0: if info = -i, the i-th argument had an illegal value
    // > 0: if info = i, and i is
    //     <= A->ncol: U(i,i) is exactly zero. The factorization has
    //         been completed, but the factor U is exactly
    //         singular, so the solution and error bounds
    //         could not be computed.
    //     = A->ncol+1: U is nonsingular, but RCOND is less than machine
    //         precision, meaning that the matrix is singular to
    //         working precision. Nevertheless, the solution and
    //         error bounds are computed because there are a number
    //         of situations where the computed solution can be more
    //         accurate than the value of RCOND would suggest.
    //     > A->ncol+1: number of bytes allocated when memory allocation
    //         failure occurred, plus A->ncol.

    /// @brief Defines the symmetry/storage of the matrix
    Mtype_t symmetry_storage;

    /// @brief Saves a double value to take a pointer used by the dummy super matrix
    double dummy;

    /// @brief Allocates a dummy structure to enable a call to factorize-only
    SuperMatrix super_mat_dummy;

    /// @brief indicates that the initialization has been completed
    int32_t initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    int32_t factorization_completed;

    /// @brief indicates that the b (rhs) and x vectors have been created
    int32_t b_and_x_vectors_created;
};

/// @brief Allocates a new SuperLU interface
struct InterfaceSuperLU *solver_superlu_new() {
    struct InterfaceSuperLU *solver = (struct InterfaceSuperLU *)malloc(sizeof(struct InterfaceSuperLU));

    if (solver == NULL) {
        return NULL;
    }

    set_default_options(&solver->options);
    solver->permutation_col = NULL;
    solver->permutation_row = NULL;
    solver->elimination_tree = NULL;
    solver->equilibration_done[0] = 'N';
    solver->row_scale_factors = NULL;
    solver->col_scale_factors = NULL;
    solver->workspace = NULL;
    solver->size_of_workspace = 0;
    solver->recip_pivot_growth = 0.0;
    solver->reciprocal_condition_number = 0.0;
    solver->forward_error = NULL;
    solver->backward_error = NULL;
    solver->symmetry_storage = SLU_GE;
    solver->dummy = 0.0;
    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;
    solver->b_and_x_vectors_created = C_FALSE;

    return solver;
}

/// @brief Deallocates the SuperLU interface
void solver_superlu_drop(struct InterfaceSuperLU *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->permutation_col != NULL) {
        free(solver->permutation_col);
    }
    if (solver->permutation_row != NULL) {
        free(solver->permutation_row);
    }
    if (solver->elimination_tree != NULL) {
        free(solver->elimination_tree);
    }
    if (solver->row_scale_factors != NULL) {
        free(solver->row_scale_factors);
    }
    if (solver->col_scale_factors != NULL) {
        free(solver->col_scale_factors);
    }
    if (solver->forward_error != NULL) {
        free(solver->forward_error);
    }
    if (solver->backward_error != NULL) {
        free(solver->backward_error);
    }
    if (solver->initialization_completed == C_TRUE) {
        StatFree(&solver->stat);
        Destroy_SuperMatrix_Store(&solver->super_mat_a);
        Destroy_SuperNode_Matrix(&solver->super_mat_l);
        Destroy_CompCol_Matrix(&solver->super_mat_u);
        Destroy_SuperMatrix_Store(&solver->super_mat_dummy);
    }
    if (solver->b_and_x_vectors_created == C_TRUE) {
        Destroy_SuperMatrix_Store(&solver->super_mat_b);
        Destroy_SuperMatrix_Store(&solver->super_mat_x);
    }

    free(solver);
}

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @note Output
/// @param condition_number Is the reciprocal condition number (if requested)
/// @note Input
/// @param ordering Is the russell ordering code to assign ColPerm
/// @param scaling Is the scaling code to assign Equil
/// @note Matrix config
/// @param ndim Is the number of rows and columns of the coefficient matrix
/// @note Matrix
/// @param col_pointers The column pointers array with size = ncol + 1
/// @param row_indices The row indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @return A success or fail code
int32_t solver_superlu_factorize(struct InterfaceSuperLU *solver,
                                 // output
                                 double *condition_number,
                                 // input
                                 int32_t ordering,
                                 C_BOOL scaling,
                                 // matrix config
                                 int32_t ndim,
                                 // matrix
                                 int32_t *col_pointers,
                                 int32_t *row_indices,
                                 double *values) {

    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->initialization_completed == C_FALSE) {
        // perform initialization

        solver->size_of_workspace = 0; // automatic
        set_default_options(&solver->options);
        solver->options.ConditionNumber = YES;
        solver->options.PrintStat = NO;

        switch (ordering) {
        case 100:
            solver->options.ColPerm = COLAMD;
            break;
        case 200:
            solver->options.ColPerm = METIS_ATA;
            break;
        case 300:
            solver->options.ColPerm = METIS_AT_PLUS_A;
            break;
        case 400:
            solver->options.ColPerm = MMD_ATA;
            break;
        case 500:
            solver->options.ColPerm = MMD_AT_PLUS_A;
            break;
        case 600:
            solver->options.ColPerm = NATURAL;
            break;
        default:
            solver->options.ColPerm = COLAMD;
            break;
        }

        if (scaling == C_TRUE) {
            solver->options.Equil = YES;
        } else {
            solver->options.Equil = NO;
        }

        int nnz = col_pointers[ndim];
        dCreate_CompCol_Matrix(&solver->super_mat_a,
                               ndim,
                               ndim,
                               nnz,
                               values,
                               row_indices,
                               col_pointers,
                               SLU_NC,
                               SLU_D,
                               SLU_GE);

        solver->permutation_col = (int *)malloc(ndim * sizeof(int));
        if (solver->permutation_col == NULL) {
            return MALLOC_ERROR;
        }
        solver->permutation_row = (int *)malloc(ndim * sizeof(int));
        if (solver->permutation_row == NULL) {
            return MALLOC_ERROR;
        }
        solver->elimination_tree = (int *)malloc(ndim * sizeof(int));
        if (solver->elimination_tree == NULL) {
            return MALLOC_ERROR;
        }
        solver->row_scale_factors = (double *)malloc(ndim * sizeof(double));
        if (solver->row_scale_factors == NULL) {
            return MALLOC_ERROR;
        }
        solver->col_scale_factors = (double *)malloc(ndim * sizeof(double));
        if (solver->col_scale_factors == NULL) {
            return MALLOC_ERROR;
        }
        solver->forward_error = (double *)malloc(sizeof(double));
        if (solver->forward_error == NULL) {
            return MALLOC_ERROR;
        }
        solver->backward_error = (double *)malloc(sizeof(double));
        if (solver->backward_error == NULL) {
            return MALLOC_ERROR;
        }

        dCreate_Dense_Matrix(&solver->super_mat_dummy, 0, 1, &solver->dummy, ndim, SLU_DN, SLU_D, SLU_GE);
        solver->super_mat_dummy.ncol = 0; // indicate not to solve the system

        StatInit(&solver->stat);

        solver->initialization_completed = C_TRUE;

    } else {
        // must reset this flag when doing multiple factorizations
        // important: assuming same structure
        solver->options.Fact = SamePattern;
    }

    // only perform the lu decomposition
    dgssvx(&solver->options,
           &solver->super_mat_a,
           solver->permutation_col,
           solver->permutation_row,
           solver->elimination_tree,
           solver->equilibration_done,
           solver->row_scale_factors,
           solver->col_scale_factors,
           &solver->super_mat_l,
           &solver->super_mat_u,
           solver->workspace,
           solver->size_of_workspace,
           &solver->super_mat_dummy,
           &solver->super_mat_dummy,
           &solver->recip_pivot_growth,
           &solver->reciprocal_condition_number,
           solver->forward_error,
           solver->backward_error,
           &solver->global_lu,
           &solver->mem_usage,
           &solver->stat,
           &solver->info);

    if (solver->info != SUCCESSFUL_EXIT) {
        return solver->info;
    }

    *condition_number = solver->reciprocal_condition_number;

    solver->options.Fact = FACTORED; // set this for the next call to solve
    solver->factorization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;
}

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param x Is the left-hand side vector (unknowns)
/// @param rhs Is the right-hand side vector
/// @return A success or fail code
int32_t solver_superlu_solve(struct InterfaceSuperLU *solver,
                             double *x,
                             double *rhs) {

    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->factorization_completed == C_FALSE) {
        return NEED_FACTORIZATION;
    }

    if (solver->b_and_x_vectors_created == C_TRUE) {
        ((DNformat *)solver->super_mat_b.Store)->nzval = rhs;
        ((DNformat *)solver->super_mat_x.Store)->nzval = x;
    } else {
        int ndim = solver->super_mat_a.nrow;
        dCreate_Dense_Matrix(&solver->super_mat_b, ndim, 1, rhs, ndim, SLU_DN, SLU_D, SLU_GE);
        dCreate_Dense_Matrix(&solver->super_mat_x, ndim, 1, x, ndim, SLU_DN, SLU_D, SLU_GE);
    }

    dgssvx(&solver->options,
           &solver->super_mat_a,
           solver->permutation_col,
           solver->permutation_row,
           solver->elimination_tree,
           solver->equilibration_done,
           solver->row_scale_factors,
           solver->col_scale_factors,
           &solver->super_mat_l,
           &solver->super_mat_u,
           solver->workspace,
           solver->size_of_workspace,
           &solver->super_mat_b,
           &solver->super_mat_x,
           &solver->recip_pivot_growth,
           &solver->reciprocal_condition_number,
           solver->forward_error,
           solver->backward_error,
           &solver->global_lu,
           &solver->mem_usage,
           &solver->stat,
           &solver->info);

    if (solver->info != SUCCESSFUL_EXIT) {
        return solver->info;
    }

    solver->b_and_x_vectors_created = C_TRUE;

    return SUCCESSFUL_EXIT;
}
