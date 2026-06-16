# KLU Wrapper Architecture

## Overview

The KLU wrapper uses a two-layer C/Rust FFI pattern. The C layer (`interface_klu.c`) manages KLU's `klu_common`, `klu_symbolic*`, and `klu_numeric*` handles, while the Rust layer (`solver_klu.rs`) provides a safe interface via `LinSolTrait`, handling COO→CSC conversion, validation, ordering/scaling mapping, and timing.

KLU (Clark Kent LU) is a sparse LU factorization solver optimized for circuit simulation matrices. It uses CSC format and operates entirely on the CPU. It supports only two ordering methods (AMD and COLAMD) and three scaling methods (none, sum, max).

---

## C Layer (`c_code/interface_klu.c`)

### State: `InterfaceKLU` struct

| Field                      | Type            | Purpose                                                            |
| -------------------------- | --------------- | ------------------------------------------------------------------ |
| `common`                   | `klu_common`    | KLU control parameters (ordering, scaling, condest) and statistics |
| `symbolic`                 | `klu_symbolic*` | Handle to symbolic factorization (pre-ordering)                    |
| `numeric`                  | `klu_numeric*`  | Handle to numeric factorization (LU factors)                       |
| `initialization_completed` | `C_BOOL`        | Symbolic analysis done                                             |
| `factorization_completed`  | `C_BOOL`        | Numeric factorization done                                         |

### Lifecycle Functions

**`solver_klu_new()`**
Allocates the C struct, initializes symbolic/numeric handles to NULL, and sets status flags to false. Returns NULL on malloc failure.

**`solver_klu_drop()`**
Null-safe teardown: frees symbolic handle via `klu_free_symbolic`, frees numeric handle via `klu_free_numeric`, then frees the struct itself. The symbolic/numeric pointers are also freed with `free()` after the KLU free calls (matching the pattern used by UMFPACK).

**`solver_klu_initialize(ordering, scaling, ndim, col_pointers, row_indices)`**

Called once per matrix structure — no values needed for symbolic phase:
1. Validates `solver != NULL` and `initialization_completed == C_FALSE`
2. Calls `klu_defaults()` to initialize common struct with KLU defaults
3. Sets `common.ordering` if `ordering >= 0` (a value of -10 = AUTO means "use default")
4. Sets `common.scale` if `scaling >= 0` (a value of -10 = AUTO means "use default")
5. Calls `klu_analyze()` — symbolic factorization using only structure (col_pointers, row_indices). Casts away const (assumes KLU does not modify).
6. If symbolic fails (returns NULL), returns `KLU_ERROR_ANALYZE` (-9)
7. Sets `initialization_completed = C_TRUE`

Note: Unlike UMFPACK and MUMPS, KLU's analyze phase does not need matrix values — only the sparsity pattern.

**`solver_klu_factorize(effective_ordering, effective_scaling, cond_estimate, compute_cond, col_pointers, row_indices, values)`**

Can be called multiple times with different values:
1. Validates `initialization_completed`
2. Frees previous numeric handle (prevents memory leak on repeated factorizations)
3. Calls `klu_factor()` — numeric factorization. Casts away const.
4. If factor fails (returns NULL), returns `KLU_ERROR_FACTOR` (-8)
5. Reads effective ordering and scaling from `common.ordering` / `common.scale`
6. If `compute_cond`: calls `klu_condest()` for condition number estimate. On failure, returns `KLU_ERROR_COND_EST` (-7). On success, reads `common.condest`.
7. Sets `factorization_completed = C_TRUE`

**`solver_klu_solve(ndim, in_rhs_out_x)`**

Simple, no verbose option:
1. Validates `factorization_completed`
2. Calls `klu_solve()` — operates in-place, overwriting the RHS buffer with the solution (`nrhs=1`)
3. Returns `SUCCESSFUL_EXIT` (KLU solve does not return an error code)

### Error Handling

KLU-specific error codes (defined in `constants.h`):

| Code | Name                 | Meaning                      |
| ---- | -------------------- | ---------------------------- |
| -9   | `KLU_ERROR_ANALYZE`  | `klu_analyze` returned NULL  |
| -8   | `KLU_ERROR_FACTOR`   | `klu_factor` returned NULL   |
| -7   | `KLU_ERROR_COND_EST` | `klu_condest` returned false |

Plus shared generic constants: `ERROR_NULL_POINTER`, `ERROR_MALLOC`, `ERROR_VERSION`, `ERROR_NOT_AVAILABLE`, `ERROR_NEED_INITIALIZATION`, `ERROR_NEED_FACTORIZATION`, `ERROR_ALREADY_INITIALIZED`.

---

## Rust Layer (`src/solver_klu.rs`)

### Opaque FFI Handle

```rust
#[repr(C)]
struct InterfaceKLU {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
```

Same pattern as other solvers. Both `InterfaceKLU` and `SolverKLU` are `unsafe impl Send`.

### Extern "C" Declarations

```rust
extern "C" {
    fn solver_klu_new() -> *mut InterfaceKLU;
    fn solver_klu_drop(solver: *mut InterfaceKLU);
    fn solver_klu_initialize(solver, ordering: i32, scaling: i32,
        ndim: i32, col_pointers: *const i32, row_indices: *const i32) -> i32;
    fn solver_klu_factorize(solver, effective_ordering: *mut i32,
        effective_scaling: *mut i32, cond_estimate: *mut f64,
        compute_cond: CcBool, col_pointers: *const i32,
        row_indices: *const i32, values: *const f64) -> i32;
    fn solver_klu_solve(solver, ndim: i32, in_rhs_out_x: *mut f64) -> i32;
}
```

Note: `initialize` does not take values — KLU's symbolic phase only needs sparsity pattern.

### `SolverKLU` Struct

| Field                | Type                | Purpose                                |
| -------------------- | ------------------- | -------------------------------------- |
| `solver`             | `*mut InterfaceKLU` | C pointer                              |
| `csc`                | `Option<CscMatrix>` | CSC copy for subsequent factorizations |
| `initialized`        | `bool`              | Analysis completed                     |
| `factorized`         | `bool`              | Factorization completed                |
| `initialized_sym`    | `Sym`               | Saved symmetry from first call         |
| `initialized_ndim`   | `usize`             | Saved dimension                        |
| `initialized_nnz`    | `usize`             | Saved nnz                              |
| `effective_ordering` | `i32`               | KLU ordering actually used             |
| `effective_scaling`  | `i32`               | KLU scaling actually used              |
| `cond_estimate`      | `f64`               | 1-norm condition number estimate       |
| `stopwatch`          | `Stopwatch`         | Cumulative timer                       |
| `time_initialize_ns` | `u128`              | Initialize time                        |
| `time_factorize_ns`  | `u128`              | Factorize time                         |
| `time_solve_ns`      | `u128`              | Solve time                             |

### `Drop` Implementation

Calls `solver_klu_drop(self.solver)` to free all C-side resources.

### `LinSolTrait` Implementation

**`factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>)`**

On first call:
1. Validates matrix (square, nnz > 0, symmetry must be `No` or `YesFull`; rejects `YesLower`/`YesUpper`)
2. Converts COO → CSC, stores in `self.csc`
3. Saves `initialized_sym`, `initialized_ndim`, `initialized_nnz`
4. Applies parameters: `ordering`, `scaling` (mapped via `klu_ordering()` / `klu_scaling()`), `compute_condition_numbers`
5. Calls `solver_klu_initialize()` (no values passed — only structure needed)
6. Records `time_initialize_ns`

On subsequent calls:
1. Validates structure unchanged (sym, ndim, nnz same as first call)
2. Updates CSC values from new COO matrix
3. Calls `solver_klu_factorize()`
4. Records `time_factorize_ns` and output values (ordering, scaling, cond_estimate)

**`solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool)`**
1. Validates factorization is done
2. Checks vector dimensions match `initialized_ndim`
3. Copies rhs → x (KLU operates in-place)
4. Calls `solver_klu_solve(ndim, x.as_mut_ptr())`
5. Records `time_solve_ns`

Note: The `verbose` parameter is ignored — KLU has no verbose output mechanism.

**`update_stats(&self, stats: &mut StatsLinSol)`**
Populates:
- Solver name: `"KLU"` or `"KLU-local"` (feature-gated)
- Condition estimate in `umfpack_rcond_estimate` field (reused for KLU)
- Effective ordering label (Amd, Colamd, or Unknown)
- Effective scaling label (No, Sum, Max, or Unknown)
- Timing fields

### Ordering and Scaling Constants

**Ordering Methods** (KLU supports only two):
| Constant              | Value | Description                              |
| --------------------- | ----- | ---------------------------------------- |
| `KLU_ORDERING_AUTO`   | -10   | Use KLU defaults (code-defined sentinel) |
| `KLU_ORDERING_AMD`    | 0     | Approximate Minimum Degree               |
| `KLU_ORDERING_COLAMD` | 1     | Column Approximate Minimum Degree        |

**Scaling Methods** (KLU supports three):
| Constant         | Value | Description                              |
| ---------------- | ----- | ---------------------------------------- |
| `KLU_SCALE_AUTO` | -10   | Use KLU defaults (code-defined sentinel) |
| `KLU_SCALE_NONE` | 0     | No scaling                               |
| `KLU_SCALE_SUM`  | 1     | Divide by sum(abs(row))                  |
| `KLU_SCALE_MAX`  | 2     | Divide by max(abs(row))                  |

The AUTO sentinel (-10) is used because the C layer checks `>= 0` to decide whether to override the KLU defaults. A negative value means "keep the defaults set by `klu_defaults()`".

**Mapping**: All unsupported generic ordering/scaling values fall back to `KLU_ORDERING_AUTO`/`KLU_SCALE_AUTO`.

### Error Mapping

`handle_klu_error_code(err: i32) -> StrError` maps:
- 3 KLU-specific codes: -9 ("klu_analyze failed"), -8 ("klu_factor failed"), -7 ("klu_condest failed")
- 7 shared generic constants from `constants.h`
- A catch-all fallback

---

## Key Design Points

1. **CSC format** — Like UMFPACK, KLU uses Compressed Sparse Column. The Rust layer converts COO → CSC.

2. **Symbolic analysis without values** — KLU's `klu_analyze()` only needs the sparsity pattern (col_pointers, row_indices), not matrix values. This differs from UMFPACK/MUMPS which pass values to their symbolic/analysis phases.

3. **Simple solve** — `klu_solve()` returns void and operates in-place. No error code, no verbose flag. The Rust layer copies rhs → x before the call.

4. **Structure-once, factorize-many** — Symbolic analysis happens only on the first `factorize()` call. Subsequent calls reuse the `symbolic` handle and only re-run numeric factorization with updated values. Previous `numeric` handle is freed before each new factorization.

5. **Limited ordering options** — KLU natively supports only AMD (0) and COLAMD (1). The wrapper uses a sentinel value of -10 to mean "use defaults" when the generic ordering enum maps to an unsupported method.

6. **Condition number** — Optionally computed during factorization via `klu_condest()`. The result is stored in `common.condest` and reported as `umfpack_rcond_estimate` in stats.

7. **No determinant** — Unlike UMFPACK and MUMPS, KLU does not provide determinant computation.

8. **Const casts** — The C layer casts away `const` from col_pointers/row_indices/values when passing to KLU functions. This assumes the KLU library does not modify these input arrays.

9. **Symmetric matrix handling** — Like UMFPACK, KLU requires `Sym::YesFull` for symmetric matrices (the full matrix, not just one triangle).
