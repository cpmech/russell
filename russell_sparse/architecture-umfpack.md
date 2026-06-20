# UMFPACK Wrapper Architecture

## Overview

The UMFPACK wrapper uses a two-layer C/Rust FFI pattern. The C layer (`interface_umfpack.c`) manages UMFPACK's symbolic/numeric handles and control/info arrays, while the Rust layer (`solver_umfpack.rs`) provides a safe, idiomatic interface via `LinSolTrait`, handling COO→CSC conversion, validation, ordering/scaling mapping, and timing.

Unlike the cuDSS wrapper, UMFPACK operates entirely on the CPU and uses CSC (Compressed Sparse Column) format.

---

## C Layer (`c_code/interface_umfpack.c`)

### State: `InterfaceUMFPACK` struct

| Field                      | Type                      | Purpose                                                               |
| -------------------------- | ------------------------- | --------------------------------------------------------------------- |
| `control`                  | `double[UMFPACK_CONTROL]` | UMFPACK control parameters (ordering, scaling, strategy, print level) |
| `info`                     | `double[UMFPACK_INFO]`    | UMFPACK output info (strategy used, ordering used, rcond, etc.)       |
| `symbolic`                 | `void*`                   | Handle to symbolic factorization results                              |
| `numeric`                  | `void*`                   | Handle to numeric factorization results                               |
| `initialization_completed` | `C_BOOL`                  | Symbolic factorization done                                           |
| `factorization_completed`  | `C_BOOL`                  | Numeric factorization done                                            |

### Lifecycle Functions

**`solver_umfpack_new()`**
Allocates the C struct, initializes control array with `umfpack_di_defaults`, and sets symbolic/numeric handles to NULL. Returns NULL on malloc failure.

**`solver_umfpack_drop()`**
Null-safe teardown: frees symbolic handle via `umfpack_di_free_symbolic`, frees numeric handle via `umfpack_di_free_numeric`, then frees the struct itself. Note that the symbolic/numeric pointers themselves are also freed with `free()` after the UMFPACK free calls.

**`solver_umfpack_initialize(ordering, scaling, verbose, enforce_unsymmetric_strategy, ndim, col_pointers, row_indices, values)`**
Called once per matrix structure:
1. Validates `solver != NULL` and `initialization_completed == C_FALSE`
2. Configures strategy: `UMFPACK_STRATEGY_AUTO` by default, or `UMFPACK_STRATEGY_UNSYMMETRIC` if enforced
3. Sets ordering and scaling from parameters
4. Sets verbose mode
5. Calls `umfpack_di_symbolic()` — symbolic factorization
6. If symbolic fails, returns the UMFPACK error code directly
7. Sets `initialization_completed = C_TRUE`

**`solver_umfpack_factorize(effective_strategy, effective_ordering, effective_scaling, rcond_estimate, determinant_coefficient, determinant_exponent, compute_determinant, verbose, col_pointers, row_indices, values)`**
Can be called multiple times with different values (same structure):
1. Validates `initialization_completed`
2. Frees previous numeric handle (to avoid memory leak on repeated factorizations)
3. Calls `umfpack_di_numeric()` — numeric factorization
4. Optionally calls `umfpack_di_report_info()` for verbose output
5. Extracts output via `info` array: strategy used, ordering used, rcond estimate
6. Copies effective scaling from `control[UMFPACK_SCALE]`
7. If requested, calls `umfpack_di_get_determinant()`; otherwise zeros the output
8. Sets `factorization_completed = C_TRUE`
9. Returns the UMFPACK numeric or determinant error code

**`solver_umfpack_solve(x, rhs, col_pointers, row_indices, values, verbose)`**
1. Validates `factorization_completed`
2. Sets verbose mode
3. Calls `umfpack_di_solve(UMFPACK_A, ...)` — solve `Ax = b` (not transpose)
4. Optionally calls `umfpack_di_report_info()`
5. Returns the UMFPACK solve error code

### Error Handling

Returns `int32_t` status codes. UMFPACK-specific codes:

| Code             | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| 0 (`UMFPACK_OK`) | Success                                         |
| 1                | Matrix is singular                              |
| 2                | Determinant is nonzero but smaller than allowed |
| 3                | Determinant is larger than allowed              |
| -1               | Not enough memory                               |
| -3               | Invalid numeric object                          |
| -4               | Invalid symbolic object                         |
| -5               | Argument missing                                |
| -6               | nrow or ncol must be > 0                        |
| -8               | Invalid matrix                                  |
| -11              | Different pattern                               |
| -13              | Invalid system                                  |
| -15              | Invalid permutation                             |
| -17              | Failed to save/load file                        |
| -18              | Ordering method failed                          |
| -911             | Internal error                                  |

Plus the shared constants: `ERROR_NULL_POINTER`, `ERROR_MALLOC`, `ERROR_VERSION`, `ERROR_NOT_AVAILABLE`, `ERROR_NEED_INITIALIZATION`, `ERROR_NEED_FACTORIZATION`, `ERROR_ALREADY_INITIALIZED`.

---

## Rust Layer (`src/solver_umfpack.rs`)

### Opaque FFI Handle

```rust
#[repr(C)]
struct InterfaceUMFPACK {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
```

Same zero-sized type + PhantomData pattern as cuDSS. Both `InterfaceUMFPACK` and `SolverUMFPACK` are `unsafe impl Send`.

### Extern "C" Declarations

```rust
extern "C" {
    fn solver_umfpack_new() -> *mut InterfaceUMFPACK;
    fn solver_umfpack_drop(solver: *mut InterfaceUMFPACK);
    fn solver_umfpack_initialize(solver, ordering: i32, scaling: i32, verbose: CcBool,
        enforce_unsymmetric_strategy: CcBool, ndim: i32,
        col_pointers: *const i32, row_indices: *const i32, values: *const f64) -> i32;
    fn solver_umfpack_factorize(solver, effective_strategy: *mut i32,
        effective_ordering: *mut i32, effective_scaling: *mut i32,
        rcond_estimate: *mut f64, determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64, compute_determinant: CcBool,
        verbose: CcBool, col_pointers: *const i32,
        row_indices: *const i32, values: *const f64) -> i32;
    fn solver_umfpack_solve(solver, x: *mut f64, rhs: *const f64,
        col_pointers: *const i32, row_indices: *const i32,
        values: *const f64, verbose: CcBool) -> i32;
}
```

### `SolverUMFPACK` Struct

| Field                     | Type                    | Purpose                                |
| ------------------------- | ----------------------- | -------------------------------------- |
| `solver`                  | `*mut InterfaceUMFPACK` | C pointer                              |
| `csc`                     | `Option<CscMatrix>`     | CSC copy for subsequent factorizations |
| `initialized`             | `bool`                  | Mirrors C `initialization_completed`   |
| `factorized`              | `bool`                  | Mirrors C `factorization_completed`    |
| `initialized_sym`         | `Sym`                   | Saved symmetry type from first call    |
| `initialized_ndim`        | `usize`                 | Saved dimension from first call        |
| `initialized_nnz`         | `usize`                 | Saved nnz from first call              |
| `effective_strategy`      | `i32`                   | UMFPACK strategy actually used         |
| `effective_ordering`      | `i32`                   | UMFPACK ordering actually used         |
| `effective_scaling`       | `i32`                   | UMFPACK scaling actually used          |
| `rcond_estimate`          | `f64`                   | Reciprocal condition number estimate   |
| `determinant_coefficient` | `f64`                   | `det = coefficient * 10^exponent`      |
| `determinant_exponent`    | `f64`                   | `det = coefficient * 10^exponent`      |
| `stopwatch`               | `Stopwatch`             | Cumulative timer                       |
| `time_initialize_ns`      | `u128`                  | Initialize time in nanoseconds         |
| `time_factorize_ns`       | `u128`                  | Factorize time in nanoseconds          |
| `time_solve_ns`           | `u128`                  | Solve time in nanoseconds              |

### `Drop` Implementation

Calls `solver_umfpack_drop(self.solver)` to free all C-side resources.

### `LinSolTrait` Implementation

**`factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>)`**

On first call:
1. Validates matrix (square, nnz > 0, symmetry must be `No` or `YesFull`; rejects `YesLower`/`YesUpper`)
2. Converts COO → CSC, stores in `self.csc`
3. Saves `initialized_sym`, `initialized_ndim`, `initialized_nnz`
4. Applies parameters: `ordering`, `scaling`, `verbose`, `compute_determinant`, `umfpack_enforce_unsymmetric_strategy`
5. Maps generic `Ordering` enum to UMFPACK ordering constants via `umfpack_ordering()`
6. Maps generic `Scaling` enum to UMFPACK scaling constants via `umfpack_scaling()`
7. Calls `solver_umfpack_initialize()`
8. Records `time_initialize_ns`

On subsequent calls:
1. Validates structure unchanged (sym, ndim, nnz same as first call)
2. Updates CSC values from new COO matrix
3. Calls `solver_umfpack_factorize()`
4. Records `time_factorize_ns` and output values (strategy, ordering, scaling, rcond, determinant)

**`solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool)`**
1. Validates factorization is done
2. Checks vector dimensions match `initialized_ndim`
3. Calls `solver_umfpack_solve()`
4. Records `time_solve_ns`

**`update_stats(&self, stats: &mut StatsLinSol)`**
Populates solver name (`"UMFPACK"` or `"UMFPACK-local"` based on feature flag), determinant (coefficient, exponent, base=10), rcond estimate, effective ordering, scaling, and strategy labels, plus timing fields.

**`get_ns_init/fact/solve()`**
Return individual timing measurements.

### Strategy, Ordering, and Scaling Constants

**Strategies** (effectively chosen by UMFPACK based on enforce_unsymmetric):
| Constant                       | Value | Description                          |
| ------------------------------ | ----- | ------------------------------------ |
| `UMFPACK_STRATEGY_AUTO`        | 0     | Auto-detect symmetric or unsymmetric |
| `UMFPACK_STRATEGY_UNSYMMETRIC` | 1     | COLAMD, no diagonal preference       |
| `UMFPACK_STRATEGY_SYMMETRIC`   | 3     | AMD(A+A'), prefer diagonal           |

**Ordering Methods** (mapped from generic `Ordering` enum):
| Constant                   | Value | Description                     |
| -------------------------- | ----- | ------------------------------- |
| `UMFPACK_ORDERING_CHOLMOD` | 0     | CHOLMOD (AMD/COLAMD then METIS) |
| `UMFPACK_ORDERING_AMD`     | 1     | AMD/COLAMD (default)            |
| `UMFPACK_ORDERING_METIS`   | 3     | METIS                           |
| `UMFPACK_ORDERING_BEST`    | 4     | Try many, pick best             |
| `UMFPACK_ORDERING_NONE`    | 5     | Natural ordering                |

**Scaling Methods** (mapped from generic `Scaling` enum):
| Constant             | Value | Description                       |
| -------------------- | ----- | --------------------------------- |
| `UMFPACK_SCALE_NONE` | 0     | No scaling                        |
| `UMFPACK_SCALE_SUM`  | 1     | Divide by sum(abs(row)) (default) |
| `UMFPACK_SCALE_MAX`  | 2     | Divide by max(abs(row))           |

### Ordering/Scaling Mapping

The functions `umfpack_ordering()` and `umfpack_scaling()` translate the generic `Ordering`/`Scaling` enums to UMFPACK-specific integer constants. Unsupported generic values fall back to `UMFPACK_DEFAULT_ORDERING`/`UMFPACK_DEFAULT_SCALE`.

### Error Mapping

`handle_umfpack_error_code(err: i32) -> StrError` maps each UMFPACK status code to a descriptive Rust error string, including:
- 15 UMFPACK-specific codes (1, 2, 3, -1, -3 through -18, -911)
- The 7 shared constants from `constants.h` (`ERROR_NULL_POINTER`, etc.)
- A catch-all fallback for unknown codes

---

## Key Design Points

1. **CSC format** — Unlike cuDSS (which uses CSR), UMFPACK uses Compressed Sparse Column. The Rust layer converts COO → CSC, not CSR.

2. **Symmetric matrix handling** — UMFPACK requires `Sym::YesFull` (the full matrix, not just one triangle). If a symmetric matrix is provided with only the lower or upper triangle, Rust rejects it.

3. **Structure-once, factorize-many** — Symbolic factorization happens only on the first `factorize()` call. Subsequent calls reuse the same symbolic handle and only re-run numeric factorization with updated values.

4. **No GPU** — All operations are CPU-side. No CUDA streams, device memory, or GPU synchronization.

5. **Control/Info arrays** — UMFPACK uses double-typed control (input) and info (output) arrays. The C layer configures control before each call and reads info after. The Rust layer receives extracted outputs via `factorize` out-parameters.

6. **Numeric handle lifecycle** — On repeated factorizations, the previous `numeric` handle is freed before creating a new one, preventing memory leaks. The symbolic handle persists across factorizations.

7. **Determinant** — Optionally computed during factorization. Result stored as `coefficient * 10^exponent` (scientific notation). Available in stats output.

8. **Reciprocal condition number** — `rcond_estimate` ≈ 1/(condition number), automatically computed during numeric factorization.
