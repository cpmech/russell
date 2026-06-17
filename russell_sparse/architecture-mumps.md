# MUMPS Wrapper Architecture

## Overview

The MUMPS wrapper uses a two-layer C/Rust FFI pattern. The C layer (`interface_mumps.c`) manages the MUMPS `DMUMPS_STRUC_C` struct and the job-based analysis→factorization→solve pipeline, while the Rust layer (`solver_mumps.rs`) provides a safe interface via `LinSolTrait`, handling COO one-based index conversion, parameter mapping, error analysis, and timing.

Unlike the other solvers, MUMPS accepts raw COO format directly (one-based Fortran-style indices) and is **not thread-safe** — tests use `#[serial]`.

---

## C Layer (`c_code/interface_mumps.c`)

### State: `InterfaceMUMPS` struct

| Field                      | Type             | Purpose                                                                                                         |
| -------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| `data`                     | `DMUMPS_STRUC_C` | MUMPS data structure containing all control arrays (ICNTL, INFOG, RINFOG, INFO) and pointers (irn, jcn, a, rhs) |
| `done_job_init`            | `int32_t`        | `JOB_INITIALIZE` completed successfully                                                                         |
| `initialization_completed` | `C_BOOL`         | Analysis phase done (matrix structure established)                                                              |
| `factorization_completed`  | `C_BOOL`         | Numeric factorization done                                                                                      |

The C code uses convenience macros for 1-based Fortran-style array access:
```c
#define ICNTL(i)   icntl[(i)-1]    // integer control (input)
#define RINFOG(i)  rinfog[(i)-1]   // real info (output)
#define INFOG(i)   infog[(i)-1]    // integer info (output)
#define INFO(i)    info[(i)-1]     // analysis info (output)
```

### Lifecycle Functions

**`solver_mumps_new()`**
Allocates the C struct, initializes irn/jcn/a pointers to NULL, and sets all flags to false.

**`solver_mumps_drop()`**
1. Nullifies `irn`, `jcn`, `a` pointers to prevent MUMPS from freeing user-provided data
2. If `JOB_INITIALIZE` was done, sets verbose to false and calls `dmumps_c()` with `JOB_TERMINATE` (-2) to free internal MUMPS resources
3. Frees the struct itself

**`solver_mumps_initialize(ordering, scaling, pct_inc_workspace, max_work_memory, openmp_num_threads, verbose, general_symmetric, positive_definite, ndim, nnz, indices_i, indices_j, values_aij)`**

Called once per matrix structure. Follows the MUMPS job pipeline:

1. **Validation**: null pointer and already-initialized checks
2. **Configure matrix type** via `data.sym`:
   - `positive_definite` → `sym = 1` (SPD)
   - `general_symmetric` → `sym = 2` (general symmetric)
   - Otherwise → `sym = 0` (unsymmetric)
3. **JOB_INITIALIZE (-1)**: Calls `dmumps_c()` silently. If `INFOG(1) != 0`, returns error.
4. **Version check**: Compares `data.version_number` against `MUMPS_VERSION` from the header. Returns `ERROR_VERSION` on mismatch.
5. **Set ICNTL parameters**:
   | ICNTL | Value                | Meaning                                      |
   | ----- | -------------------- | -------------------------------------------- |
   | 5     | 0                    | Assembled matrix (not elemental)             |
   | 6     | 7                    | Auto permutation (AMD with automatic choice) |
   | 7     | `ordering`           | Fill-reducing ordering method                |
   | 8     | `scaling`            | Scaling strategy                             |
   | 14    | `pct_inc_workspace`  | Percent increase in workspace                |
   | 16    | `openmp_num_threads` | Number of OpenMP threads                     |
   | 18    | 0                    | Centralized (sequential) input               |
   | 23    | `max_work_memory`    | Maximum working memory (MB)                  |
   | 28    | 1                    | Sequential analysis                          |
   | 29    | 0                    | Ignored (no parallel environment)            |
6. **Set matrix data**: `n`, `nz`, `irn`, `jcn`, `a` (all pointers point to Rust-owned arrays; must remain valid)
7. **JOB_ANALYZE (1)**: Calls `dmumps_c()` with verbose setting. If `INFO(1) != 0`, returns error.
8. Sets `initialization_completed = C_TRUE`

**`solver_mumps_factorize(effective_ordering, effective_scaling, determinant_coefficient, determinant_exponent, compute_determinant, verbose)`**

1. Validates `initialization_completed`
2. If `compute_determinant`: sets `ICNTL(33) = 1` (request determinant) and `ICNTL(8) = 0` (disable scaling — recommended when computing determinant)
3. Calls `dmumps_c()` with `JOB_FACTORIZE (2)`
4. Reads outputs: `effective_ordering` from `INFOG(7)`, `effective_scaling` from `INFOG(33)`
5. If determinant was computed (`ICNTL(33) == 1`): reads coefficient from `RINFOG(12)` and exponent from `INFOG(34)` (base-2). Otherwise zeros both.
6. Sets `factorization_completed = C_TRUE`
7. Returns `INFOG(1)` status

**`solver_mumps_solve(rhs, error_analysis_array_len_8, error_analysis_option, verbose)`**

1. Validates `factorization_completed`
2. Sets `ICNTL(11)` = error analysis option: 0 (none), 1 (all/slow), 2 (errors only)
3. Sets `data.rhs` pointer (in-place, MUMPS overwrites RHS with solution)
4. Calls `dmumps_c()` with `JOB_SOLVE (3)`
5. If error analysis was requested, reads the 8-element array from `RINFOG(4)` through `RINFOG(11)`:
   - `[0]` norm_a, `[1]` norm_x, `[2]` resid, `[3]` omega1, `[4]` omega2
   - `[5]` delta, `[6]` cond1, `[7]` cond2 (only when option=1)
6. Returns `INFOG(1)` status

### Error Handling

MUMPS returns error codes via `INFOG(1)`. The complete set of codes handled at the Rust level:

| Codes             | Category                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| -1 through -56    | MUMPS internal errors (matrix singular, workspace too small, MPI issues, etc.)                                                                                     |
| -70 through -79   | Save/restore errors                                                                                                                                                |
| -90               | Out-of-core management error                                                                                                                                       |
| -800              | Temporary release error                                                                                                                                            |
| +1, +2, +4, +8    | Warning-level errors                                                                                                                                               |
| Generic constants | `ERROR_NULL_POINTER`, `ERROR_MALLOC`, `ERROR_VERSION`, `ERROR_NOT_AVAILABLE`, `ERROR_NEED_INITIALIZATION`, `ERROR_NEED_FACTORIZATION`, `ERROR_ALREADY_INITIALIZED` |

---

## Rust Layer (`src/solver_mumps.rs`)

### Opaque FFI Handle

```rust
#[repr(C)]
struct InterfaceMUMPS {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
```

Same pattern as other solvers. Both `InterfaceMUMPS` and `SolverMUMPS` are `unsafe impl Send`.

### Extern "C" Declarations

```rust
extern "C" {
    fn solver_mumps_new() -> *mut InterfaceMUMPS;
    fn solver_mumps_drop(solver: *mut InterfaceMUMPS);
    fn solver_mumps_initialize(solver, ordering: i32, scaling: i32,
        pct_inc_workspace: i32, max_work_memory: i32, openmp_num_threads: i32,
        verbose: CcBool, general_symmetric: CcBool, positive_definite: CcBool,
        ndim: i32, nnz: i32, indices_i: *const i32,
        indices_j: *const i32, values_aij: *const f64) -> i32;
    fn solver_mumps_factorize(solver, effective_ordering: *mut i32,
        effective_scaling: *mut i32, determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64, compute_determinant: CcBool,
        verbose: CcBool) -> i32;
    fn solver_mumps_solve(solver, rhs: *mut f64,
        error_analysis_array_len_8: *mut f64,
        error_analysis_option: i32, verbose: CcBool) -> i32;
}
```

### `SolverMUMPS` Struct

| Field                        | Type                  | Purpose                                         |
| ---------------------------- | --------------------- | ----------------------------------------------- |
| `solver`                     | `*mut InterfaceMUMPS` | C pointer                                       |
| `initialized`                | `bool`                | Analysis phase completed                        |
| `factorized`                 | `bool`                | Factorization completed                         |
| `initialized_sym`            | `Sym`                 | Saved symmetry from first call                  |
| `initialized_ndim`           | `usize`               | Saved dimension                                 |
| `initialized_nnz`            | `usize`               | Saved nnz                                       |
| `effective_ordering`         | `i32`                 | MUMPS ordering actually used                    |
| `effective_scaling`          | `i32`                 | MUMPS scaling actually used                     |
| `effective_num_threads`      | `i32`                 | OpenMP thread count passed to MUMPS (ICNTL(16)) |
| `determinant_coefficient`    | `f64`                 | `det = coefficient * 2^exponent`                |
| `determinant_exponent`       | `f64`                 | `det = coefficient * 2^exponent`                |
| `error_analysis_option`      | `i32`                 | ICNTL(11): 0 (none), 1 (all), 2 (errors)        |
| `error_analysis_array_len_8` | `Vec<f64>`            | Error analysis results from RINFOG              |
| `stopwatch`                  | `Stopwatch`           | Cumulative timer                                |
| `time_initialize_ns`         | `u128`                | Initialize time                                 |
| `time_factorize_ns`          | `u128`                | Factorize time                                  |
| `time_solve_ns`              | `u128`                | Solve time                                      |
| `fortran_indices_i`          | `Vec<i32>`            | One-based row indices (converted from COO)      |
| `fortran_indices_j`          | `Vec<i32>`            | One-based column indices (converted from COO)   |

**Note**: Unlike cuDSS and UMFPACK, MUMPS does NOT store a CSR or CSC copy. It keeps the raw COO indices (converted to one-based Fortran indexing).

### `Drop` Implementation

Calls `solver_mumps_drop(self.solver)` to free MUMPS resources and the C struct.

### `LinSolTrait` Implementation

**`factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>)`**

On first call:
1. Validates matrix (square, nnz > 0, symmetry ≤ `Sym::YesLower`; rejects `YesFull`/`YesUpper`)
2. Saves `initialized_sym`, `initialized_ndim`, `initialized_nnz`
3. Converts zero-based COO indices to one-based Fortran indices by adding 1 to each `indices_i[k]` and `indices_j[k]`
4. Maps generic `Ordering` to MUMPS ordering constants via `mumps_ordering()`
5. Maps generic `Scaling` to MUMPS scaling constants via `mumps_scaling()`
6. Determines thread count: if Intel MKL detected or user overrides, uses `par.mumps_num_threads`; otherwise defaults to 1 (avoids OpenBLAS threading bug)
7. Applies parameters: `mumps_pct_inc_workspace`, `mumps_max_work_memory`, `error_analysis_option`
8. Calls `solver_mumps_initialize()` with one-based indices and original values
9. Records `time_initialize_ns`

On subsequent calls:
1. Validates structure unchanged (sym, ndim, nnz)
2. Calls `solver_mumps_factorize()` (MUMPS receives updated values directly from `data.a`, which points to `mat.values`)
3. Records `time_factorize_ns` and output values

**`solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool)`**
1. Validates factorization is done, vectors match `initialized_ndim`
2. Copies rhs → x (MUMPS operates in-place, overwriting the RHS vector with the solution)
3. Calls `solver_mumps_solve()` with `x.as_mut_ptr()` as the RHS buffer
4. Reads error analysis results into `error_analysis_array_len_8`
5. Records `time_solve_ns`

**`update_stats(&self, stats: &mut StatsLinSol)`**
Populates:
- Solver name: `"MUMPS"`
- Determinant: coefficient, exponent, base=2.0
- Effective ordering label (Amd, Amf, Auto, Metis, Pord, Qamd, Scotch, or Unknown)
- Effective scaling label (Auto, Column, Diagonal, No, RowCol, RowColIter, RowColRig, scaling-during-analysis, or Unknown)
- Effective thread count
- Error analysis: norm_a, norm_x, scaled_residual, omega1, omega2, delta_x, cond1, cond2
- Timing fields

### Ordering and Scaling Constants

**Ordering Methods** (ICNTL(7)):
| Constant                | Value | Description                |
| ----------------------- | ----- | -------------------------- |
| `MUMPS_ORDERING_AMD`    | 0     | Approximate Minimum Degree |
| `MUMPS_ORDERING_AMF`    | 2     | Approximate Minimum Fill   |
| `MUMPS_ORDERING_AUTO`   | 7     | Automatic choice           |
| `MUMPS_ORDERING_METIS`  | 5     | METIS                      |
| `MUMPS_ORDERING_PORD`   | 4     | PORD                       |
| `MUMPS_ORDERING_QAMD`   | 6     | QAMD                       |
| `MUMPS_ORDERING_SCOTCH` | 3     | SCOTCH                     |

**Scaling Methods** (ICNTL(8)):
| Constant                     | Value | Description                  |
| ---------------------------- | ----- | ---------------------------- |
| `MUMPS_SCALING_AUTO`         | 77    | Automatic choice             |
| `MUMPS_SCALING_COLUMN`       | 3     | Column scaling               |
| `MUMPS_SCALING_DIAGONAL`     | 1     | Diagonal scaling             |
| `MUMPS_SCALING_NO`           | 0     | No scaling                   |
| `MUMPS_SCALING_ROW_COL`      | 4     | Row-column scaling           |
| `MUMPS_SCALING_ROW_COL_ITER` | 7     | Iterative row-column scaling |
| `MUMPS_SCALING_ROW_COL_RIG`  | 8     | Rigorous row-column scaling  |

Unsupported generic `Ordering`/`Scaling` values fall back to `MUMPS_ORDERING_AUTO`/`MUMPS_SCALING_AUTO`.

### Error Mapping

`handle_mumps_error_code(err: i32) -> StrError` maps 78 MUMPS-specific error codes to descriptive Rust strings, plus the 7 shared generic constants, with a catch-all for unknowns.

### Thread Safety and OpenMP

MUMPS is **not thread-safe**. The solver must not be used concurrently from multiple threads. Tests use `#[serial]` to enforce sequential execution.

The number of OpenMP threads is passed via `ICNTL(16)`. The Rust code defaults to 1 thread to avoid a known issue with OpenBLAS, unless:
- Intel MKL is detected (via `using_intel_mkl()`)
- User explicitly sets `par.mumps_num_threads != 0`
- User enables `par.mumps_override_prevent_nt_issue_with_openblas`

---

## Key Design Points

1. **COO format with one-based indices** — Unlike cuDSS (CSR) and UMFPACK (CSC), MUMPS accepts raw COO data. The Rust layer converts zero-based indices to Fortran-style one-based by adding 1.

2. **Job-based pipeline** — MUMPS uses a stateful job model: `-1` init → `1` analyze → `2` factorize → `3` solve → `-2` terminate. The C layer orchestrates these, setting ICNTL parameters before each phase and reading INFOG/RINFOG after.

3. **In-place solve** — MUMPS overwrites the RHS vector with the solution. The Rust layer copies `rhs` into `x` before the solve call, and passes `x` as the RHS buffer.

4. **Version check at init** — The C code compares the runtime MUMPS library version against the compile-time header version. Mismatch returns `ERROR_VERSION` before any analysis occurs.

5. **Pointer aliasing** — `irn`, `jcn`, `a` in the MUMPS struct point directly to Rust-owned `Vec<i32>`/`Vec<f64>` arrays. The C code must not free them; on drop, these pointers are nullified before calling `JOB_TERMINATE`.

6. **Determinant (base-2)** — Unlike UMFPACK (base-10), MUMPS reports the determinant as `coefficient * 2^exponent`. Scaling is automatically disabled when computing the determinant (`ICNTL(8) = 0`).

7. **Error analysis** — MUMPS can optionally compute condition numbers and backward error estimates during solve. Results populate an 8-element array from `RINFOG(4-11)`. Full analysis (option=1) is expensive.

8. **Structure-once, factorize-many** — Analysis (JOB_ANALYZE) runs only on the first `factorize()` call. Subsequent calls with different values run only JOB_FACTORIZE, reusing the established structure and symbolic factorization. MUMPS reads updated values directly from the user-provided `a` array.

9. **Scaling disabled for determinant** — When `compute_determinant` is true, the C layer sets `ICNTL(8) = 0` (no scaling), per MUMPS recommendation. This means scaling output may show "No" even if a scaling method was requested.
