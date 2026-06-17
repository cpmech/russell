# cuDSS Wrapper Architecture

## Overview

The cuDSS wrapper uses a two-layer C/Rust FFI pattern. The C layer (`interface_cudss.cu`) manages all CUDA/cuDSS resources and GPU memory directly, while the Rust layer (`solver_cudss.rs`) provides a safe, idiomatic interface via `LinSolTrait`, handling COO→CSR conversion, validation, and timing.

---

## C Layer (`c_code/interface_cudss.cu`)

### State: `InterfaceCUDSS` struct

Holds all CUDA and cuDSS resources:

| Field                      | Type            | Purpose                                     |
| -------------------------- | --------------- | ------------------------------------------- |
| `stream`                   | `cudaStream_t`  | CUDA stream for async operations            |
| `handle`                   | `cudssHandle_t` | cuDSS library handle                        |
| `config`                   | `cudssConfig_t` | Solver configuration (pivot strategy, etc.) |
| `data`                     | `cudssData_t`   | Solver workspace data                       |
| `aa_mat`                   | `cudssMatrix_t` | Sparse coefficient matrix (CSR format)      |
| `b_vec`                    | `cudssMatrix_t` | Dense right-hand side vector                |
| `x_vec`                    | `cudssMatrix_t` | Dense solution vector                       |
| `gpu_row_pointers`         | `int*`          | GPU row pointers array                      |
| `gpu_col_indices`          | `int*`          | GPU column indices array                    |
| `gpu_values`               | `double*`       | GPU matrix values array                     |
| `gpu_b`                    | `double*`       | GPU RHS array                               |
| `gpu_x`                    | `double*`       | GPU solution array                          |
| `ndim`                     | `int`           | System dimension                            |
| `nnz`                      | `int`           | Number of non-zeros                         |
| `initialization_completed` | `C_BOOL`        | Symbolic factorization done                 |
| `factorization_completed`  | `C_BOOL`        | Numeric factorization done                  |

### Lifecycle Functions

**`solver_cudss_new()`**
Creates CUDA stream → cuDSS handle → sets stream on handle → creates config → creates data → allocates C struct. Properly tears down partially-allocated resources on any failure. Returns NULL on error.

**`solver_cudss_drop()`**
Null-safe teardown, destroying resources in reverse creation order: matrices → data → config → handle → stream, then frees all GPU arrays and the struct itself.

**`solver_cudss_initialize(verbose, general_symmetric, positive_definite, ndim, row_pointers, col_indices, values)`**
Called once per matrix structure:
1. Stores `ndim` and `nnz` (= `row_pointers[ndim]`)
2. Allocates GPU memory for row pointers, col indices, values, b, x
3. Copies host→device: row pointers, col indices, initial values
4. Creates dense cuDSS matrices for b and x (`CUDSS_R_64F`, column-major)
5. Sets pivot strategy (`CUDSS_PIVOT_AUTO`)
6. Creates sparse CSR cuDSS matrix for A:
   - `positive_definite` → `CUDSS_MTYPE_SPD` (lower triangle)
   - `general_symmetric` → `CUDSS_MTYPE_SYMMETRIC` (lower triangle)
   - Otherwise → `CUDSS_MTYPE_GENERAL` (full matrix)
7. Runs symbolic factorization (`CUDSS_PHASE_ANALYSIS`)
8. Synchronizes stream
9. Sets `initialization_completed = C_TRUE`

**`solver_cudss_factorize(verbose, values)`**
Can be called multiple times with different values (same structure):
1. Validates `initialization_completed`
2. Copies updated values host→device
3. Notifies cuDSS via `cudssMatrixSetValues`
4. Runs numeric factorization (`CUDSS_PHASE_FACTORIZATION`)
5. Synchronizes stream
6. Sets `factorization_completed = C_TRUE`

**`solver_cudss_solve(x, rhs, verbose)`**
1. Validates `factorization_completed`
2. Copies RHS host→device
3. Runs solve (`CUDSS_PHASE_SOLVE`)
4. Synchronizes stream
5. Copies solution device→host into `x`

### Error Handling

Returns `int32_t` status codes (defined in `constants.h`):

| Code | Name                            | Description                          |
| ---- | ------------------------------- | ------------------------------------ |
| 0    | `SUCCESSFUL_EXIT`               | Operation completed successfully     |
| 100  | `ERROR_CUDA_MALLOC`             | GPU memory allocation failed         |
| 200  | `ERROR_CUDA_MEMCPY`             | Host↔device memory copy failed       |
| 300  | `ERROR_CUDA_SYNCHRONIZE`        | CUDA stream synchronization failed   |
| 400  | `ERROR_CUDSS_SET_PIVOT`         | `cudssConfigSet` (pivot type) failed |
| 500  | `ERROR_CUDSS_MATRIX_CREATE_DN`  | `cudssMatrixCreateDn` failed         |
| 550  | `ERROR_CUDSS_MATRIX_SET_VALUES` | `cudssMatrixSetValues` failed        |
| 600  | `ERROR_CUDSS_MATRIX_CREATE_CSR` | `cudssMatrixCreateCsr` failed        |
| 700  | `ERROR_CUDSS_SYM_FACTORIZATION` | Symbolic factorization failed        |
| 800  | `ERROR_CUDSS_NUM_FACTORIZATION` | Numeric factorization failed         |
| 900  | `ERROR_CUDSS_SOLVE`             | Solve phase failed                   |

---

## Rust Layer (`src/solver_cudss.rs`)

### Opaque FFI Handle

```rust
#[repr(C)]
struct InterfaceCUDSS {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
```

Uses the zero-sized `[u8; 0]` + `PhantomData` pattern to represent the C struct opaquely. Both `InterfaceCUDSS` and `SolverCUDSS` are `unsafe impl Send`.

### Extern "C" Declarations

```rust
extern "C" {
    fn solver_cudss_new() -> *mut InterfaceCUDSS;
    fn solver_cudss_drop(solver: *mut InterfaceCUDSS);
    fn solver_cudss_initialize(solver, verbose: CcBool, ...) -> i32;
    fn solver_cudss_factorize(solver, verbose: CcBool, values: *const f64) -> i32;
    fn solver_cudss_solve(solver, x: *mut f64, rhs: *const f64, verbose: CcBool) -> i32;
}
```

Boolean parameters use `CcBool` (= `i32`, with 1/0 for true/false).

### `SolverCUDSS` Struct

| Field                | Type                  | Purpose                                |
| -------------------- | --------------------- | -------------------------------------- |
| `solver`             | `*mut InterfaceCUDSS` | C pointer                              |
| `csr`                | `Option<CsrMatrix>`   | CSR copy for subsequent factorizations |
| `initialized`        | `bool`                | Mirrors C `initialization_completed`   |
| `factorized`         | `bool`                | Mirrors C `factorization_completed`    |
| `initialized_sym`    | `Sym`                 | Saved symmetry type from first call    |
| `initialized_ndim`   | `usize`               | Saved dimension from first call        |
| `initialized_nnz`    | `usize`               | Saved nnz from first call              |
| `stopwatch`          | `Stopwatch`           | Cumulative timer                       |
| `time_initialize_ns` | `u128`                | Initialize time in nanoseconds         |
| `time_factorize_ns`  | `u128`                | Factorize time in nanoseconds          |
| `time_solve_ns`      | `u128`                | Solve time in nanoseconds              |

### `Drop` Implementation

Calls `solver_cudss_drop(self.solver)` to free all C-side resources.

### `LinSolTrait` Implementation

**`factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>)`**

On first call:
1. Validates matrix (square, nnz > 0, symmetry ≤ `Sym::YesLower`)
2. Converts COO → CSR, stores in `self.csr`
3. Saves `initialized_sym`, `initialized_ndim`, `initialized_nnz`
4. Applies parameters: `verbose`, `positive_definite`
5. Derives `general_symmetric` from `mat.symmetric == Sym::YesLower`
6. Calls `solver_cudss_initialize()`
7. Records `time_initialize_ns`

On subsequent calls:
1. Validates structure unchanged (sym, ndim, nnz same as first call)
2. Updates CSR values from new COO matrix
3. Calls `solver_cudss_factorize()`
4. Records `time_factorize_ns`

**`solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool)`**
1. Validates factorization is done
2. Checks vector dimensions match `initialized_ndim`
3. Calls `solver_cudss_solve()`
4. Records `time_solve_ns`

**`update_stats(&self, stats: &mut StatsLinSol)`**
Populates solver name and timing fields.

**`get_ns_init/fact/solve()`**
Return individual timing measurements.

### Error Mapping

`handle_cudss_error_code(err: i32) -> StrError` maps each C integer code to a specific Rust error string identifying the failed operation, with a catch-all fallback for unknown codes.

---

## Key Design Points

1. **C layer knows nothing of COO/CSR types** — it receives raw `int*` and `double*` arrays. All COO→CSR conversion and validation lives in Rust.

2. **Structure-once, factorize-many** — symbolic factorization (initialize) happens only on the first `factorize()` call. Subsequent calls with different values reuse the same GPU allocations and matrix structure, only re-running numeric factorization.

3. **Resource ownership** — the C struct owns all GPU and cuDSS resources. Rust holds only an opaque pointer and triggers cleanup via `Drop`.

4. **Synchronization** — every cuDSS operation is followed by `cudaStreamSynchronize` to ensure completion before returning to the caller or copying data.

5. **Tests are serialized** — `#[serial]` attribute prevents concurrent GPU access across tests.
