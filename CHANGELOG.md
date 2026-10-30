# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.1.0] - 2026-10-29

### `russell_nonlin`

#### Fixed

- `Stop::MinLambda` and `Stop::MaxLambda` now land exactly on the target under
  `Method::Arclength`. The lambda-imposed mode is entered when the remaining
  distance lies inside the step rather than only on a predictor overshoot, the
  step is capped so it cannot overshoot the target, and a polish step closes the
  drift left by a listed Δλ; previously a segment ended up to 1e-6 short (and
  3.4e-3 short on a curved path with a listed Δλ).
- The θ parameter is reset at the start of every arclength step, so the
  lambda-imposed mode is a per-step decision instead of a leftover from the
  previous step.
- The step length of a lambda-imposed step is computed once per target, so a
  rejected step retries with the halved length instead of repeating the same
  failing step until `Status::ContinuedFailure`.

### `russell_sparse`

#### Added

- `DirectMethod { Auto { flop_ratio_threshold }, Simplicial, Supernodal }` and
  `LinSolParams::direct_method`, so the caller picks the LU factorization and the threshold is
  visible as `DirectMethod::DEFAULT_FLOP_RATIO_THRESHOLD`. The proxy it compares is the sum of the
  squared supernode column counts over `nnz(A)`, taken from the symbolic analysis the solver runs
  anyway. Re-exported from the prelude.

#### Changed

- The default is `DirectMethod::Auto { 1.0 }`: supernodal above the proxy, simplicial at or below
  it. It used to be simplicial. Measured on a 192-dof lattice with three dofs per node (fill 2.96),
  the supernodal factorization takes 0.604 ms against 1.248 ms, because the front elimination is
  contiguous down the columns of a column-major panel while the simplicial path gathers per entry.
  A small, sparse pattern can still prefer `DirectMethod::Simplicial`.

#### Changed (breaking)

- `LinSolParams::direct_supernodal` is removed; `direct_method` replaces it, with no alias. A caller
  that set the field must now choose `DirectMethod::Supernodal` (same result),
  `DirectMethod::Simplicial`, or `DirectMethod::Auto`. A `russell_sparse = "5"` requirement picks
  this up on `cargo update`; pin `=5.0` to stay on the old field.

#### Documentation

- The README says what the native iterative solvers can do and where they do not: with the default
  algebraic multigrid they reach a relative error of 1e-6 on BiCGStab 12 of 33 real SuiteSparse
  matrices, GMRES 7 and CG 1, so production work belongs on Direct, UMFPACK or MUMPS and the
  iterative path is used where it has been measured to converge. The introduction no longer calls
  the library a solver of "large sparse systems" for all four backends.
- MUMPS is described as supporting multithreaded execution, which is what it does; "parallel"
  also reads as distributed or multi-process.
- The CUDA references left over in the umfpack architecture note and in the statistics docs are
  gone.

## [5.0.0] - 2026-10-28

### Changed

- `Genie::Native` is now `Genie::Direct`, and `SolverNative` is now `SolverDirect`.
- The `LinSolTrait` methods `factorize` and `get_ns_fact` are now `preprocess` and
  `get_ns_preprocess`.
- The statistics phases `factorize_array` and `factorize` are now `preprocess_array`
  and `preprocess`.
- `LinSolParams.native_supernodal` is now `direct_supernodal`.
- `Genie::from` returns a `Result` and rejects unknown names.
- `russell_sparse` depends on `algebra_linear` 2.

### Added

- `Genie::Iterative(IterativeMethod)` with CG, BiCGStab, and restarted GMRES,
  using algebraic multigrid or identity preconditioning.
- `LinSolParams::iterative` with the tolerance, iteration limit, restart, thread,
  and AMG settings.
- The `solve_matrix_market` options `--iter-*` and `--amg-*`, and the `cg`,
  `bicgstab`, and `gmres` values for `--genie` (`-g`).
- The `StatsLinSolIterative` block in the JSON statistics.

### Removed

- The old names and their compatibility shims: there are no aliases, by design.

### Fixed

- `StatsLinSol.blas_lib` reports `Native` for a pure-Rust build instead of claiming
  OpenBLAS.
- The Direct adapter's pattern check now sees a reordered matrix.

## [4.0.1] - 2026-10-27

### Changed

- The READMEs of every crate use one wording for the feature and library
  dependency instructions: the default build needs no external library, and each
  crate describes the features it exposes with the same sentences. The root
  README no longer calls OpenBLAS the default backend. `russell_sparse` is
  published as 4.1.1 because 4.1.0 is already on crates.io.

## [4.1.0] - 2026-10-21

### Added

- `LinSolParams.native_supernodal` selects the experimental supernodal LU in the
  Native solver. The CLI exposes it as `--native-supernodal`.

### Changed

- The Native solver (`SolverNative`, `ComplexSolverNative`) uses the simplicial LU
  by default. It is faster than the supernodal path on every matrix in the
  downstream benchmark; the supernodal path remains available for experiments.
- `russell_sparse` depends on `algebra_linear` 1.2, which compresses the supernodal
  fronts, applies the relaxation pass, skips the unused supernodal symbolic
  analysis, and aligns the automatic method threshold with the upstream one.

## [4.0.0] - 2026-10-21

### Changed (breaking)

- **The default build needs no external library.** `russell_lab` now performs dense
  linear algebra with the pure-Rust kernels of
  [`algebra_linear`](https://crates.io/crates/algebra_linear); OpenBLAS and Intel MKL
  are opt-in accelerators (`open_blas`, `intel_mkl`), no longer defaults.
- **`russell_sparse` has a pure-Rust default solver.** `Genie::Native`
  (`SolverNative`, `ComplexSolverNative`) solves sparse systems with an
  `algebra_linear` sparse LU. It needs no external library and runs on every target,
  WebAssembly included.
- **UMFPACK is opt-in** through the `umfpack` feature; the C interface was compiled
  by default before. The build looks for a local install under
  `RUSSELL_LOCAL_PREFIX` first and falls back to the package-manager library. The
  old `local_umfpack` feature is gone: `umfpack` covers both sources.
- **MUMPS is separate** (`mumps`), always compiled locally and found through
  `RUSSELL_LOCAL_PREFIX`. It is available on Linux, macOS, and Windows with the GNU
  toolchain (MSYS2).
- **Determinant evaluation is not available for `Genie::Native`**: requesting it
  returns `"determinant evaluation is not available for this Genie"`.
- **Edition 2024 and Rust 1.85** are required.

### Added

- `Genie::Native`, `SolverNative`, `ComplexSolverNative`, and `SolverNative::stats()`,
  which exposes fill ratio, flop estimate, and timings.
- UMFPACK for the MSVC toolchain on Windows, taken from vcpkg; the MSYS2 route keeps
  UMFPACK and MUMPS.
- MUMPS compilation on macOS (`zscripts/macos-compile-mumps.bash`, built with
  Homebrew's gfortran and Accelerate).
- WebAssembly support for `wasm32-unknown-unknown` and `wasm32-wasip1`, with a
  browser smoke test in CI.
- `CHANGELOG.md` (this file).

### Notes

- `StatsLinSol` is unchanged; the Native counters are exposed by
  `SolverNative::stats()`.
- A binary distribution that links `russell_lab` also links `algebra_linear`
  (BSD-3-Clause) and must reproduce its notice; see the crate README.
