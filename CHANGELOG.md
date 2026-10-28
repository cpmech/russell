# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
