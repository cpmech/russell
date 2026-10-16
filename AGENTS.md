# AGENTS.md

Notes for automated agents and contributors working in this repository.

## Build backend (read this first)

`russell_lab` has **two** BLAS/LAPACK backends, and the default needs nothing
installed:

- **Native Rust (default)** — `russell_lab/src/native/`, plain Rust, nothing to
  find and nothing to link. It builds for any target, including
  `wasm32-unknown-unknown`, which has no libc for the C shim and no BLAS to link.
  This is what `cargo build` and `cargo test` use with no extra flags.
- **External library** — the C shim in `russell_lab/c_code/interface_blas.c`,
  selected by exactly one of:
  - **`oblas`** — OpenBLAS, resolved via `pkg-config` with a Homebrew fallback.
    Needs the development headers (e.g. `lapack.h`). Also the reference netlib
    BLAS/LAPACK.
  - **`mkl`** — Intel oneAPI MKL, expected under
    `/opt/intel/oneapi/mkl/<version>`; set `MKL_VERSION` to override `latest`.

`oblas` and `mkl` are mutually exclusive; `build.rs` panics if both are on, rather
than leaving the linker to report two providers of the same symbols. Enabling
neither prints which path was taken, so a build log always records it.

The `oblas` and `mkl` features are forwarded from every dependent crate
(`russell_tensor`, `russell_sparse`, `russell_stat`, `russell_ode`,
`russell_pde`, `russell_nonlin`). Their own defaults are empty: they do **not**
force an external library, and Cargo cannot disable a dependency's default
features from the outside, so a crate that wants one has to say so in the
dependency declaration.

### What the native backend does not implement yet

The eigen-decompositions (`dsyev`, `dgeev`, `dggev`), the SVD (`dgesvd`), and
everything complex (`zpotrf`, `zgetrf`, `zgetri`, `zgesv`, `zheev`, `zgeev`,
`zggev`, `zgesvd`) are not written yet. Calling one returns a runtime error naming
the feature to enable — deliberately *not* a link error, so the rest of the suite
still builds and runs. Build with `--features oblas` (or `mkl`) when you need
them.

Implementing the rest is expected to follow `native/lapack.rs`'s existing shape:
column-major, 0-based, no workspace arguments.

### The `oracle` feature (development only)

`--features oracle` compiles the shim against a **locally built** LAPACK and links
it, so the native routines can be tested for numerical parity against the
reference. This is the only practicable way to check an eigenvalue or an SVD.

It requires `RUSSELL_ORACLE_ROOT` pointing at a CMake build directory of LAPACK
3.12.1 with `include/` and `lib/` holding `lapack.h`, `cblas.h`, `liblapack.so`
and `libcblas.so`. There is **no default**, deliberately: hardcoding one would put
a developer's directory layout into released source. `build.rs` checks for the
files and fails with an explanation rather than a link error.

```bash
RUSSELL_ORACLE_ROOT=/path/to/lapack/build cargo test -p russell_lab --features oracle
```

Anything the oracle links is a reference for the *tests*; the crate's own symbols
come from the native or accelerator path.

Extra features:

- `local_sparse` (and `cudss`) on `russell_sparse`, `russell_ode`,
  `russell_pde`, `russell_nonlin` — require locally compiled MUMPS/SuiteSparse
  (see `zscripts/*-compile-mumps.bash` and `zscripts/*-compile-suitesparse.bash`).

## Test / lint commands

```bash
# Tests, native backend: no external library needed.
cargo test -p russell_lab
cargo test -p russell_tensor

# Tests, external backend (needs OpenBLAS headers, or MKL installed).
cargo test -p russell_lab --features oblas
cargo test -p russell_lab --features mkl

# Numerical parity of the native routines against the reference LAPACK.
RUSSELL_ORACLE_ROOT=/path/to/lapack/build cargo test -p russell_lab --features oracle

# Whole workspace (sparse crates need the locally compiled libraries)
cargo test --workspace --features local_sparse

# Formatting (CI runs the check)
cargo fmt --all
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets

# Coverage (nightly + llvm-tools-preview); project requires >95% lines
cargo llvm-cov --workspace --features local_sparse \
  --ignore-filename-regex 'build.rs|mem_check.rs|mem_check_lab.rs|solve_matrix_market.rs|amplifier1t.rs|brusselator_pde.rs' \
  --fail-under-lines 95
```


`cargo test` also runs the doc examples. Always keep the tree rustfmt-clean and
warning-free.

## Coding conventions

- **Double lower-case names denote a single capital letter**, matching the symbols
  used in doc comments and the mathematical formulas: `aa` ⇔ `A`, `bb` ⇔ `B`,
  `pp` ⇔ `P`, ... For example, the scalar coefficients `A`, `B`, `C` in the
  derivative formulas of the invariants are coded as `aa`, `bb`, `cc`.
- **Index ranges:** `m`, `n` (and `p`, `q`) are reserved for the Kelvin-Mandel
  components, ranging over `1..N`. Do **not** use them as dummy Cartesian
  indices (which range over `1,2,3`); use `r`, `s`, `t`, ... for those instead
  (e.g. `∂I3/∂a_ij = ½ ϵ_ikl ϵ_jrs a_kr a_ls`).
- **Permutation (Levi-Civita) tensor:** use the *lunate* epsilon `ϵ` (U+03F5) in
  formulas, not the curly `ε` (U+03B5), which is reserved for strain.
- **Prefer the `small_*` helpers** from `russell_lab` over the generic `mat_*` / `Matrix`
  versions whenever the size is known at compile time — e.g. for 3×3 (second-order tensor)
  and 2×2 operations. Use `small_mat_mat_mul`, `small_mat_t_mat_mul`, `small_mat_inv`,
  `small_mat_eigen_sym_jacobi`, etc., which operate on stack `[[f64; N]; N]` / `[f64; N]`
  arrays and avoid heap allocation; get the stack matrix from a tensor with
  `Tensor2::to_std_matrix_slice` (or `to_std_matrix_slice`-style helpers).
- **No developer paths in source.** Not in code, not in defaults, not in doc
  comments. An absolute path under someone's home directory ends up published the
  moment the crate is. Use an environment variable, and prefer *requiring* it over
  supplying a default.

## BLAS/LAPACK invariants

These are the rules the code depends on. Each one has been violated at least once.

- **Matrix layout.** Everything in `russell_lab` is **column-major**, and
  `native::blas` / `native::lapack` index `data[i + j * n]` to match `Matrix`.
  `lapack.h` binds directly to the Fortran routines and is therefore column-major
  too. **`LAPACKE_*` is the row-major API and will silently transpose every
  matrix** — never use it here.
- **LAPACK's string arguments are a hidden-argument ABI.** `lapack.h` in a build
  with `LAPACK_FORTRAN_STRLEN_END` passes hidden trailing string lengths
  (`FORTRAN_STRLEN` = `size_t`). Prefer the shim's `c_*` wrappers, which take
  `C_BOOL` and supply those lengths themselves; hand-declaring the ABI is an easy
  way to get garbage rather than an error.
- **A routine that writes one triangle leaves the other alone.** `dpotrf` with
  `'L'` writes only the lower triangle and `'U'` only the upper; the rest keeps
  whatever the caller had. So: mirror the triangle you *read* into the one you
  need (the caller's other half may legitimately be zeros), and write back only
  the requested triangle. `dsyrk` behaves the same way.
- **Check the contraction length, not just the output shape.** `C := A Aᵀ` and
  `C := Aᵀ A` are both `(n,n)` but contract over different lengths — `k` from `A`'s
  columns in one case and from its rows in the other. A dispatch that passes the
  same shape to both is silently wrong for non-square `A`. This bit
  `mat_sym_rank_op`.
- **`idamax` is 1-based in BLAS and 0-based in Rust.** `find_index_abs_max` and
  `native::blas::idamax` return 0-based indices; the first maximum wins.
- **Scale before squaring.** A naive sum of squares overflows for entries around
  `1e200` and underflows around `1e-200`. `dnrm2` and `dznrm2` use Blue's scaling.

## Dispatch between the native and external backends

Every call site that needs BLAS follows the same shape:

```rust
#[cfg(any(feature = "oblas", feature = "mkl"))]
{
    // the C call, with a comment naming the shim's symbol
}
#[cfg(not(any(feature = "oblas", feature = "mkl")))]
// the native call, which uses usize lengths and no workspace argument
```

Two things worth knowing before adding one:

1. **The `extern "C"` declaration must be gated with the same condition.** If it
   is not, a native-only build fails to link. If it *is* but the call site is not
   dispatched, the build fails to compile — so migrate the dispatch **first** and
   gate the declaration second, one file at a time. Gating first turns every
   unmigrated call site into an error and leaves the tree unable to run tests.
2. **Routines without a native implementation return
   `LapackError::NotImplemented`** rather than being left to fail at link time. A
   runtime error can name the feature to enable; a link error about a mangled
   symbol cannot.


## Code intelligence (CodeGraph)

This repository is indexed by CodeGraph (a `.codegraph/` directory exists at the
root; see the global AGENTS.md for the full guidance). Reach for
`codegraph_explore` **before** grep/read when you need to understand or locate
code — one call returns the relevant symbols' verbatim source plus the call
paths, which is cheaper and more accurate than a search/read loop.

The index keeps itself fresh: a file watcher with a debounced auto-sync
(~2 s, `CODEGRAPH_WATCH_DEBOUNCE_MS`), a per-file staleness banner on tool
responses, and a connect-time catch-up. A path can therefore look stale for a
few seconds right after a rename/create — this once surfaced `eigen2_values.rs`
moments after it had been renamed to `eigen_values.rs` (the next check already
showed the correct name). When that happens, or when a tool response carries the
staleness banner, verify the filename on disk (`ls` / glob) and `Read` the
specific file for line-level edits.

If a path is still wrong after the debounce window — or the watcher is disabled
(e.g. a sandbox, or `CODEGRAPH_NO_DAEMON=1`) — force a refresh:
`codegraph status` (reports a `### Pending sync:` list), `codegraph sync`
(incremental) or `codegraph index` (full rebuild); `codegraph unlock` clears a
stale lock file.

## CI

Workflows live in `.github/workflows/` (`ubuntu.yml`, `macos.yml`, `rocky.yml`,
`arch.yml`, `windows.yml`). The Ubuntu job compiles MUMPS/SuiteSparse via
`zscripts/debian-compile-*.bash`, runs `cargo test --features local_sparse`, and
on nightly enforces the coverage threshold shown above.
