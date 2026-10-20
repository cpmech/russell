# AGENTS.md

Notes for automated agents and contributors working in this repository.

## Build backend (read this first)

`russell_lab` has **two** BLAS/LAPACK backends, and the default needs nothing
installed:

- **`algebra_linear` (default)** — a separate crate of pure Rust kernels, nothing to
  find and nothing to link. It builds for any target, including
  `wasm32-unknown-unknown`, which has no libc for the C shim and no BLAS to link.
  This is what `cargo build` and `cargo test` use with no extra flags.
- **External library** — the C shim in `russell_lab/c_code/interface_blas.c`,
  selected by exactly one of:
  - **`open_blas`** — OpenBLAS, resolved via `pkg-config` with a Homebrew fallback.
    Needs the development headers (e.g. `lapack.h`). Also the reference netlib
    BLAS/LAPACK.
  - **`intel_mkl`** — Intel oneAPI MKL, expected under
    `/opt/intel/oneapi/mkl/<version>`; set `MKL_VERSION` to override `latest`.

`intel_mkl` takes precedence: enabling both links MKL (OpenBLAS is compiled in only under
`open_blas` without `intel_mkl`), and `build.rs` says so in the build log. There is no
`compile_error!` for the combination, so `--all-features` is a valid build and exercises the
MKL path. Enabling neither prints which path was taken, so a build log always records it.

The `open_blas` and `intel_mkl` features are forwarded from every dependent crate
(`russell_tensor`, `russell_sparse`, `russell_stat`, `russell_ode`,
`russell_pde`, `russell_nonlin`). Their own defaults are empty: they do **not**
force an external library, and Cargo cannot disable a dependency's default
features from the outside, so a crate that wants one has to say so in the
dependency declaration.

### Reference parity lives in `algebra_linear`

This crate has no reference-LAPACK harness any more: the `dev_lapack` feature moved
with the kernels. To compare the kernels against a locally built LAPACK 3.12.1, run
it there (`ALGEBRA_LINEAR_DEV_LAPACK_ROOT=... cargo test --features dev_lapack` in
the `algebra_linear` checkout). Nothing here needs a reference build, and there is no
`RUSSELL_DEV_LAPACK_ROOT` any more.

Extra features:

- `local_umfpack` and `mumps` on `russell_sparse`, `russell_ode`, `russell_pde`,
  `russell_nonlin` — use the locally compiled SuiteSparse and MUMPS installed by
  `zscripts/*-compile-suitesparse.bash` and `zscripts/*-compile-mumps.bash`. Both
  read `RUSSELL_LOCAL_PREFIX` (`.../include/umfpack`, `.../lib/umfpack`,
  `.../include/mumps`, `.../lib/mumps`) and fail the build if it is unset or wrong.
  Without `local_umfpack` the UMFPACK comes from the package manager, found with
  pkg-config; UMFPACK is required in every configuration, `mumps` is optional.

## The BLAS/LAPACK kernels live in `algebra_linear`

The port is no longer in this repository: it was extracted to
[`algebra_linear`](https://crates.io/crates/algebra_linear) (`BSD-3-Clause`), which
this crate depends on. Everything about the port — its conventions, its measured
state, the bug history, the reference-parity harness (`dev_lapack` lives there, not
here), and the traps it has already paid for — is in that crate's `AGENTS.md` and
`README`. Read those before touching a kernel; nothing in this repository implements
one.

What stays here: the dispatchers (`base::blas_dispatch`, `base::lapack_dispatch`),
the C shim and `build.rs` that link OpenBLAS or MKL, `Matrix`/`Vector`, the `small_*`
stack kernels, and every public API.

> **Historical, in part.** The rest of this section and the sections below
> ("Performance: the blocking boundary", "Loose ends", "Deliberately uncovered
> code", "BLAS/LAPACK invariants") describe the kernels, which now live in the
> `algebra_linear` crate — the `src/native/...` paths in them no longer exist here.
> Read them together with that crate's `AGENTS.md`; the dispatch rules below apply
> to this repository.

### Performance: the blocking boundary

Worth knowing before optimizing, because the obvious move is the wrong one.

`mat_mat_mul` routes through `base::blas_dispatch::dgemm`, which is
`native::blas::dgemm` -- a Goto-style blocked kernel with a 4x4 AVX2+FMA
micro-kernel (MC=128, KC=256, NC=256) -- unless an accelerated feature is on.
There is no `portable` module any more: `wasm32` takes the same dispatch and gets
the native kernel's scalar fallback, and the suite runs there under `wasmtime` in
CI. Benchmarks:
`benchmarks/matmul_benchmark.rs` and `benchmarks/decomp_benchmark.rs`.

**The LAPACK layer is BLAS-3-free end to end**, so the blocked `dgemm` does not
accelerate any decomposition:

| Routine | Kernel used | BLAS-3? |
|---|---|---|
| `dsyev` | `dsytd2`, reflectors via a hand-written `dgemv` loop | no |
| `dgesvd` / `zgesvd` | `dgebd2` / `zgebd2` | no |
| `dgeev`, `dhseqr` | `dlahqr` (`dlarfg`/`dlanv2`/`dlartg` only) | no |
| `dggev` / `zggev` | `dhgeqz` | no |

Measured single-core, release: `mat_mat_mul` went 2.2 -> 21 Gelem/s, while `dsyev`
runs at 0.4 ns/flop (~2.8 GFLOP/s) and `dgesvd` at 0.2 ns/flop, against the
kernel's 0.05 ns/flop.

`dsytrd` was ported anyway and delivered far less than that headroom suggested:
~8-10% on `dsyev` at n >= 400, because the reduction is only ~29% of `dsyev` and
the blocked algorithm still performs the same n^3/3 symmetric product -- only the
rank-2 update becomes BLAS-3. Real headroom for `dsyev` needs the two-stage
`sy2sb` + `sb2st` reduction, which is a different algorithm, not a blocking of
this one. `dgesvd` remains unblocked. The rules of thumb:

* Port the blocked variant only when the target routine's flop count is dominated
  by panel updates that can become `dgemm`. `dlahqr` can never benefit -- it has
  no BLAS-3 to give.
* **A blocked routine can be slower than the unblocked one for small input.**
  `dsytrd` measured slower below n ~ 300 and faster above, so `dsyev` picks by
  size (threshold 128) rather than always paying for the blocking. Always
  blocking would have regressed every small case.
* **Some quantities are not invariants.** The first `dsytrd` test compared the
  reduced `d`/`e`/`tau` entries element-wise against the unblocked reduction, to
  1e-12. That is unsatisfiable and the test was at fault: the two reductions use
  different Householder sign conventions, so their off-diagonals are not
  comparable. Against reference LAPACK's own `dsytrd` and `dsytd2` on the same
  problem, `|e|` differs by 4e-1 at n=128 while the eigenvalues agree to 6e-14,
  and the reference's blocked-vs-unblocked `|d|` alone reaches 3.2e-12. Check the
  spectrum, which is invariant, not the intermediate factors.
* The user-facing `mat_mat_mul` is where the kernel change pays, and it is the
  entry point users actually call, which is why it was worth doing first.
* `dlaqr0` has one benefit that is *not* about throughput: aggressive early
  deflation detects converged eigenvalues many sweeps earlier, cutting the
  iteration count independently of kernel speed. It is the only part with no
  `dlahqr` analog.

### Loose ends

- The crate is warning-free and clippy-clean (`cargo clippy --workspace
  --all-targets` exits 0). Keeping it that way needs the pattern described under
  "Dispatch between the native and external backends": the shim-only imports are
  `cfg`-gated and the functions with a shim branch carry a `cfg_attr`'d allow.
  Note that a warning count is not a pass/fail signal by itself -- clippy's
  deny-by-default lints are *errors*, so read the exit code, not the count (a
  `| grep -c warning` once reported "0 warnings" on a build that had failed).
- `dhseqr`/`zhseqr` implement the unblocked (`dlahqr`/`zlahqr`) path only, which
  is what the reference uses for n <= 15; the blocked `dlaqr0`/`zlaqr0` fallback
  for large matrices is not ported. Non-convergence `INFO` is *reported*, not
  asserted: `dsyev`/`zheev` (via `dsteqr`/`zsteqr`), `dhseqr` (via `dlahqr`) and
  `dgesvd`/`zgesvd` (via `dbdsqr`/`zbdsqr`) return `LapackError::Convergence`.
  `dhseqr`, `dgeev` and `dgesvd` are exercised through NaN-matrix tests;
  `dsyev`/`zheev` are not, because a NaN matrix reads as zero under the norm gate
  and takes the reference's `ANORM == 0` early return.
- `dgesvd`/`zgesvd` use the direct bidiagonalization path for both shapes, not
  the QR/LQ pre-reduction the reference uses for very tall or wide inputs.
  Agreement with the reference is ~1e-15.
- `--features open_blas` builds and passes on this machine: the headers are under
  `/usr/include/openblas` and `pkg-config` finds them. All configurations (plain,
  `open_blas`, `intel_mkl`, `--all-features`, `dev_lapack`, and the wasm target)
  are runnable here, so an accelerated branch never has to be reasoned about
  without being checked.
- The web/WASM consumer of this crate has not been re-checked since the native
  backend landed.


## Test / lint commands

```bash
# Tests, native backend: no external library needed.
cargo test -p russell_lab
cargo test -p russell_tensor

# Tests, external backend (needs OpenBLAS headers, or MKL installed).
cargo test -p russell_lab --features open_blas
cargo test -p russell_lab --features intel_mkl

# Numerical parity of the native routines against the reference LAPACK.
# reference parity now runs in the algebra_linear checkout:
ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build cargo test --features dev_lapack

# Whole workspace (sparse crates need the locally compiled libraries)
RUSSELL_LOCAL_PREFIX=/usr/local cargo test --workspace --features local_umfpack,mumps

# Formatting (CI runs the check)
cargo fmt --all
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets

# Coverage (nightly + llvm-tools-preview); project requires >95% lines, counted once
# per source line (llvm-cov's `--fail-under-lines` counts a line once per
# monomorphization, so const-generic dimension dispatch reads as uncovered; the
# `DA:` records of the lcov export are merged per line and are what the gate uses)
cargo llvm-cov --workspace --features local_umfpack,mumps \
  --ignore-filename-regex 'build.rs|mem_check.rs|mem_check_lab.rs|solve_matrix_market.rs|amplifier1t.rs|brusselator_pde.rs' \
  --lcov --output-path lcov.info
bash zscripts/check-coverage.bash lcov.info 95
```


## Deliberately uncovered code

The 95% gate counts source lines, so what is left uncovered is a decision, not an
accident. This is where those decisions are recorded. **A PR that adds code no test
can reach adds it here in the same PR** — otherwise the next person spends a day
rediscovering why a file sits at 84%.

Measured 2026-10-19 with the command above: workspace **74,192 / 76,273 lines =
97.27%**, 2,081 missed. The native/LAPACK rows were refreshed on 2026-10-20 from the
lib-only run quoted below, which reports the same counts for those files; every other
row is unchanged from the workspace measurement.

| where | missed | why |
|---|---|---|
| `russell_lab/src/native/lapack/qz.rs` | 547 | `dhgeqz`'s and `dggbal_p`'s rescue and permutation arms, `dtgevc`'s guard paths. Sweeping `dgeev` and `dggev` over the reference's own spectra — including a singular `B` — moved this file by zero lines, because those families are built to be *solvable* and cannot reach code whose job is to refuse or rescue. `dget54`, the checker for those factors, needs `Q` and `Z` out of `dggev`; the owner decided the port's API does not grow for it, so that check is not writable here |
| `native/lapack/zeig.rs` | 204 | `zsteqr`'s remaining deflation arms, `zlarfg`'s underflow rescue, `zlatrs_upper`'s rescaling |
| `native/lapack/eig.rs` | 203 | `dlaln2` and `dlanv2`: 2x2 blocks whose entries force a rescale |
| `native/lapack/sym.rs`, `svd.rs`, `zsvd.rs` | 96, 76, 51 | `dsteqr`'s remaining arms, `dbdsqr`/`zbdsqr` convergence for repeated and zero singular values |
| `russell_pde/src/transfinite_samples.rs`, `russell_lab/src/algo/bspline.rs` | 79, 50 | `if SAVE_FIGURE { ... }` blocks: plotting code that writes SVG to a developer's disk |
| `russell_tensor/src/polar_higham.rs` | 67 | the `dd2 == 0` branch, which needs an isotropic spectrum, and the ill-conditioned fallbacks |
| `russell_nonlin/src/solver.rs`, `solver_arclength.rs` | 33, 27 | scenario code: the `dlambda/ds` guard needs a turning point reached while the listed step is still legal, the precise-stop redo needs `Stop::UComp`, and the failure limits need a deliberately failing system |
| `russell_sparse/src/util.rs` | 31 | the `nvidia-smi` probe without a GPU, the `/proc/cpuinfo` fallback that `lscpu` pre-empts, and the false arms of `cfg!(feature = ...)` in a build that has the feature |
| `native/lapack/examples.rs` | 18 | short-circuited false arms of the two example checkers. `DGET33`'s structural-violation penalty and `DGET31`'s `scale > 1` and `INFO ∉ {0, 1}` penalties are unreachable for a correct routine (`DGET31`'s own run reports `NINFO(1) = 0`); the tails of the `&&` chains in the `RES = ZERO` tests never evaluate; and the 2x2 complex arm of the `INFO` transcription is never selected by this grid, whose largest pivoted entry is always a diagonal one |
| `native/lapack/auxiliary.rs` | 74 | the shared helpers' rescue arms: `dladiv`'s seven rescale-branch lines are constant-folded out when the fixture's literals reach the inliner (its twelve cases match the reference bit for bit, but the compiler removes the branches it can decide), `lascl_g`'s guards for a zero `cfrom` and for overflow, and the degenerate blocks of `dlae2`/`dlaev2` that the symmetric sweeps never produce |
| `native/lapack/families.rs` | 43 | harness argument checks the tests do not exercise, the `wn == 0` arm of a reflector built from a zero random vector in `dlagge`/`dlagsy` (probability zero), `dlatms`'s `INFO = 2` guard for an all-zero spectrum (unreachable: `latm1` always scales the maximum to one), and the assertion-failure lines of the fixture tests |
| `russell_tensor/src/eigen_projectors.rs` | 26 | `if VERBOSE { ... }` diagnostics |

The remaining ~400 lines are spread over many files, none above 25.

**Measured 2026-10-20 for the SEP and SVD sweeps** (`cargo llvm-cov -p russell_lab
--lib --lcov`, 39,065 / 40,559 lines = 96.32%, 1,494 missed): the four sweeps moved
**122 production lines**, and they were not evenly distributed:

| sweep | moved | where |
|---|---|---|
| `dsyev` | 6 | `dsteqr` |
| `zheev` | 99 | `zsteqr` 93, `zlasr_v` 6 |
| `dgesvd` | **0** | — |
| `zgesvd` | 17 | `zbdsqr` |

So the value here is not the line count. `dgesvd` moved nothing because `dbdsqr`'s
convergence and rescue arms were already reached by the hand-written shapes in
`svd.rs`; the family grid adds cases, not branches. The complex twins moved the most
for the opposite reason — `zsteqr` had almost no coverage before, so an ordinary
well-conditioned spectrum was enough.

Two rules that follow:

- **Do not refactor production code to satisfy the tool.** The five dyadic kernels in
  `russell_tensor` read as 50% covered because `--fail-under-lines` counts a line once
  per monomorphization; the `DA:` gate exists so that this is not a reason to rewrite
  working code.
- **When a gap is called unreachable, say what was tried.** The LAPACK entry above is
  trustworthy because five generators were ported and verified against LAPACK's own
  build (`families.rs`) before the sweeps were run.
- **The percentage moves when function layout changes, and that is the tool, not the
  tests.** llvm-cov's `DA:` records include lines that carry no code — comments, a
  `const` declared inside a function, a literal argument on its own line — whenever
  inlining puts a region boundary there. Consolidating two copies of `dlaln2` into
  `auxiliary.rs` moved the gate from 96.60% to 95.37% while every executed line kept
  its hit count (the `dgebal` entry went from 139 to 139): of `eig.rs`'s 586 missed
  lines, 416 have no code in them. Judge a refactor by the hit counts of the lines
  that do work, and say so in the PR instead of hunting the total.

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
  `small_mat_svd` is the one exception to "no heap": it wraps `dgesvd` (native
  Rust by default, the C shim with `open_blas`/`intel_mkl`) and `dgesvd` allocates its own
  workspace. Do not reintroduce an `AᵀA`-based SVD for it — forming `AᵀA` squares
  the condition number, and at κ = 1e8 that left `russell_tensor`'s polar
  rotation non-orthogonal by 0.28.
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
- **Non-convergence is an error, not a debug assertion.** An `INFO > 0` from an
  iterative routine -- `dsteqr`, `zsteqr`, `dlahqr`, `dbdsqr`, `zbdsqr`, and the
  QZ iterations -- becomes `LapackError::Convergence`, never a `debug_assert`.
  A debug assertion only fires in a debug build, so a release build handed back
  partly converged eigenvalues and singular vectors as if they had converged.
  `dhseqr`, `dgeev` and `dgesvd` are tested for this with a NaN matrix (the
  iteration limit really is exhausted); `dsyev`/`zheev` cannot be, because the
  norm gate reads NaN as zero and takes the zero-matrix early return.

## Dispatch between the native and external backends

BLAS and the main LAPACK drivers go through one module, not a `#[cfg]` pair per
call site:

- `base::blas_dispatch` -- dgemm (+tn/nt/tt), dgemv (+trans), ddot, dnrm2, dasum,
  idamax, dlange, zgemm (+tn/nt/tt), dger, zger, dsyrk, zherk, daxpy, zaxpy,
  zscal, dznrm2, zgemv (+trans), zsyrk.
- `base::lapack_dispatch` -- dgesv, dgetrf, dgetri, dgeev (+ left vectors, and
  eigenvalues only), dpotrf, dsyev, dgesvd, dggev, zggev.

Both mirror `algebra_linear`'s signatures (`usize` lengths, no workspace argument,
`Result<(), LapackError>`), so a call site reads the same in every configuration.
Add an entry there, not a branch in the caller. `intel_mkl` wins when both
features are on; there is no `compile_error!` for the pair, so `--all-features`
is a valid build and exercises the MKL path.

**A routine stays on the old per-file path when it has no twin to mirror or no
measured gain.** That is a decision per routine, documented where it is made, not
a TODO: the complex routines and `small_mat_svd` (a stack array, which a LAPACK
workspace defeats); and on the BLAS side `dcopy`/`zcopy` (one `memmove`), `dscal` and `zlange` (measured slower
through the library at the sizes that matter). Those keep this shape:

```rust
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
{
    // the C call, with a comment naming the shim's symbol
}
#[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
// the native call, which uses usize lengths and no workspace argument
```

Before adding to either path:

1. **A new shim entry goes in both `c_code/interface.h` and the `.c`.** The
   header is what `build.rs` tracks now, so a wrapper added to the source file
   alone can be silently not rebuilt.
2. **The `extern "C"` declaration must be gated with the same condition** -- in
   the files that still hold one. If it is not, a native-only build fails to
   link. If it *is* but the call site is not dispatched, the build fails to
   compile, so migrate the dispatch **first** and gate the declaration second,
   one file at a time. The native arm calls `algebra_linear`.
3. **Routines without a native implementation return
   `LapackError::NotImplemented`** rather than being left to fail at link time. A
   runtime error can name the feature to enable; a link error about a mangled
   symbol cannot.
4. **A native driver returns `Result<(), LapackError>`, never LAPACK's `INFO`.**
   `INFO < 0` and `INFO > 0` both become an error at the driver boundary, so a
   caller never has to remember which integer means what. Internal helpers
   (`dlahqr`, `dhgeqz`, `dsteqr`, `dbdsqr`, `dtrevc3`, ...) keep returning `INFO`
   because that is what the reference source says and what a diff against it
   should show.
5. **The function-scope `lda`/`ldu`/`lwork`/`work`/`info` declarations and their
   `#[cfg_attr(... allow(unused_variables, ...))]` belong only to the files still
   on the per-file path.** The nine dispatched drivers keep their scratch arrays
   inside `lapack_dispatch`, which sizes the workspace from LAPACK's
   `lwork = -1` query instead of the hard-coded minimum two call sites used.


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
`zscripts/debian-compile-*.bash` (into `/usr/local`, with `RUSSELL_LOCAL_PREFIX`
set accordingly), runs `cargo test --features local_umfpack,mumps`, and on nightly
enforces the coverage threshold shown above.
