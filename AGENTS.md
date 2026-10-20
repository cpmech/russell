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

### The `dev_lapack` feature (development only)

`--features dev_lapack` compiles the shim against a **locally built** LAPACK and links
it, so the native routines can be tested for numerical parity against the
reference. This is the only practicable way to check an eigenvalue or an SVD.

It requires `RUSSELL_DEV_LAPACK_ROOT` pointing at a CMake build directory of LAPACK
3.12.1 with `include/` and `lib/` holding `lapack.h`, `cblas.h`, `liblapack.so`
and `libcblas.so`. There is **no default**, deliberately: hardcoding one would put
a developer's directory layout into released source. `build.rs` checks for the
files and fails with an explanation rather than a link error.

```bash
RUSSELL_DEV_LAPACK_ROOT=/path/to/lapack/build cargo test -p russell_lab --features dev_lapack
```

Anything the reference build links is a reference for the *tests*; the crate's own symbols
come from the native or accelerator path.

Extra features:

- `local_umfpack` and `mumps` on `russell_sparse`, `russell_ode`, `russell_pde`,
  `russell_nonlin` — use the locally compiled SuiteSparse and MUMPS installed by
  `zscripts/*-compile-suitesparse.bash` and `zscripts/*-compile-mumps.bash`. Both
  read `RUSSELL_LOCAL_PREFIX` (`.../include/umfpack`, `.../lib/umfpack`,
  `.../include/mumps`, `.../lib/mumps`) and fail the build if it is unset or wrong.
  Without `local_umfpack` the UMFPACK comes from the package manager, found with
  pkg-config; UMFPACK is required in every configuration, `mumps` is optional.

## Current state of the native backend

Work in progress on the `native-lapack` branch. Read this before adding to it.

### Measured state

```bash
cargo test -p russell_lab --lib                     # 1258 pass, 0 fail, 0 ignored
RUSSELL_DEV_LAPACK_ROOT=/home/dorival/01-Code/fortran/lapack-3.12.1/build-gfortran \
  cargo test -p russell_lab --lib --features dev_lapack # 1271 pass, 0 fail
```

Both configurations are known-good on the current tree of `native-lapack`, with
**zero** failures. The `dev_lapack` run adds the parity tests and is otherwise the same
native path.

The `dev_lapack` feature needs `RUSSELL_DEV_LAPACK_ROOT` set to a valid CMake build
directory. The LAPACK *source* tree is not one -- it has no `include/` or `lib/`.
For the tree on this machine that is
`/home/dorival/01-Code/fortran/lapack-3.12.1/build-gfortran`.

The default cargo target directory is not writable here; prefix commands with
`CARGO_TARGET_DIR=/home/dorival/Documents/deepseek-harness/default-workspace/dsh-target`.

Every LAPACK routine the crate's public API needs is implemented natively:
`dgeev` (with balancing, and left eigenvectors), `dsyev`, `dgesvd`, `dggev`
(with left eigenvectors), and the complex `zgetrf`/`zgetri`/`zgesv`/`zpotrf`,
`zheev`, `zgeev`, `zgesvd`, `zggev`. `mat_pseudo_inverse` works through
`dgesvd`. There are no `not_implemented` guards left on these paths.

Two things that were *not* the whole story, recorded so they are not chased
again:

* The original 61 failures were not all `NotImplemented`. `mat_eigenvalues`,
  `mat_eigen`, `mat_eigen_sym`, `mat_gen_eigen` and `mat_svd` declared their
  `extern "C"` call behind `#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]`
  with no `#[cfg(not(...))]` counterpart, so a native build fell through to
  `Ok(())` with the outputs still zero -- 26 tests returning **silently wrong
  results** rather than an error. Commit `4cc08ae1` added the missing branches.
* Some tests assert on an exact error *string*, so a native path that reports
  the same condition in different words still fails. Keep the wording identical
  to the shim's.

### Bugs found and fixed

Two long-lived bugs produced *silently wrong numbers* rather than errors, and
both were initially misdiagnosed:

* `zherk_rank` conjugated the **second** index in the `A^H A` case; BLAS's
  `zherk('C')` conjugates the **first**. The two give complex conjugates, so the
  shape and real parts were right and only the imaginary parts were wrong, which
  is why it survived inspection. Commit `dff1851e`.
* `dgeev` looked inaccurate on defectively repeated eigenvalues and was
  diagnosed as a `dlahqr` deflation bug -- wrongly. `dlahqr` is a faithful port;
  at the entry in question reference `dlahqr` does not deflate it either, and raw
  reference `dhseqr` on the unbalanced matrix is no more accurate. The cause was
  the missing `dgebal`. **When `dgeev` looks inaccurate, check `dgebal` before
  suspecting the QR iteration.**

The same mistake recurred in different clothes across the ports — **an index,
an extent, or a length taken from the wrong axis**:

- `norm_char` mapped the `Norm` discriminants *by position* rather than by
  constant, so every norm was wrong and `Norm::Max` returned the Frobenius norm.
  `c_code/interface_blas.c` is the source of truth for that mapping.
- `dsyrk_rank` contracted over `A`'s columns for both orientations, so
  `C := A^T A` was wrong for non-square `A`.
- `complex_mat_t_mat_mul` transposed `A` as if it were `(m, k)` when it is
  `(k, m)`.
- `zgebal` applied the row scaling as `a[t + k*n] *= ginv` over `t` -- column
  `K`, rows `i..n` -- instead of row `i`, columns `K..n`. The complex twin of
  `dgebal` then stopped being a similarity: the scaling loop diverged and
  `scale` came out as `2e-292` instead of a power of two. Found by running the
  complex routine over the real one's fixtures and comparing, which is the
  cheapest way to test a twin.
- `dlarfg`'s underflow branch scaled `X` and `ALPHA` *down* by `SAFMIN` and
  returned `BETA = -0` for a subnormal tail. The reference multiplies by
  `RSAFMN = 1/SAFMN` (`SAFMN = DLAMCH('S')/DLAMCH('E')` = 1e-292, not
  `DLAMCH('S')`) and takes `BETA` back at the end by `SAFMN**KNT`, one
  multiplication at a time because the power overflows.
- `solve_lin_sys` reported a different singular error string from the shim's.
- `mat_cholesky` with `upper = true` needed care twice: LAPACK's `'U'` *reads*
  only the upper triangle (so the other half may be zeros and must be mirrored),
  and *writes* only the upper triangle (so the rest must be left alone).
- `dlange`/`zlange` rejected `'O'`, which is LAPACK's other spelling of the
  one-norm and what the shim actually sends — a panic, not a wrong number.

### What is verified

`native::blas` and `native::lapack` are tested for parity against the reference
LAPACK through the `dev_lapack` feature (see above): `dgesv`, `dgetrf`+`dgetri`,
`dpotrf`, `dhseqr`'s eigenvalues, `dsyev`+`zheev`, `dgesvd`, `dggev`, and the
complex `zpotrf`/`zgetrf`+`zgetri`/`zgesv`, at 1e-10 relative. Extend those parity
tests as routines are added — they are what caught the `dpotrf` triangle
misunderstanding.

Where a result is not unique (eigenvectors, and the *order* of eigenvalues), the
tests check the defining equation or a canonical order instead of comparing
against the reference's output directly; where it is unique (singular values, an
inverse, a Cholesky factor), they compare element-wise, because that is what
catches a conjugation error, which an invariant-based check would miss.

LAPACK's own suite (`TESTING/LIN`, `TESTING/EIG`) tests the routines we ported, not
just the drivers, and its checks transfer directly: `dchkbl`/`dchkbk` for
`dgebal`/`dgebak`, `dhst01` for `dgehrd`, `dsyt21` for `dsytrd` + `dorgtr`,
`dstt21`/`dstt22` for `dsteqr`, `dbdt01`/`dbdt03` for `dgebd2`/`dorgbr`/`dbdsqr`.
Each is a residual divided by the norms and by `n * eps` --
`norm(A - Q H Q^T) / (norm(A) * n * eps)` -- with 20 as the pass threshold. Prefer
that form to an absolute tolerance: it is a different assertion at n=1 and at
n=20, and the measured values land between 0.1 and 12, so a wrong index (which
shows up as `1e14`) cannot hide under a hand-set `1e-12`.

Transcribe LAPACK's fixtures, but check them against the reference before trusting
them: the fifth example of `TESTING/dbal.in` still states `dgebal`'s older
permutation order (`(4,3,5)`, where 3.12.1 gives `(4,5,3)`), and `dchkbl` only
*reports* its largest error instead of asserting it, so the drift has never been
flagged -- its own run prints "example number having largest error = 5".

Run the `dev_lapack` parity tests in the same target directory as the default build.
Combining `--features dev_lapack` with an accelerator does link -- the reference
copy's entry points are prefixed (`dev_lapack_c_*`) so they do not collide -- but
the process then loads two BLAS implementations, which is a smoke test at best;
run `dev_lapack` alone.

### Hard constraints

- **Everything is column-major**, indexing `data[i + j * n]`, matching `Matrix`.
  **Never use `LAPACKE_*`** -- it is the row-major API and will silently
  transpose every matrix. `lapack.h` binds directly to the Fortran routines and
  is correctly column-major.
- **No developer paths in source** -- not in code, not in defaults, not in doc
  comments. Use environment variables, and prefer *requiring* one over supplying
  a default.
- **Migrate the dispatch before gating the `extern "C"` declaration**, one file
  at a time. Gating first turns every unmigrated call site into a compile error
  and leaves the tree unable to run tests.
- **Preserve the headline property:** `cargo check --workspace` must pass with no
  external BLAS present. Verify it after changes.
- Keep the tree rustfmt-clean.
- **American English everywhere** -- comments, documentation, identifiers, test
  names and commit messages. Use `behavior`, `normalize`, `factorization`,
  `center`, `modeling`, `toward`: the British variants (`-our` for `-or`, `-ise`
  for `-ize`, `-re` for `-er`, `-ce` for `-se`, doubled `l`) are not used. The
  workspace was cleaned once (139 occurrences in 26 files);
  `russell_lab/tests/american_english.rs` scans `.rs`/`.md`/`.py` and fails the
  suite on a regression, so a new form belongs in that file's list -- or better,
  not in the tree.
- **Do not change the behavior of the `open_blas` / `intel_mkl` accelerated paths.** They
  must stay consistent with the native ones, including error strings.

### Reporting

Report failures and uncertainty plainly. If you turn out to be wrong about
something you said earlier, say so. Do not claim a verification you did not run
-- a previous session claimed the `dev_lapack` build worked when it had not compiled
since a rename, and that had to be corrected.

Two lessons from this work are worth more than the rest, because both cost hours:

1. **Port the reference literally; do not re-derive an algorithm whose source is
   available.** The two attempts to hand-derive (`dgehrd`, `dlahqr`) both
   produced code that compiled, ran, looked orthogonal, and was wrong. Every
   success came from transcription.
2. **A test that cannot fail is worse than no test.** Four times in this
   work a test passed while checking nothing: `mat_eigenvalues` and friends
   returning zeros, `mat_eigen_lr`'s test never inspecting `u`, my `dggev`
   acceptance test reading back the matrix `dggev` had overwritten, and
   `dgeev_eigenvectors_satisfy_their_defining_equation` decomposing a zero
   matrix. When adding a test, check that a deliberate mistake makes it fail.

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
RUSSELL_DEV_LAPACK_ROOT=/path/to/lapack/build cargo test -p russell_lab --features dev_lapack

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
- `base::lapack_dispatch` -- dgesv, dpotrf, dsyev, dgesvd, dgeev.

Both mirror the `native::*` signatures (`usize` lengths, no workspace argument,
`Result<(), LapackError>`), so a call site reads the same in every configuration.
Add an entry there, not a branch in the caller. `intel_mkl` wins when both
features are on; there is no `compile_error!` for the pair, so `--all-features`
is a valid build and exercises the MKL path.

**A routine stays on the old per-file path when it has no twin to mirror or no
measured gain.** That is a decision per routine, documented where it is made, not
a TODO: `dgetrf`/`dgetri`, the complex routines, `dggev`/`zggev`, `dlange` and
`small_mat_svd` (a stack array, which a LAPACK workspace defeats); and on the
BLAS side `dcopy`/`zcopy` (one `memmove`), `dscal` and `zlange` (measured slower
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
   one file at a time.
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
   on the per-file path.** The five dispatched drivers keep their scratch arrays
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
