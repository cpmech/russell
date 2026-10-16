//! Native Rust BLAS and LAPACK.
//!
//! This is the default backend: no external library, nothing to link, and it
//! builds for any target including `wasm32-unknown-unknown`. The `oblas` and
//! `mkl` features exist to swap these routines for a tuned library, not to make
//! the crate work in the first place.
//!
//! # Layout and conventions
//!
//! * Matrices are column-major, matching `crate::Matrix`'s storage, so the
//!   routines here index `data[i + j * nrow]` directly.
//! * Indices are 0-based and lengths are `usize`, unlike the Fortran the routines
//!   are derived from (`BLAS`/`LAPACK` are 1-based and take workspace arguments
//!   that are an artefact of being written in Fortran, not part of the maths).
//! * Nothing takes a workspace argument. The point of a Rust implementation is
//!   that workspace can be allocated internally.
//!
//! # Provenance and verification
//!
//! Each routine names the LAPACK 3.12.1 source it follows, so it can be checked
//! against the reference. Correctness is tested for parity against the reference
//! LAPACK where the environment provides it, and against closed-form results
//! otherwise; see the tests in each module.

pub mod blas;
pub mod lapack;
