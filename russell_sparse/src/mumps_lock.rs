//! Process-wide serialization of the MUMPS FFI calls

use std::sync::{Mutex, MutexGuard};

/// Process-wide mutex shared by the real and the complex MUMPS solvers
///
/// `dmumps_c` and `zmumps_c` link against the same MUMPS common library and the same OpenMP
/// runtime, so one mutex must cover both. MUMPS keeps process-global state and tears down its
/// OpenMP worker team when it finalizes, hence every call that enters the library is serialized.
static MUMPS_LOCK: Mutex<()> = Mutex::new(());

/// Waits for, and locks, the process-wide MUMPS mutex
///
/// A poisoned mutex is recovered rather than propagated: a panic in one thread must not disable
/// every later MUMPS call (including `Drop`) for the rest of the process.
pub(crate) fn lock_mumps() -> MutexGuard<'static, ()> {
    MUMPS_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
