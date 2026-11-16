//! Dispatch of every MUMPS FFI call onto one dedicated worker thread

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Mutex, OnceLock, mpsc};
use std::thread;

/// A raw pointer that may cross into the MUMPS worker thread
///
/// This is sound because [run_on_mumps_thread] dispatches one job at a time and the caller
/// blocks until the worker has finished it: the pointed-to data is never used concurrently,
/// stays alive across the call, and is only touched by the worker while the caller waits.
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Returns the wrapped pointer
    ///
    /// Calling this method (rather than reading the field) makes a closure capture the whole
    /// wrapper, so the closure inherits [Send] instead of capturing the bare raw pointer.
    pub(crate) fn get(&self) -> *mut T {
        self.0
    }
}

/// One erased unit of work for the MUMPS worker
///
/// `state` points to the caller's stack frame, which stays alive because the caller blocks on
/// its completion channel until [trampoline] has run.
struct Job {
    state: *mut (),
    run: unsafe fn(*mut ()),
}

unsafe impl Send for Job {}

/// The caller-owned state that a [Job] refers to
struct State<F, R> {
    f: Option<F>,
    outcome: Option<thread::Result<R>>,
    done: mpsc::Sender<()>,
}

/// Sender side of the process-wide MUMPS worker queue
static WORKER: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();

/// Returns the sender of the lazily spawned MUMPS worker, creating the worker on first use
fn worker_sender() -> &'static Mutex<mpsc::Sender<Job>> {
    WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        thread::Builder::new()
            .name("mumps-worker".into())
            .spawn(move || {
                // One job at a time: this loop is what serializes every MUMPS call.
                while let Ok(job) = rx.recv() {
                    unsafe {
                        (job.run)(job.state);
                    }
                }
            })
            .expect("failed to spawn the mumps-worker thread");
        Mutex::new(tx)
    })
}

/// Runs `f` on the process-wide MUMPS worker thread and returns its result
///
/// The call blocks until the closure returns. All MUMPS FFI calls go through this function, so
/// the worker runs one closure at a time; that both serializes the calls and keeps MUMPS's
/// OpenMP teams bound to a single OS master thread that never exits mid-run.
///
/// The function is not reentrant: a closure must not call it again. MUMPS calls never do, so a
/// nested call would simply queue behind the running one and deadlock.
///
/// A panic in the closure is caught by the worker and resumed on the caller, while the worker
/// thread survives and stays available for later calls.
///
/// Because the caller blocks until the closure returns, the closure may borrow the caller's
/// stack and `&mut` fields: those borrows are still valid for the whole execution.
pub(crate) fn run_on_mumps_thread<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let mut state = State {
        f: Some(f),
        outcome: None,
        done: done_tx,
    };
    let job = Job {
        state: (&mut state as *mut State<F, R>).cast::<()>(),
        run: trampoline::<F, R>,
    };

    {
        let sender = worker_sender();
        let guard = sender.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .send(job)
            .unwrap_or_else(|_| panic!("the mumps-worker thread is gone; failed to submit a MUMPS job"));
        // drop the sender guard before waiting, so another caller can enqueue its job
    }

    done_rx
        .recv()
        .unwrap_or_else(|_| panic!("the mumps-worker thread died before completing a MUMPS job"));

    match state.outcome.take().expect("the mumps-worker thread left no outcome") {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    }
}

/// Runs one state's closure, captures its result (or panic), then signals the caller
unsafe fn trampoline<F, R>(state: *mut ())
where
    F: FnOnce() -> R,
    R: Send,
{
    let state = unsafe { &mut *state.cast::<State<F, R>>() };
    let f = state.f.take().expect("the MUMPS job state was already consumed");
    state.outcome = Some(catch_unwind(AssertUnwindSafe(f)));
    let _ = state.done.send(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn returns_the_value_from_the_mumps_worker_thread() {
        let (value, thread_name) = run_on_mumps_thread(|| (1234, thread::current().name().map(str::to_string)));
        assert_eq!(value, 1234);
        assert_eq!(thread_name.as_deref(), Some("mumps-worker"));
    }

    #[test]
    fn resumes_a_panic_on_the_caller_and_keeps_the_worker_alive() {
        let payload = std::panic::catch_unwind(|| run_on_mumps_thread(|| -> () { panic!("boom") }))
            .expect_err("the worker panic was not resumed on the caller");
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()));
        assert_eq!(message, Some("boom"));

        // the worker survived the panicking job
        assert_eq!(run_on_mumps_thread(|| 7), 7);
        let name = run_on_mumps_thread(|| thread::current().name().map(str::to_string));
        assert_eq!(name.as_deref(), Some("mumps-worker"));
    }

    #[test]
    fn never_runs_two_calls_at_the_same_time() {
        const THREADS: usize = 8;
        const CALLS: usize = 50;
        let in_flight = AtomicUsize::new(0);
        let max_in_flight = AtomicUsize::new(0);
        let results = AtomicUsize::new(0);

        thread::scope(|scope| {
            for t in 0..THREADS {
                let in_flight = &in_flight;
                let max_in_flight = &max_in_flight;
                let results = &results;
                scope.spawn(move || {
                    for k in 0..CALLS {
                        let arg = t * CALLS + k;
                        let returned = run_on_mumps_thread(|| {
                            let now = in_flight.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                            max_in_flight.fetch_max(now, AtomicOrdering::SeqCst);
                            // widen the window in which a broken dispatcher would overlap
                            thread::yield_now();
                            let before = in_flight.fetch_sub(1, AtomicOrdering::SeqCst);
                            assert_eq!(before, 1, "two MUMPS jobs ran at the same time");
                            arg
                        });
                        assert_eq!(returned, arg, "a result was crossed between callers");
                        results.fetch_add(1, AtomicOrdering::SeqCst);
                    }
                });
            }
        });

        assert_eq!(in_flight.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(max_in_flight.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(results.load(AtomicOrdering::SeqCst), THREADS * CALLS);
    }

    #[test]
    fn borrows_caller_stack_data_until_the_closure_returns() {
        struct Accumulator {
            total: f64,
        }

        let values = [1.0, 2.5, 3.5];
        let mut acc = Accumulator { total: 0.5 };
        let result = run_on_mumps_thread(|| {
            let sum: f64 = values.iter().sum();
            acc.total += sum;
            acc.total * 2.0
        });
        assert_eq!(acc.total, 7.5);
        assert_eq!(result, 15.0);
    }
}
