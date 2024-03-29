use crate::{complex_vec_copy, to_i32, StrError};
use crate::{CcBool, ComplexVector, Stopwatch};
use crate::{ERROR_ALREADY_INITIALIZED, ERROR_NEED_INITIALIZATION, ERROR_NULL_POINTER, SUCCESSFUL_EXIT};
use num_complex::Complex64;

/// Opaque struct holding a C-pointer to InterfaceFFTW
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceFFTW {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceFFTW {}

// Enforce Send on the Rust structure
//
// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for FFTw1d {}

extern "C" {
    // typedef double fftw_complex[2]; // compatible with `*mut Complex64`
    fn interface_fftw_new() -> *mut InterfaceFFTW;
    fn interface_fftw_drop(interface: *mut InterfaceFFTW);
    fn interface_fftw_initialize(
        interface: *mut InterfaceFFTW,
        num_point: i32,
        data: *mut Complex64,
        inverse: CcBool,
        measure: CcBool,
    ) -> i32;
    fn interface_fftw_execute(interface: *mut InterfaceFFTW) -> i32;
}

/// Wraps FFTW to compute discrete Fourier transforms in 1D
///
/// Computes the forward transform:
///
/// ```text
///        N-1       -i n 2 π k / N       
/// U[n] =  Σ  u[k] e                     
///        k=0
///           __
/// with i = √-1    and   n = 0, ···, N-1
/// ```
///
/// Or the inverse transform:
///
/// ```text
///        N-1       i n 2 π k / N
/// V[n] =  Σ  v[k] e                     
///        k=0
///
/// thus u[n] = V[n] / N
/// ```
///
/// Kreyszig's book, page 531:
///
/// "The FFT is a computational method for the DFT that needs only `O(N)log₂(N)` operations instead of `O(N²)`.
pub struct FFTw1d {
    /// Holds a pointer to the C interface to FFTW
    interface: *mut InterfaceFFTW,

    /// Indicates whether FFTW has been initialized or not (just once)
    initialized: bool,

    /// Holds the length of the array given to initialize
    initialized_dim: usize,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on execute in nanoseconds
    time_execute_ns: u128,
}

impl Drop for FFTw1d {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            interface_fftw_drop(self.interface);
        }
    }
}

impl FFTw1d {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let interface = interface_fftw_new();
            if interface.is_null() {
                return Err("c-code failed to allocated the FFTW interface");
            }
            Ok(FFTw1d {
                interface,
                initialized: false,
                initialized_dim: 0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_execute_ns: 0,
            })
        }
    }

    /// Performs the fast Fourier transform
    ///
    /// **Warning:** The vector dimension must remain the same during subsequent calls to `execute`.
    ///
    /// Computes the (forward) transform:
    ///
    /// ```text
    ///        N-1       -i n 2 π k / N       
    /// U[n] =  Σ  u[k] e                     
    ///        k=0
    ///           __
    /// with i = √-1    and   n = 0, ···, N-1
    /// ```
    ///
    /// Or the inverse transform:
    ///
    /// ```text
    ///        N-1       i n 2 π k / N
    /// V[n] =  Σ  v[k] e                     
    ///        k=0
    ///
    /// thus u[n] = V[n] / N
    /// ```
    ///
    /// # Output
    ///
    /// `uu` -- Either the (forward) transform `U` or the inverse transform `V`
    ///
    /// # Input
    ///
    /// `u` -- The input vector with dimension `N`
    /// `inverse` -- Requests the inverse transform; otherwise the forward transform is computed
    /// `measure` -- (slower initialization) use the `FFTW_MEASURE` flag for better optimization analysis
    ///
    /// **Note:** Both transforms are non-normalized; thus the user may have to
    /// multiply the results by `(1/N)` if computing inverse transforms.
    pub fn execute(
        &mut self,
        uu: &mut ComplexVector,
        u: &ComplexVector,
        inverse: bool,
        measure: bool,
    ) -> Result<(), StrError> {
        // check
        let dim = u.dim();
        if uu.dim() != dim {
            return Err("u and uu arrays must have the same dimension");
        }
        if self.initialized {
            if dim != self.initialized_dim {
                return Err("subsequent calls to 'execute' must use vectors with the same dim as the first call");
            }
        } else {
            self.initialized_dim = dim;
        }

        // handle options
        let c_inverse = if inverse { 1 } else { 0 };
        let c_measure = if measure { 1 } else { 0 };

        // call initialize just once
        if !self.initialized {
            self.stopwatch.reset();
            complex_vec_copy(uu, u).unwrap();
            let num_point = to_i32(dim);
            unsafe {
                let status = interface_fftw_initialize(
                    self.interface,
                    num_point,
                    uu.as_mut_data().as_mut_ptr(),
                    c_inverse,
                    c_measure,
                );
                if status != SUCCESSFUL_EXIT {
                    return Err(handle_fftw_error(status));
                }
            }
            self.time_initialize_ns = self.stopwatch.stop();
            self.initialized = true;
        }

        // call execute
        self.stopwatch.reset();
        unsafe {
            let status = interface_fftw_execute(self.interface);
            if status != SUCCESSFUL_EXIT {
                return Err(handle_fftw_error(status));
            }
        }
        self.time_execute_ns = self.stopwatch.stop();
        Ok(())
    }

    /// Returns the nanoseconds spent on initialize
    pub fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    /// Returns the nanoseconds spent on execute
    pub fn get_ns_exec(&self) -> u128 {
        self.time_execute_ns
    }
}

/// Handles the status originating from the C-code
fn handle_fftw_error(status: i32) -> StrError {
    match status {
        ERROR_NULL_POINTER => "FFTW failed due to NULL POINTER error",
        ERROR_NEED_INITIALIZATION => "FFTW failed because INITIALIZATION is needed",
        ERROR_ALREADY_INITIALIZED => "FFTW failed because INITIALIZATION has been completed already",
        _ => "Error: unknown error returned by c-code (FFTW)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::FFTw1d;
    use crate::{complex_vec_approx_eq, cpx, math::PI, ComplexVector};
    use num_complex::Complex64;

    /// Uses Euler's formula to compute exp(-i⋅x) = cos(x) - i⋅sin(x)
    ///
    /// "mix" -- "minus i times x"
    fn exp_mix(x: f64) -> Complex64 {
        cpx!(f64::cos(x), -f64::sin(x))
    }

    /// Computes (naively) the discrete Fourier transform of u (very slow / for testing only)
    ///
    /// ```text
    ///      N-1     -i n xₖ                 2 π k
    /// Uₙ =  Σ  uₖ e          where    xₖ = —————
    ///      k=0                               N
    ///      __
    /// i = √-1   and   n = 0, ···, N-1
    /// ```
    ///
    /// See Equation (18) on page 530 of Kreyszig's book
    fn naive_dft1d(u: &ComplexVector) -> ComplexVector {
        let nn = u.dim();
        let mut uu = ComplexVector::new(nn);
        if nn < 1 {
            return uu;
        }
        let den = nn as f64;
        for n in 0..nn {
            for k in 0..nn {
                let xk = 2.0 * PI * ((n * k) as f64) / den;
                uu[n] += u[k] * exp_mix(xk);
            }
        }
        uu
    }

    #[test]
    fn execute_works() {
        // Kreyszig Example 4 on Page 530
        let mut fft = FFTw1d::new().unwrap();
        let u = ComplexVector::from(&[cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(4.0, 0.0), cpx!(9.0, 0.0)]);
        let mut uu = ComplexVector::new(u.dim());
        fft.execute(&mut uu, &u, false, false).unwrap();
        // println!("uu =\n{}", uu);
        let uu_correct = &[cpx!(14.0, 0.0), cpx!(-4.0, 8.0), cpx!(-6.0, 0.0), cpx!(-4.0, -8.0)];
        complex_vec_approx_eq(uu.as_data(), uu_correct, 1e-15);
        let uu_naive = naive_dft1d(&u);
        // println!("uu_naive =\n{}", uu_naive);
        complex_vec_approx_eq(uu_naive.as_data(), uu_correct, 1e-14);
    }
}
