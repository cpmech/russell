#![allow(unused)]

use crate::CcBool;
use crate::StrError;
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
// unsafe impl Send for FourierTransform1D {}

extern "C" {
    fn interface_fftw_new() -> *mut InterfaceFFTW;
    fn interface_fftw_drop(interface: *mut InterfaceFFTW);
    fn interface_fftw_initialize(
        interface: *mut InterfaceFFTW,
        data: *mut Complex64,
        inverse: CcBool,
        measure: CcBool,
    ) -> i32;
    fn interface_fftw_execute(interface: *mut InterfaceFFTW) -> i32;
}

/// Implements a FFTW3 plan structure to compute direct or inverse 1D Fourier transforms
///
/// Computes the forward transform:
///
/// ```text
///        N-1         -i 2 π j k / N       
/// X[k] =  Σ  x[j] ⋅ e                     
///        j=0
///           __
/// with i = √-1
/// ```
///
/// Or the inverse transform:
///
/// ```text
///        N-1         +i 2 π j k / N
/// Y[k] =  Σ  y[j] ⋅ e                     
///        j=0
///
/// thus x[k] = Y[k] / N
/// ```
///
/// # Notes
///
/// * FFTW says "so you should initialize your input data after creating the plan."
///   Therefore, the plan can be created and reused several times.
///   <http://www.fftw.org/fftw3_doc/Planner-Flags.html>
///   Also: "The plan can be reused as many times as needed. In typical high-performance
///   applications, many transforms of the same size are computed"
///   <http://www.fftw.org/fftw3_doc/Introduction.html>
///
/// **Warning:** The size of `data` must remain the same between calls to `execute`
pub struct FourierTransform1d {
    interface: *mut InterfaceFFTW,
}

impl Drop for FourierTransform1d {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            interface_fftw_drop(self.interface);
        }
    }
}

impl FourierTransform1d {
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let interface = interface_fftw_new();
            if interface.is_null() {
                return Err("c-code failed to allocated the FFTW interface");
            }
            Ok(FourierTransform1d { interface })
        }
    }
}
