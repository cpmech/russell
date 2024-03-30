use crate::StrError;
use crate::{complex_vec_copy, to_i32, CcBool, ComplexMatrix, ComplexVector, Stopwatch};
use crate::{ERROR_ALREADY_INITIALIZED, ERROR_NEED_INITIALIZATION, ERROR_NULL_POINTER, SUCCESSFUL_EXIT};
use num_complex::Complex64;
use num_traits::Zero;

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
unsafe impl Send for FFTw {}

extern "C" {
    // typedef double fftw_complex[2]; // compatible with `*mut Complex64`
    fn interface_fftw_new() -> *mut InterfaceFFTW;
    fn interface_fftw_drop(interface: *mut InterfaceFFTW);
    fn interface_fftw_init_1d(
        interface: *mut InterfaceFFTW,
        n0: i32,
        data: *mut Complex64,
        inverse: CcBool,
        measure: CcBool,
    ) -> i32;
    fn interface_fftw_init_2d(
        interface: *mut InterfaceFFTW,
        n0: i32,
        n1: i32,
        data_row_major: *mut Complex64,
        inverse: CcBool,
        measure: CcBool,
    ) -> i32;
    fn interface_fftw_init_3d(
        interface: *mut InterfaceFFTW,
        n0: i32,
        n1: i32,
        n2: i32,
        data_row_major: *mut Complex64,
        inverse: CcBool,
        measure: CcBool,
    ) -> i32;
    fn interface_fftw_execute(interface: *mut InterfaceFFTW) -> i32;
}

/// Wraps FFTW to compute discrete Fourier transforms in 1D, 2D and 3D
///
/// WARNING: FFTW is not thread-safe
///
/// In 1D, computes the forward transform:
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
/// In 2D, computes the (forward) transform:
///
/// ```text
///            N1-1 N0-1           -i 2 π k1 l1 / N1  -i 2 π k0 l0 / N0
/// A[l0,l1] =   Σ    Σ  a[k0,k1] e                  e
///            k1=0 k0=0
/// ```
///
/// In 3D, computes the (forward) transform:
///
/// ```text
///                N2-1 N1-1 N0-1              -i 2 π k2 l2 / N2  -i 2 π k1 l1 / N1  -i 2 π k0 l0 / N0
/// S[l2][l0,l1] =   Σ    Σ    Σ  s[k2][k0,k1] e                  e                  e
///                k2=0 k1=0 k0=0
/// ```
///
/// In 3D, note that `l2` appears first in `S[l2][l0,l1]` because we employ an array of matrices to hold the data
///
/// Kreyszig's book, page 531:
///
/// "The FFT is a computational method for the DFT that needs only `O(N)log₂(N)` operations instead of `O(N²)`.
///
/// # Literature
///
/// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
///   Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
pub struct FFTw {
    /// Holds a pointer to the C interface to FFTW
    interface: *mut InterfaceFFTW,

    /// Indicates whether FFTW has been initialized or not (just once)
    initialized: bool,

    /// Holds the ranks of the data along each direction
    initialized_ranks: [usize; 3],

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on copying data (e.g., to convert col-major to row-major)
    time_copy_data_ns: u128,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on execute in nanoseconds
    time_execute_ns: u128,

    /// Data in row-major format (as required by FFTW) used by the multi-dimension calculations
    ///
    /// ```text
    ///     ┌                        ┐
    /// A = │ A00→B0  A01→B1  A02→B2 │
    ///     │ A10→B3  A11→B4  A12→B5 │
    ///     └                        ┘
    ///                     (n0,n1)=(2,3)
    ///
    ///     m = 0      1      2      3      4      5
    /// B = {  A00    A01    A02    A10    A11    A12 }
    ///       0+0⋅3  1+0⋅3  2+0⋅3  0+1⋅3  1+1⋅3  2+1⋅3
    ///
    /// m = k + n2 ⋅ (j + n1 ⋅ i)
    /// i = m / n1
    /// j = m % n1
    /// ```
    data_row_major: Vec<Complex64>,
}

impl Drop for FFTw {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            interface_fftw_drop(self.interface);
        }
    }
}

impl FFTw {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let interface = interface_fftw_new();
            if interface.is_null() {
                return Err("c-code failed to allocated the FFTW interface");
            }
            Ok(FFTw {
                interface,
                initialized: false,
                initialized_ranks: [0, 0, 0],
                stopwatch: Stopwatch::new(),
                time_copy_data_ns: 0,
                time_initialize_ns: 0,
                time_execute_ns: 0,
                data_row_major: Vec::new(),
            })
        }
    }

    /// Computes the discrete Fourier transform in 1D using the FFT method
    ///
    /// WARNING: FFTW is not thread-safe
    ///
    /// **Warning:** The vector dimension must remain the same during subsequent calls to `execute`.
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
    pub fn dft_1d(
        &mut self,
        uu: &mut ComplexVector,
        u: &ComplexVector,
        inverse: bool,
        measure: bool,
    ) -> Result<(), StrError> {
        // check
        let n0 = u.dim();
        if n0 < 2 {
            return Err("the vector length must be ≥ 2");
        }
        if uu.dim() != n0 {
            return Err("vectors must have the same lengths");
        }
        if self.initialized {
            if self.initialized_ranks[1] > 0 || self.initialized_ranks[2] > 0 {
                return Err("cannot compute a 1D DFT after a multi-dimensional DFT has been computed");
            }
            if n0 != self.initialized_ranks[0] {
                return Err("subsequent calls must use vectors with the same lengths as in the first call");
            }
        } else {
            self.initialized_ranks[0] = n0;
        }

        // call initialize just once
        if !self.initialized {
            let c_inverse = if inverse { 1 } else { 0 };
            let c_measure = if measure { 1 } else { 0 };
            self.stopwatch.reset();
            complex_vec_copy(uu, u).unwrap();
            unsafe {
                let status = interface_fftw_init_1d(
                    self.interface,
                    to_i32(n0),
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

    /// Computes the discrete Fourier transform in 2D using the FFT method
    ///
    /// WARNING: FFTW is not thread-safe
    ///
    /// **Warning:** The matrix dimension must remain the same during subsequent calls to `execute`.
    ///
    /// # Output
    ///
    /// `aa` -- Either the (forward) transform `A` or the inverse transform `B`
    ///
    /// # Input
    ///
    /// `a` -- The input matrix with dimension `(N0,N1)`
    /// `inverse` -- Requests the inverse transform; otherwise the forward transform is computed
    /// `measure` -- (slower initialization) use the `FFTW_MEASURE` flag for better optimization analysis
    ///
    /// **Note:** Both transforms are non-normalized; thus the user may have to
    /// multiply the results by `(1/N)` if computing inverse transforms.
    pub fn dft_2d(
        &mut self,
        aa: &mut ComplexMatrix,
        a: &ComplexMatrix,
        inverse: bool,
        measure: bool,
    ) -> Result<(), StrError> {
        // check
        let (n0, n1) = a.dims();
        if n0 < 2 || n1 < 2 {
            return Err("the matrix dimensions be ≥ 2 along each direction, i.e., at least (2, 2)");
        }
        if aa.dims() != (n0, n1) {
            return Err("matrices must have the same dimensions");
        }
        if self.initialized {
            if self.initialized_ranks[0] > 0 && self.initialized_ranks[1] == 0 {
                return Err("cannot compute a 2D DFT after a 1D DFT has been computed");
            }
            if self.initialized_ranks[2] > 0 {
                return Err("cannot compute a 2D DFT after a 3D DFT has been computed");
            }
            if n0 != self.initialized_ranks[0] || n1 != self.initialized_ranks[1] {
                return Err("subsequent calls must use matrices with the same dimensions as in the first call");
            }
        } else {
            self.initialized_ranks[0] = n0;
            self.initialized_ranks[1] = n1;
        }

        // call initialize just once
        if !self.initialized {
            // convert col-major to row-major
            self.stopwatch.reset();
            self.data_row_major.resize(n0 * n1, Complex64::zero());
            for i in 0..n0 {
                for j in 0..n1 {
                    self.data_row_major[j + n1 * i] = a.get(i, j);
                }
            }
            self.time_copy_data_ns = self.stopwatch.stop();

            // initialization
            let c_inverse = if inverse { 1 } else { 0 };
            let c_measure = if measure { 1 } else { 0 };
            self.stopwatch.reset();
            unsafe {
                let status = interface_fftw_init_2d(
                    self.interface,
                    to_i32(n0),
                    to_i32(n1),
                    self.data_row_major.as_mut_ptr(),
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

        // convert row-major to col-major
        for i in 0..n0 {
            for j in 0..n1 {
                aa.set(i, j, self.data_row_major[j + n1 * i]);
            }
        }
        Ok(())
    }

    /// Computes the discrete Fourier transform in 3D using the FFT method
    ///
    /// WARNING: FFTW is not thread-safe
    ///
    /// **Warning:** The dimensions must remain the same during subsequent calls to `execute`.
    ///
    /// # Output
    ///
    /// `ss` -- Either the (forward) transform or the inverse transform
    ///
    /// # Input
    ///
    /// `s` -- The input array of matrices with length `N2`; each matrix has the same dimension `(N0,N1)`
    /// `inverse` -- Requests the inverse transform; otherwise the forward transform is computed
    /// `measure` -- (slower initialization) use the `FFTW_MEASURE` flag for better optimization analysis
    ///
    /// **Note:** Both transforms are non-normalized; thus the user may have to
    /// multiply the results by `(1/N)` if computing inverse transforms.
    pub fn dft_3d(
        &mut self,
        ss: &mut Vec<ComplexMatrix>,
        s: &Vec<ComplexMatrix>,
        inverse: bool,
        measure: bool,
    ) -> Result<(), StrError> {
        // check
        let n2 = s.len();
        if n2 < 1 {
            return Err("the length of the array must be ≥ 1");
        }
        if ss.len() != n2 {
            return Err("the arrays must have the same lengths");
        }
        let (n0, n1) = s[0].dims();
        if n0 < 1 || n1 < 1 {
            return Err("the matrix dimensions be ≥ 1 along each direction, i.e., at least (1, 1)");
        }
        for p in 0..n2 {
            if s[p].dims() != (n0, n1) || ss[p].dims() != (n0, n1) {
                return Err("matrices must have the same dimensions");
            }
        }
        if self.initialized {
            if self.initialized_ranks[0] > 0 && self.initialized_ranks[1] == 0 {
                return Err("cannot compute a 3D DFT after a 1D DFT has been computed");
            }
            if self.initialized_ranks[1] > 0 && self.initialized_ranks[2] == 0 {
                return Err("cannot compute a 3D DFT after a 2D DFT has been computed");
            }
            if n0 != self.initialized_ranks[0] || n1 != self.initialized_ranks[1] || n2 != self.initialized_ranks[2] {
                return Err("subsequent calls must use data with the same dimensions as in the first call");
            }
        } else {
            self.initialized_ranks[0] = n0;
            self.initialized_ranks[1] = n1;
            self.initialized_ranks[2] = n2;
        }

        // call initialize just once
        if !self.initialized {
            // convert input to row-major
            self.stopwatch.reset();
            self.data_row_major.resize(n0 * n1 * n2, Complex64::zero());
            for i in 0..n0 {
                for j in 0..n1 {
                    for k in 0..n2 {
                        self.data_row_major[k + n2 * (j + n1 * i)] = s[k].get(i, j);
                    }
                }
            }
            self.time_copy_data_ns = self.stopwatch.stop();

            // initialization
            let c_inverse = if inverse { 1 } else { 0 };
            let c_measure = if measure { 1 } else { 0 };
            self.stopwatch.reset();
            unsafe {
                let status = interface_fftw_init_3d(
                    self.interface,
                    to_i32(n0),
                    to_i32(n1),
                    to_i32(n2),
                    self.data_row_major.as_mut_ptr(),
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

        // extract row-major data
        for i in 0..n0 {
            for j in 0..n1 {
                for k in 0..n2 {
                    ss[k].set(i, j, self.data_row_major[k + n2 * (j + n1 * i)]);
                }
            }
        }
        Ok(())
    }

    /// Returns the nanoseconds spent on copying data (e.g., to convert col-major to row-major)
    pub fn get_ns_copy(&self) -> u128 {
        self.time_copy_data_ns
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
    use super::FFTw;
    use crate::{complex_vec_approx_eq, cpx, math::PI, ComplexMatrix, ComplexVector};
    use num_complex::Complex64;

    /// Uses Euler's formula to compute exp(-i⋅x) = cos(x) - i⋅sin(x)
    ///
    /// "mix" -- "minus i times x"
    fn exp_mix(x: f64) -> Complex64 {
        cpx!(f64::cos(x), -f64::sin(x))
    }

    /// Computes (naively) the discrete Fourier transform in 1D (very slow / for testing only)
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
    fn naive_dft_1d(u: &ComplexVector) -> ComplexVector {
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

    /// Computes (naively) the discrete Fourier transform in 2D (very slow / for testing only)
    fn naive_dft_2d(a: &ComplexMatrix) -> ComplexMatrix {
        let (n0, n1) = a.dims();
        let mut aa = ComplexMatrix::new(n0, n1);
        if n0 < 1 || n1 < 1 {
            return aa;
        }
        let den0 = n0 as f64;
        let den1 = n1 as f64;
        for l0 in 0..n0 {
            for l1 in 0..n1 {
                for k0 in 0..n0 {
                    for k1 in 0..n1 {
                        let xk = 2.0 * PI * ((l0 * k0) as f64) / den0;
                        let yk = 2.0 * PI * ((l1 * k1) as f64) / den1;
                        aa.add(l0, l1, a.get(k0, k1) * exp_mix(xk) * exp_mix(yk));
                    }
                }
            }
        }
        aa
    }

    /// Computes (naively) the discrete Fourier transform in 3D (very slow / for testing only)
    fn naive_dft_3d(s: &Vec<ComplexMatrix>) -> Vec<ComplexMatrix> {
        let n2 = s.len();
        if n2 < 1 {
            return Vec::new();
        }
        let (n0, n1) = s[0].dims();
        let mut ss = vec![ComplexMatrix::new(n0, n1); n2];
        if n0 < 1 || n1 < 1 {
            return ss;
        }
        let den0 = n0 as f64;
        let den1 = n1 as f64;
        let den2 = n2 as f64;
        for l0 in 0..n0 {
            for l1 in 0..n1 {
                for l2 in 0..n2 {
                    for k0 in 0..n0 {
                        for k1 in 0..n1 {
                            for k2 in 0..n2 {
                                let xk = 2.0 * PI * ((l0 * k0) as f64) / den0;
                                let yk = 2.0 * PI * ((l1 * k1) as f64) / den1;
                                let zk = 2.0 * PI * ((l2 * k2) as f64) / den2;
                                ss[l2].add(l0, l1, s[k2].get(k0, k1) * exp_mix(xk) * exp_mix(yk) * exp_mix(zk));
                            }
                        }
                    }
                }
            }
        }
        ss
    }

    #[test]
    fn dft_1d_works() {
        // Kreyszig Example 4 on Page 530
        let mut fft = FFTw::new().unwrap();
        let u = ComplexVector::from(&[cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(4.0, 0.0), cpx!(9.0, 0.0)]);
        let mut uu = ComplexVector::new(u.dim());
        fft.dft_1d(&mut uu, &u, false, false).unwrap();
        println!("uu =\n{}", uu);
        let uu_correct = &[cpx!(14.0, 0.0), cpx!(-4.0, 8.0), cpx!(-6.0, 0.0), cpx!(-4.0, -8.0)];
        complex_vec_approx_eq(uu.as_data(), uu_correct, 1e-15);
        let uu_naive = naive_dft_1d(&u);
        println!("uu_naive =\n{}", uu_naive);
        complex_vec_approx_eq(uu_naive.as_data(), uu_correct, 1e-14);
    }

    #[test]
    fn dft_2d_works() {
        let (m, n) = (2, 4);
        let mut a = ComplexMatrix::new(m, n);
        let mut k = 0;
        for i in 0..m {
            for j in 0..n {
                a.set(i, j, cpx!(k as f64, (k + 1) as f64));
                k += 2;
            }
        }
        println!("a =\n{}", a);

        // compute DFT
        let mut fft = FFTw::new().unwrap();
        let mut aa = ComplexMatrix::new(m, n);
        fft.dft_2d(&mut aa, &a, false, false).unwrap();
        println!("aa =\n{}", aa);

        // compare with "naive" computation
        let aa_naive = naive_dft_2d(&a);
        println!("aa_naive =\n{:.1}", aa_naive);
        complex_vec_approx_eq(aa.as_data(), aa_naive.as_data(), 1e-13);
    }

    #[test]
    fn dft_3d_works() {
        let (m, n, ns) = (2, 4, 2);
        let mut s = vec![ComplexMatrix::new(m, n); ns];
        let mut k = 0;
        for p in 0..ns {
            for i in 0..m {
                for j in 0..n {
                    s[p].set(i, j, cpx!(k as f64, (k + 1) as f64));
                    k += 2;
                }
            }
            println!("s[{}] =\n{}", p, s[p]);
        }

        // compute DFT
        let mut fft = FFTw::new().unwrap();
        let mut ss = vec![ComplexMatrix::new(m, n); ns];
        fft.dft_3d(&mut ss, &s, false, false).unwrap();
        for p in 0..ns {
            println!("ss[{}] =\n{}", p, ss[p]);
        }

        // compare with "naive" computation
        let ss_naive = naive_dft_3d(&s);
        for p in 0..ns {
            println!("ss_naive[{}] =\n{:.1}", p, ss_naive[p]);
            complex_vec_approx_eq(ss[p].as_data(), ss_naive[p].as_data(), 1e-13);
        }
    }
}
