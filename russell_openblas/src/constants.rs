// from /usr/include/x86_64-linux-gnu/cblas.h
// from /usr/include/lapacke.h

pub(crate) const LAPACK_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;
pub(crate) const CBLAS_UPPER: i32 = 121;
pub(crate) const CBLAS_LOWER: i32 = 122;

#[inline]
pub(crate) fn cblas_transpose(transpose: bool) -> i32 {
    if transpose {
        return CBLAS_TRANS;
    }
    CBLAS_NO_TRANS
}

#[inline]
pub(crate) fn cblas_uplo(up: bool) -> i32 {
    if up {
        return CBLAS_UPPER;
    }
    CBLAS_LOWER
}

#[inline]
pub(crate) fn lapack_uplo(up: bool) -> u8 {
    if up {
        return b'U';
    }
    b'L'
}

#[inline]
pub(crate) fn lapack_job_vlr(calculate: bool) -> u8 {
    if calculate {
        return b'V';
    }
    b'N'
}
