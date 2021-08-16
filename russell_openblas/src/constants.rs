pub const CBLAS_COL_MAJOR: i32 = 102;
pub const CBLAS_NO_TRANS: i32 = 111;
pub const CBLAS_TRANS: i32 = 112;

pub fn cblas_transpose(transpose: bool) -> i32 {
    if transpose {
        return CBLAS_TRANS;
    }
    CBLAS_NO_TRANS
}
