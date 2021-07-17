#[derive(Debug)]
pub enum Error {
    /// Represents all cases of `std::fmt::Error`.
    FmtError(std::fmt::Error),

    /// Represents all cases of `std::io::Error`.
    IoError(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Error::FmtError(ref err) => err.fmt(f),
            Error::IoError(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Error::FmtError(_) => None,
            Error::IoError(_) => None,
        }
    }
}

impl From<std::fmt::Error> for Error {
    fn from(err: std::fmt::Error) -> Error {
        Error::FmtError(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::IoError(err)
    }
}
