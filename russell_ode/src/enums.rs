/// Holds information about the numerical method to solve (approximate) ODEs
#[derive(Clone, Copy, Debug)]
pub struct Information {
    pub order: usize,
    pub order_of_estimator: usize, // 0 means no error estimator available
    pub implicit: bool,
    pub embedded: bool,
    pub multiple_stages: bool,
    pub first_step_same_as_last: bool,
}

/// Specifies the numerical method to solve (approximate) ODEs
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics ISSN 0179-3632, 528p
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    /// Radau method (Radau IIA) (implicit, order 5, embedded)
    Radau5,

    /// Backward Euler method (implicit, order 1)
    BwEuler,

    /// Forward Euler method (explicit, order 1)
    FwEuler,

    /// Runge (Kutta) method (mid-point) (explicit, order 2)
    ///
    /// Reference: page 135 of Hairer, Nørsett, and Wanner (2008)
    Rk2,

    /// Runge (Kutta) method (explicit, order 3)
    ///
    /// Reference: page 135 of Hairer, Nørsett, and Wanner (2008)
    Rk3,

    /// Heun method (explicit, order 3)
    ///
    /// Reference: page 135 of Hairer, Nørsett, and Wanner (2008)
    Heun3,

    /// "The" Runge-Kutta method (explicit, order 4)
    ///
    /// Reference: page 138 of Hairer, Nørsett, and Wanner (2008)
    Rk4,

    /// Runge-Kutta method (alternative) (explicit, order 4, 3/8-Rule)
    ///
    /// Reference: page 138 of Hairer, Nørsett, and Wanner (2008)
    Rk4alt,

    /// Modified Euler method (explicit, order 2(1), embedded)
    MdEuler,

    /// Merson method (explicit, order 4("5"), embedded)
    ///
    /// "5" means that the order 5 is for linear equations with constant coefficients;
    /// otherwise the method is of order3.
    ///
    /// Reference: page 167 of Hairer, Nørsett, and Wanner (2008)
    Merson4,

    /// Zonneveld method (explicit, order 4(3), embedded)
    ///
    /// Reference: page 167 of Hairer, Nørsett, and Wanner (2008)
    Zonneveld4,

    /// Fehlberg method (explicit, order 4(5), embedded)
    ///
    /// Note: this method gives identically zero error estimates for quadrature problems `y'=f(x)` (see page 180 of ref # 1).
    Fehlberg4,

    /// Dormand-Prince method (explicit, order 5(4), embedded)
    DoPri5,

    /// Verner method (explicit, order 6(5), embedded)
    Verner6,

    /// Fehlberg method (explicit, order 7(8), embedded)
    ///
    /// Note: this method gives identically zero error estimates for quadrature problems `y'=f(x)` (see page 180 of ref # 1).
    Fehlberg7,

    /// Dormand-Prince method (explicit, order 8(5,3), embedded)
    DoPri8,
}

impl Method {
    #[rustfmt::skip]
    pub fn information(&self) -> Information {
        match self {
            Method::Radau5     => Information { order: 5, order_of_estimator: 4, implicit: true,  embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::BwEuler    => Information { order: 1, order_of_estimator: 0, implicit: true,  embedded: false, multiple_stages: false, first_step_same_as_last: false },
            Method::FwEuler    => Information { order: 1, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: false, first_step_same_as_last: false },
            Method::Rk2        => Information { order: 2, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: true,  first_step_same_as_last: false },
            Method::Rk3        => Information { order: 3, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: true,  first_step_same_as_last: false },
            Method::Heun3      => Information { order: 3, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: true,  first_step_same_as_last: false },
            Method::Rk4        => Information { order: 4, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: true,  first_step_same_as_last: false },
            Method::Rk4alt     => Information { order: 4, order_of_estimator: 0, implicit: false, embedded: false, multiple_stages: true,  first_step_same_as_last: false },
            Method::MdEuler    => Information { order: 2, order_of_estimator: 1, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::Merson4    => Information { order: 4, order_of_estimator: 3, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::Zonneveld4 => Information { order: 4, order_of_estimator: 3, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::Fehlberg4  => Information { order: 4, order_of_estimator: 4, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::DoPri5     => Information { order: 5, order_of_estimator: 4, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: true  },
            Method::Verner6    => Information { order: 6, order_of_estimator: 5, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::Fehlberg7  => Information { order: 7, order_of_estimator: 8, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
            Method::DoPri8     => Information { order: 8, order_of_estimator: 7, implicit: false, embedded: true,  multiple_stages: true,  first_step_same_as_last: false },
        }
    }
}
