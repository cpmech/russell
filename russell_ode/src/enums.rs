/// Specifies the numerical ODE solver method
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics ISSN 0179-3632, 528p
///
/// # Notes
///
/// 1. Fehlberg's methods give identically zero error estimates for quadrature problems `y'=f(x)`;
///    see page 180 of [1]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OdeMethod {
    /// Forward Euler method (explicit, order 1)
    FwEuler,

    /// Modified Euler method (explicit, order 2(1))
    MdEuler,

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

    /// Merson method (explicit, order 4("5"))
    ///
    /// "5" means that the order 5 is for linear equations with constant coefficients;
    /// otherwise the method is of order3.
    ///
    /// Reference: page 167 of Hairer, Nørsett, and Wanner (2008)
    Merson4,

    /// Zonneveld method (explicit, order 4(3))
    ///
    /// Reference: page 167 of Hairer, Nørsett, and Wanner (2008)
    Zonneveld4,

    /// Fehlberg method (explicit, order 4(5))
    Fehlberg4,

    /// Dormand-Prince method (explicit, order 5(4))
    DoPri5,

    /// Verner method (explicit, order 6(5))
    Verner6,

    /// Fehlberg method (explicit, order 7(8))
    Fehlberg7,

    /// Dormand-Prince method (explicit, order 8(5,3))
    DoPri8,

    /// Backward Euler method (implicit, order 1)
    BwEuler,

    /// Radau method (Radau IIA) (implicit, order 5)
    Radau5,
}
