/// Holds information about the numerical method to solve (approximate) ODEs
#[derive(Clone, Copy, Debug)]
pub struct Information {
    /// Is the order of y1 (corresponding to B); i.e., the "p" constant
    pub order: usize,

    /// Is he order of error estimator (embedded only); i.e., the "q" constant
    ///
    /// For DoPri5(4): q = 4 = min(order(y1), order(y1bar))
    pub order_of_estimator: usize, // 0 means no error estimator available

    /// Indicates implicit method instead of explicit
    pub implicit: bool,

    /// Indicates that the method has embedded error estimator
    pub embedded: bool,

    /// Indicates that the method has more than one stage
    pub multiple_stages: bool,

    /// Indicates that the first step's coefficient is equal to the last step's coefficient (FSAL)
    ///
    /// See explanation about `FSAL` in Hairer-Nørsett-Wanner Part I, pages 167 and 178
    pub first_step_same_as_last: bool,
}

/// Specifies the numerical method to solve (approximate) ODEs
///
/// # Recommended methods
///
/// * [Method::DoPri5] for ODE systems and non-stiff problems using moderate tolerances
/// * [Method::DoPri8] for ODE systems and non-stiff problems using strict tolerances
/// * [Method::Radau5] for ODE and DAE systems, possibly stiff, with moderate to strict tolerances
///
/// **Note:** A *Stiff problem* arises due to a combination of conditions, such as
/// the ODE system equations, the initial values, the stepsize, and the numerical method.
///
/// # Limitations
///
/// * Currently, the only method that can solve DAE systems is [Method::Radau5]
/// * Currently, *dense output* is only available for [Method::DoPri5], [Method::DoPri8], and [Method::Radau5]
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    /// Radau method (Radau IIA) (implicit, order 5, embedded) for ODEs and DAEs
    Radau5,

    /// Backward Euler method (implicit, order 1, unconditionally stable)
    BwEuler,

    /// Forward Euler method (explicit, order 1, conditionally stable)
    ///
    /// **Note:** This method is interesting for didactic purposes only
    /// and should not be used in production codes.
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
    /// Returns information about the method
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

    /// Returns a description of the method
    pub fn description(&self) -> &'static str {
        match self {
            Method::Radau5 => "Radau method (Radau IIA) (implicit, order 5, embedded)",
            Method::BwEuler => "Backward Euler method (implicit, order 1)",
            Method::FwEuler => "Forward Euler method (explicit, order 1)",
            Method::Rk2 => "Runge (Kutta) method (mid-point) (explicit, order 2)",
            Method::Rk3 => "Runge (Kutta) method (explicit, order 3)",
            Method::Heun3 => "Heun method (explicit, order 3)",
            Method::Rk4 => "(The) Runge-Kutta method (explicit, order 4)",
            Method::Rk4alt => "Runge-Kutta method (alternative) (explicit, order 4, 3/8-Rule)",
            Method::MdEuler => "Modified Euler method (explicit, order 2(1), embedded)",
            Method::Merson4 => "Merson method (explicit, order 4('5'), embedded)",
            Method::Zonneveld4 => "Zonneveld method (explicit, order 4(3), embedded)",
            Method::Fehlberg4 => "Fehlberg method (explicit, order 4(5), embedded)",
            Method::DoPri5 => "Dormand-Prince method (explicit, order 5(4), embedded)",
            Method::Verner6 => "Verner method (explicit, order 6(5), embedded)",
            Method::Fehlberg7 => "Fehlberg method (explicit, order 7(8), embedded)",
            Method::DoPri8 => "Dormand-Prince method (explicit, order 8(5,3), embedded)",
        }
    }

    /// Returns a list of explicit Runge-Kutta methods
    ///
    /// **Note:** FwEuler is also an explicit RK method; however it is not included
    /// in this list because it is implemented separately.
    pub fn erk_methods() -> Vec<Method> {
        vec![
            Method::Rk2,
            Method::Rk3,
            Method::Heun3,
            Method::Rk4,
            Method::Rk4alt,
            Method::MdEuler,
            Method::Merson4,
            Method::Zonneveld4,
            Method::Fehlberg4,
            Method::DoPri5,
            Method::Verner6,
            Method::Fehlberg7,
            Method::DoPri8,
        ]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn information_clone_copy_and_debug_work() {
        let info = Information {
            order: 2,
            order_of_estimator: 1,
            implicit: false,
            embedded: true,
            multiple_stages: true,
            first_step_same_as_last: false,
        };
        let copy = info;
        let clone = info.clone();
        assert_eq!(format!("{:?}", info), "Information { order: 2, order_of_estimator: 1, implicit: false, embedded: true, multiple_stages: true, first_step_same_as_last: false }");
        assert_eq!(copy.order, 2);
        assert_eq!(copy.order_of_estimator, 1);
        assert_eq!(copy.implicit, false);
        assert_eq!(copy.embedded, true);
        assert_eq!(copy.first_step_same_as_last, false);
        assert_eq!(clone.order, 2);
        assert_eq!(clone.order_of_estimator, 1);
        assert_eq!(clone.implicit, false);
        assert_eq!(clone.embedded, true);
        assert_eq!(clone.first_step_same_as_last, false);
    }

    #[test]
    fn method_clone_copy_and_debug_work() {
        let method = Method::BwEuler;
        let copy = method;
        let clone = method.clone();
        assert_eq!(format!("{:?}", method), "BwEuler");
        assert_eq!(copy, Method::BwEuler);
        assert_eq!(clone, Method::BwEuler);
    }

    #[test]
    #[rustfmt::skip]
    fn methods_information_works() {
        let m = Method::Radau5    ; let i=m.information(); assert_eq!(i.order,5); assert_eq!(i.order_of_estimator,4); assert_eq!(i.implicit,true,); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::BwEuler   ; let i=m.information(); assert_eq!(i.order,1); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,true,); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,false); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::FwEuler   ; let i=m.information(); assert_eq!(i.order,1); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,false); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Rk2       ; let i=m.information(); assert_eq!(i.order,2); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Rk3       ; let i=m.information(); assert_eq!(i.order,3); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Heun3     ; let i=m.information(); assert_eq!(i.order,3); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Rk4       ; let i=m.information(); assert_eq!(i.order,4); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Rk4alt    ; let i=m.information(); assert_eq!(i.order,4); assert_eq!(i.order_of_estimator,0); assert_eq!(i.implicit,false); assert_eq!(i.embedded,false); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::MdEuler   ; let i=m.information(); assert_eq!(i.order,2); assert_eq!(i.order_of_estimator,1); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Merson4   ; let i=m.information(); assert_eq!(i.order,4); assert_eq!(i.order_of_estimator,3); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Zonneveld4; let i=m.information(); assert_eq!(i.order,4); assert_eq!(i.order_of_estimator,3); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Fehlberg4 ; let i=m.information(); assert_eq!(i.order,4); assert_eq!(i.order_of_estimator,4); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::DoPri5    ; let i=m.information(); assert_eq!(i.order,5); assert_eq!(i.order_of_estimator,4); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,true );
        let m = Method::Verner6   ; let i=m.information(); assert_eq!(i.order,6); assert_eq!(i.order_of_estimator,5); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::Fehlberg7 ; let i=m.information(); assert_eq!(i.order,7); assert_eq!(i.order_of_estimator,8); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
        let m = Method::DoPri8    ; let i=m.information(); assert_eq!(i.order,8); assert_eq!(i.order_of_estimator,7); assert_eq!(i.implicit,false); assert_eq!(i.embedded,true ); assert_eq!(i.multiple_stages,true ); assert_eq!(i.first_step_same_as_last,false);
    }

    #[test]
    fn explicit_and_implicit_methods_work() {
        let erk = Method::erk_methods();
        assert_eq!(
            erk,
            &[
                Method::Rk2,
                Method::Rk3,
                Method::Heun3,
                Method::Rk4,
                Method::Rk4alt,
                Method::MdEuler,
                Method::Merson4,
                Method::Zonneveld4,
                Method::Fehlberg4,
                Method::DoPri5,
                Method::Verner6,
                Method::Fehlberg7,
                Method::DoPri8,
            ]
        );
    }

    #[test]
    fn description_works() {
        for m in [Method::Radau5, Method::BwEuler, Method::FwEuler] {
            assert!(m.description().len() > 0);
        }
        for m in Method::erk_methods() {
            assert!(m.description().len() > 0);
        }
    }
}
