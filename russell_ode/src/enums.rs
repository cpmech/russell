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

    pub fn explicit_methods() -> Vec<Method> {
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

    pub fn implicit_methods() -> Vec<Method> {
        vec![Method::Radau5, Method::BwEuler]
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
        let explicit = Method::explicit_methods();
        let implicit = Method::implicit_methods();
        assert_eq!(explicit.len() + implicit.len(), 15);
        assert_eq!(
            explicit,
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
        assert_eq!(implicit, &[Method::Radau5, Method::BwEuler]);
    }
}
