use crate::StrError;
use crate::{Grid2d, NaturalBcs2d, Side};
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (EBC) for 2D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x and `ny` points along y.
pub struct EssentialBcs2d<'a> {
    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    pub(crate) periodic_along_x: bool,

    /// Indicates that the boundary is periodic along x (bottom ϕ values equal top ϕ values)
    ///
    /// If false, the bottom/top boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    pub(crate) periodic_along_y: bool,

    /// Holds the functions to compute essential boundary conditions (EBC)
    ///
    /// The function is `f(x, y) -> value`
    ///
    /// (4) → (Xmin, Xmax, Ymin, Ymax); corresponding to the 4 sides
    pub(crate) functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Indicates the sides where essential boundary conditions are applied
    pub(crate) sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax
}

impl<'a> EssentialBcs2d<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        EssentialBcs2d {
            periodic_along_x: false,
            periodic_along_y: false,
            functions: vec![
                Arc::new(|_, _| 0.0), // Xmin
                Arc::new(|_, _| 0.0), // Xmax
                Arc::new(|_, _| 0.0), // Ymin
                Arc::new(|_, _| 0.0), // Ymax
            ],
            sides: [false; 4],
        }
    }

    /// Sets periodic boundary condition
    pub fn set_periodic(&mut self, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x, y) -> value`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin | Side::Xmax => self.periodic_along_x = false,
            Side::Ymin | Side::Ymax => self.periodic_along_y = false,
        };
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self) {
        self.periodic_along_x = false;
        self.periodic_along_y = false;
        self.functions = vec![
            Arc::new(|_, _| 0.0), // Xmin
            Arc::new(|_, _| 0.0), // Xmax
            Arc::new(|_, _| 0.0), // Ymin
            Arc::new(|_, _| 0.0), // Ymax
        ];
        self.sides[0] = true;
        self.sides[1] = true;
        self.sides[2] = true;
        self.sides[3] = true;
    }

    /// Makes sure that all sides have either EBC or NBC, but not both
    pub(crate) fn validate(&self, nbcs: &NaturalBcs2d) -> Result<(), StrError> {
        if self.sides[0] && nbcs.sides[0] {
            return Err("Xmin side must not have both EBC and NBC");
        }
        if self.sides[1] && nbcs.sides[1] {
            return Err("Xmax side must not have both EBC and NBC");
        }
        if self.sides[2] && nbcs.sides[2] {
            return Err("Ymin side must not have both EBC and NBC");
        }
        if self.sides[3] && nbcs.sides[3] {
            return Err("Ymax side must not have both EBC and NBC");
        }
        if self.periodic_along_x {
            if nbcs.sides[0] || nbcs.sides[1] {
                return Err("Periodic X does not allow NBC on Xmin or Xmax");
            }
        } else {
            if !self.sides[0] && !nbcs.sides[0] {
                return Err("Xmin side is missing either EBC or NBC");
            }
            if !self.sides[1] && !nbcs.sides[1] {
                return Err("Xmax side is missing either EBC or NBC");
            }
        }
        if self.periodic_along_y {
            if nbcs.sides[2] || nbcs.sides[3] {
                return Err("Periodic Y does not allow NBC on Ymin or Ymax");
            }
        } else {
            if !self.sides[2] && !nbcs.sides[2] {
                return Err("Ymin side is missing either EBC or NBC");
            }
            if !self.sides[3] && !nbcs.sides[3] {
                return Err("Ymax side is missing either EBC or NBC");
            }
        }
        Ok(())
    }

    /// Returns the nodes with EBCs
    pub(crate) fn get_nodes(&self, grid: &Grid2d) -> Vec<usize> {
        let mut nodes = Vec::new();
        for index in 0..4 {
            if self.sides[index] {
                let side = Side::from_index(index);
                for &m in grid.get_nodes_on_side(side) {
                    if !nodes.contains(&m) {
                        nodes.push(m);
                    }
                }
            }
        }
        nodes
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EssentialBcs2d;
    use crate::{Grid2d, NaturalBcs2d, Side};

    #[test]
    fn new_works() {
        let ebcs = EssentialBcs2d::new();
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.periodic_along_y);
        assert!(!ebcs.sides[0]);
        assert!(!ebcs.sides[1]);
        assert!(!ebcs.sides[2]);
        assert!(!ebcs.sides[3]);
    }

    #[test]
    fn set_periodic_works() {
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_periodic(true, true);
        assert!(ebcs.periodic_along_x);
        assert!(ebcs.periodic_along_y);
        ebcs.set_periodic(false, false);
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.periodic_along_y);
    }

    #[test]
    fn set_works() {
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_periodic(true, true); // should be reset
        ebcs.set(Side::Xmin, |_, _| 1.0);
        assert!(!ebcs.periodic_along_x);
        assert!(ebcs.periodic_along_y); // Y should remain periodic
        assert!(ebcs.sides[0]);
        assert!(!ebcs.sides[1]);
        assert_eq!((ebcs.functions[0])(0.0, 0.0), 1.0);

        ebcs.set(Side::Ymax, |_, _| 2.0);
        assert!(!ebcs.periodic_along_y); // Y should be reset now
        assert!(ebcs.sides[3]);
        assert_eq!((ebcs.functions[3])(0.0, 0.0), 2.0);
    }

    #[test]
    fn set_homogeneous_works() {
        let mut ebcs = EssentialBcs2d::new();
        ebcs.set_periodic(true, true); // should be reset
        ebcs.set_homogeneous();
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.periodic_along_y);
        for i in 0..4 {
            assert!(ebcs.sides[i]);
            assert_eq!((ebcs.functions[i])(123.0, 456.0), 0.0);
        }
    }

    #[test]
    fn validate_works() {
        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();

        // 1. Missing BCs
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmin side is missing either EBC or NBC")
        );

        ebcs.set(Side::Xmin, |_, _| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmax side is missing either EBC or NBC")
        );

        // 2. Valid configuration (mixed)
        nbcs.set(Side::Xmax, |_, _| 0.0);
        ebcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(ebcs.validate(&nbcs), Ok(()));

        // 3. Both EBC and NBC on same side
        ebcs.set(Side::Xmax, |_, _| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmax side must not have both EBC and NBC")
        );

        // 4. Periodic with NBC
        let mut ebcs = EssentialBcs2d::new();
        let mut nbcs = NaturalBcs2d::new();
        ebcs.set_periodic(true, false);
        nbcs.set(Side::Xmin, |_, _| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Periodic X does not allow NBC on Xmin or Xmax")
        );

        // 5. Periodic without NBC (Valid)
        let mut nbcs = NaturalBcs2d::new();
        // Need to set Y BCs to pass validation
        nbcs.set(Side::Ymin, |_, _| 0.0);
        nbcs.set(Side::Ymax, |_, _| 0.0);
        assert_eq!(ebcs.validate(&nbcs), Ok(()));
    }

    #[test]
    fn get_nodes_works() {
        let mut ebcs = EssentialBcs2d::new();
        let grid = Grid2d::new(&[0.0, 0.5, 1.0], &[0.0, 0.5, 1.0]).unwrap();
        // Grid 3x3:
        // 6 7 8
        // 3 4 5
        // 0 1 2

        assert_eq!(ebcs.get_nodes(&grid).len(), 0);

        ebcs.set(Side::Xmin, |_, _| 0.0);
        // Xmin nodes: 0, 3, 6
        let mut nodes = ebcs.get_nodes(&grid);
        nodes.sort();
        assert_eq!(nodes, &[0, 3, 6]);

        ebcs.set(Side::Ymin, |_, _| 0.0);
        // Ymin nodes: 0, 1, 2. Combined with Xmin: 0, 1, 2, 3, 6
        let mut nodes = ebcs.get_nodes(&grid);
        nodes.sort();
        assert_eq!(nodes, &[0, 1, 2, 3, 6]);
    }
}
