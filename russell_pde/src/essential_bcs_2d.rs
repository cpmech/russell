use crate::StrError;
use crate::{Grid2d, NaturalBcs2d, Side};
use std::sync::Arc;

/// Manages essential (Dirichlet) boundary conditions for 2D PDE problems
///
/// Essential boundary conditions (EBCs), also known as Dirichlet boundary conditions,
/// specify the values of the solution at the domain boundaries. For a 2D domain,
/// these conditions take the form:
///
/// * At x = xₘᵢₙ: ϕ(xₘᵢₙ, y) = f(xₘᵢₙ, y)
/// * At x = xₘₐₓ: ϕ(xₘₐₓ, y) = f(xₘₐₓ, y)
/// * At y = yₘᵢₙ: ϕ(x, yₘᵢₙ) = f(x, yₘᵢₙ)
/// * At y = yₘₐₓ: ϕ(x, yₘₐₓ) = f(x, yₘₐₓ)
///
/// where f is a user-defined function that computes the boundary value.
///
/// # Examples
///
/// ```
/// use russell_pde::{EssentialBcs2d, Side};
///
/// let mut ebcs = EssentialBcs2d::new();
///
/// // Set a constant value at the left boundary
/// ebcs.set(Side::Xmin, |_x, _y| 10.0);
///
/// // Set a spatially-varying value at the bottom boundary
/// ebcs.set(Side::Ymin, |x, y| x * x + y * y);
///
/// // Set periodic boundaries along x
/// ebcs.set_periodic(true, false);
/// ```
pub struct EssentialBcs2d<'a> {
    /// Indicates that the boundary is periodic along x
    ///
    /// When `true`, the solution satisfies ϕ(xₘᵢₙ, y) = ϕ(xₘₐₓ, y) for all y,
    /// creating a periodic domain in the x-direction. This is commonly used for
    /// problems with cyclic symmetry or when simulating infinite domains.
    pub(crate) periodic_along_x: bool,

    /// Indicates that the boundary is periodic along y
    ///
    /// When `true`, the solution satisfies ϕ(x, yₘᵢₙ) = ϕ(x, yₘₐₓ) for all x,
    /// creating a periodic domain in the y-direction. This is commonly used for
    /// problems with cyclic symmetry or when simulating infinite domains.
    pub(crate) periodic_along_y: bool,

    /// Functions to compute essential boundary condition values
    ///
    /// Each function has the signature `f(x, y) -> value` where:
    /// * `x`, `y` are the spatial coordinates at the boundary
    /// * `value` is the prescribed solution value at that location
    ///
    /// The vector contains 4 functions corresponding to [Xmin, Xmax, Ymin, Ymax] boundaries.
    pub(crate) functions: Vec<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'a>>,

    /// Flags indicating which sides have essential boundary conditions
    ///
    /// Array of 4 booleans: [Xmin, Xmax, Ymin, Ymax]
    /// * `true`: Essential boundary condition is active on this side
    /// * `false`: No essential boundary condition on this side
    pub(crate) sides: [bool; 4], // Xmin, Xmax, Ymin, Ymax
}

impl<'a> EssentialBcs2d<'a> {
    /// Creates a new instance with no boundary conditions set
    ///
    /// # Returns
    ///
    /// Returns an `EssentialBcs2d` instance with:
    /// * No periodic boundaries in either direction
    /// * No active essential boundary conditions
    /// * Default zero-valued functions for all sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::EssentialBcs2d;
    ///
    /// let ebcs = EssentialBcs2d::new();
    /// // Initially, no boundary conditions are active
    /// ```
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

    /// Sets or un-sets periodic boundary conditions along x and y
    ///
    /// When enabled, this enforces:
    /// * Along x: ϕ(xₘᵢₙ, y) = ϕ(xₘₐₓ, y) for all y
    /// * Along y: ϕ(x, yₘᵢₙ) = ϕ(x, yₘₐₓ) for all x
    ///
    /// # Input
    ///
    /// * `along_x` - If `true`, enables periodic boundaries along x; if `false`, disables them
    /// * `along_y` - If `true`, enables periodic boundaries along y; if `false`, disables them
    ///
    /// # Notes
    ///
    /// * Periodic boundaries can be set independently in each direction
    /// * Setting periodic boundaries automatically clears existing essential BCs on those sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::EssentialBcs2d;
    ///
    /// let mut ebcs = EssentialBcs2d::new();
    ///
    /// // Periodic in x, but not in y
    /// ebcs.set_periodic(true, false);
    ///
    /// // Periodic in both directions
    /// ebcs.set_periodic(true, true);
    /// ```
    pub fn set_periodic(&mut self, along_x: bool, along_y: bool) {
        self.periodic_along_x = along_x;
        self.periodic_along_y = along_y;
        if along_x {
            self.sides[0] = false; // Xmin
            self.sides[1] = false; // Xmax
        }
        if along_y {
            self.sides[2] = false; // Ymin
            self.sides[3] = false; // Ymax
        }
    }

    /// Sets an essential (Dirichlet) boundary condition on a specified side
    ///
    /// This method prescribes the solution value at a boundary using a function that
    /// can depend on the spatial coordinates.
    ///
    /// # Input
    ///
    /// * `side` - The boundary side (`Side::Xmin`, `Side::Xmax`, `Side::Ymin`, or `Side::Ymax`)
    /// * `f` - Function with signature `f(x, y) -> value` that computes the boundary value
    ///   * `x`, `y`: Spatial coordinates at the boundary
    ///   * `value`: The prescribed solution value ϕ(x, y)
    ///
    /// # Notes
    ///
    /// * Setting an essential boundary condition on Xmin or Xmax will disable periodic boundaries along x
    /// * Setting an essential boundary condition on Ymin or Ymax will disable periodic boundaries along y
    /// * The function `f` is stored and will be called during boundary condition assembly
    /// * The function must be thread-safe (`Send + Sync`)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{EssentialBcs2d, Side};
    ///
    /// let mut ebcs = EssentialBcs2d::new();
    ///
    /// // Constant boundary value
    /// ebcs.set(Side::Xmin, |_x, _y| 5.0);
    ///
    /// // Spatially-varying boundary value
    /// ebcs.set(Side::Ymin, |x, y| x.sin() * y.cos());
    ///
    /// // Temperature profile along a boundary
    /// ebcs.set(Side::Ymax, |x, _y| 100.0 * (1.0 - x));
    /// ```
    pub fn set(&mut self, side: Side, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin | Side::Xmax => self.periodic_along_x = false,
            Side::Ymin | Side::Ymax => self.periodic_along_y = false,
        };
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Sets homogeneous (zero) essential boundary conditions on all boundaries
    ///
    /// This is a convenience method that sets ϕ = 0 on all four boundaries:
    /// * ϕ(xₘᵢₙ, y) = 0
    /// * ϕ(xₘₐₓ, y) = 0
    /// * ϕ(x, yₘᵢₙ) = 0
    /// * ϕ(x, yₘₐₓ) = 0
    ///
    /// Homogeneous boundary conditions are commonly used in many physics problems,
    /// such as:
    /// * Heat conduction with fixed zero temperature at boundaries
    /// * Wave equations with fixed boundaries
    /// * Potential problems with grounded boundaries
    /// * Membrane vibration with clamped edges
    ///
    /// # Notes
    ///
    /// * This method disables any periodic boundary conditions in both directions
    /// * All four boundaries are set to zero
    /// * Equivalent to calling `set(side, |_, _| 0.0)` for all four sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::EssentialBcs2d;
    ///
    /// let mut ebcs = EssentialBcs2d::new();
    /// ebcs.set_homogeneous();
    /// // Now ϕ = 0 on all boundaries
    /// ```
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

    /// Validates boundary conditions against natural boundary conditions
    ///
    /// This method ensures that the boundary conditions are properly specified by checking:
    ///
    /// 1. Each boundary side has either an essential BC (EBC) or a natural BC (NBC), but not both
    /// 2. If periodic boundaries are set along x, no natural BCs are specified on Xmin or Xmax
    /// 3. If periodic boundaries are set along y, no natural BCs are specified on Ymin or Ymax
    /// 4. If not periodic in a direction, both boundaries in that direction must have either EBC or NBC
    ///
    /// # Input
    ///
    /// * `nbcs` - Reference to the natural boundary conditions to validate against
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the boundary conditions are valid
    /// * `Err(StrError)` with a descriptive message if validation fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * A side has both EBC and NBC
    /// * Periodic boundaries in a direction are combined with NBCs on those sides
    /// * A side is missing both EBC and NBC
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{EssentialBcs2d, NaturalBcs2d, Side};
    ///
    /// let mut ebcs = EssentialBcs2d::new();
    /// let mut nbcs = NaturalBcs2d::new();
    ///
    /// ebcs.set(Side::Xmin, |_, _| 0.0);
    /// nbcs.set(Side::Xmax, |_, _| 0.0);
    /// ebcs.set(Side::Ymin, |_, _| 0.0);
    /// nbcs.set(Side::Ymax, |_, _| 0.0);
    ///
    /// // Valid: each side has either EBC or NBC
    /// assert!(ebcs.validate(&nbcs).is_ok());
    /// ```
    pub fn validate(&self, nbcs: &NaturalBcs2d) -> Result<(), StrError> {
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

    /// Returns the grid node indices where essential boundary conditions are applied
    ///
    /// This method collects all unique node indices from all sides that have active EBCs.
    /// If a node belongs to multiple sides (e.g., a corner node), it is included only once.
    ///
    /// # Input
    ///
    /// * `grid` - Reference to the 2D grid
    ///
    /// # Returns
    ///
    /// A vector of node indices where EBCs are active. The node indices are not necessarily
    /// sorted and duplicates are removed.
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
        ebcs.set(Side::Xmin, |_, _| 123.0); // to ensure it is reset
        ebcs.set(Side::Xmax, |_, _| 123.0); // to ensure it is reset
        ebcs.set(Side::Ymin, |_, _| 123.0); // to ensure it is reset
        ebcs.set(Side::Ymax, |_, _| 123.0); // to ensure it is reset

        ebcs.set_periodic(true, true);
        assert!(ebcs.periodic_along_x);
        assert!(ebcs.periodic_along_y);
        assert_eq!(ebcs.sides[0], false); // Xmin EBC should be reset
        assert_eq!(ebcs.sides[1], false); // Xmax EBC should be reset
        assert_eq!(ebcs.sides[2], false); // Ymin EBC should be reset
        assert_eq!(ebcs.sides[3], false); // Ymax EBC should be reset

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
