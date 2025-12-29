use crate::StrError;
use crate::{Grid1d, NaturalBcs1d, Side};
use std::sync::Arc;

/// Manages essential (Dirichlet) boundary conditions for 1D PDE problems
///
/// Essential boundary conditions (EBCs), also known as Dirichlet boundary conditions,
/// specify the values of the solution at the domain boundaries. For a 1D domain,
/// these conditions take the form:
///
/// * At x = xₘᵢₙ: ϕ(xₘᵢₙ) = f(xₘᵢₙ)
/// * At x = xₘₐₓ: ϕ(xₘₐₓ) = f(xₘₐₓ)
///
/// where f is a user-defined function that computes the boundary value.
///
/// # Examples
///
/// ```
/// use russell_pde::{EssentialBcs1d, Side};
///
/// let mut ebcs = EssentialBcs1d::new();
///
/// // Set a constant value at the left boundary
/// ebcs.set(Side::Xmin, |_x| 10.0);
///
/// // Set a spatially-varying value at the right boundary
/// ebcs.set(Side::Xmax, |x| x * x);
/// ```
pub struct EssentialBcs1d<'a> {
    /// Indicates that the boundary is periodic along x
    ///
    /// When `true`, the solution satisfies ϕ(xₘᵢₙ) = ϕ(xₘₐₓ), creating a periodic domain.
    /// This is commonly used for problems with cyclic symmetry or when simulating
    /// infinite domains.
    pub(crate) periodic_along_x: bool,

    /// Functions to compute essential boundary condition values
    ///
    /// Each function has the signature `f(x) -> value` where:
    /// * `x` is the spatial coordinate at the boundary
    /// * `value` is the prescribed solution value at that location
    ///
    /// The vector contains 2 functions corresponding to `[Xmin, Xmax]` boundaries.
    pub(crate) functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Flags indicating which sides have essential boundary conditions
    ///
    /// Array of 2 booleans: `[Xmin, Xmax]`
    /// * `true`: Essential boundary condition is active on this side
    /// * `false`: No essential boundary condition on this side
    pub(crate) sides: [bool; 2],
}

impl<'a> EssentialBcs1d<'a> {
    /// Creates a new instance with no boundary conditions set
    ///
    /// # Returns
    ///
    /// Returns an `EssentialBcs1d` instance with:
    /// * No periodic boundaries
    /// * No active essential boundary conditions
    /// * Default zero-valued functions for all sides
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::EssentialBcs1d;
    ///
    /// let ebcs = EssentialBcs1d::new();
    /// // Initially, no boundary conditions are active
    /// ```
    pub fn new() -> Self {
        EssentialBcs1d {
            periodic_along_x: false,
            functions: vec![
                Arc::new(|_| 0.0), // Xmin
                Arc::new(|_| 0.0), // Xmax
            ],
            sides: [false; 2],
        }
    }

    /// Sets or un-sets periodic boundary conditions along x
    ///
    /// When enabled, this enforces ϕ(xₘᵢₙ) = ϕ(xₘₐₓ), making the domain periodic.
    ///
    /// # Input
    ///
    /// * `along_x` - If `true`, enables periodic boundaries; if `false`, disables them
    ///
    /// # Note
    ///
    /// Setting periodic boundary conditions will automatically clear any existing
    /// essential boundary conditions on the x boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{EssentialBcs1d, Side};
    ///
    /// let mut ebcs = EssentialBcs1d::new();
    /// ebcs.set_periodic(true);
    /// // Now the solution is periodic: ϕ(xₘᵢₙ) = ϕ(xₘₐₓ)
    /// ```
    pub fn set_periodic(&mut self, along_x: bool) {
        self.periodic_along_x = along_x;
        if along_x {
            self.sides[0] = false; // Xmin
            self.sides[1] = false; // Xmax
        }
    }

    /// Sets an essential (Dirichlet) boundary condition on a specified side
    ///
    /// This method prescribes the solution value at a boundary using a function that
    /// can depend on the spatial coordinate.
    ///
    /// # Input
    ///
    /// * `side` - The boundary side (must be `Side::Xmin` or `Side::Xmax`)
    /// * `f` - Function with signature `f(x) -> value` that computes the boundary value
    ///   * `x`: Spatial coordinate at the boundary
    ///   * `value`: The prescribed solution value ϕ(x)
    ///
    /// # Notes
    ///
    /// * Setting an essential boundary condition will disable any periodic boundary condition
    /// * The function `f` is stored and will be called during boundary condition assembly
    /// * The function must be thread-safe (`Send + Sync`)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{EssentialBcs1d, Side};
    ///
    /// let mut ebcs = EssentialBcs1d::new();
    ///
    /// // Constant boundary value
    /// ebcs.set(Side::Xmin, |_x| 5.0);
    ///
    /// // Spatially-varying boundary value
    /// ebcs.set(Side::Xmax, |x| x.sin());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if an invalid side is provided (only `Side::Xmin` and `Side::Xmax` are valid for 1D).
    pub fn set(&mut self, side: Side, f: impl Fn(f64) -> f64 + Send + Sync + 'a) {
        self.periodic_along_x = false;
        let index = side as usize;
        self.functions[index] = Arc::new(f);
        self.sides[index] = true;
    }

    /// Sets homogeneous (zero) essential boundary conditions on all boundaries
    ///
    /// This is a convenience method that sets ϕ(xₘᵢₙ) = 0 and ϕ(xₘₐₓ) = 0.
    ///
    /// # Notes
    ///
    /// * This method disables any periodic boundary conditions
    /// * Both left and right boundaries are set to zero
    /// * Equivalent to calling `set(Side::Xmin, |_| 0.0)` and `set(Side::Xmax, |_| 0.0)`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::EssentialBcs1d;
    ///
    /// let mut ebcs = EssentialBcs1d::new();
    /// ebcs.set_homogeneous();
    /// // Now ϕ(xₘᵢₙ) = 0 and ϕ(xₘₐₓ) = 0
    /// ```
    pub fn set_homogeneous(&mut self) {
        self.periodic_along_x = false;
        self.functions = vec![
            Arc::new(|_| 0.0), // Xmin
            Arc::new(|_| 0.0), // Xmax
        ];
        self.sides[0] = true;
        self.sides[1] = true;
    }

    /// Validates boundary conditions against natural boundary conditions
    ///
    /// This method ensures that the boundary conditions are properly specified by checking:
    ///
    /// 1. Each boundary side has either an essential BC (EBC) or a natural BC (NBC), but not both
    /// 2. If periodic boundaries are set, no natural BCs are specified on those boundaries
    /// 3. If not periodic, all boundaries must have either an EBC or NBC defined
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
    /// * Periodic boundaries are combined with NBCs
    /// * A side is missing both EBC and NBC
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_pde::{EssentialBcs1d, NaturalBcs1d, Side};
    ///
    /// let mut ebcs = EssentialBcs1d::new();
    /// let mut nbcs = NaturalBcs1d::new();
    ///
    /// ebcs.set(Side::Xmin, |_| 0.0);
    /// nbcs.set(Side::Xmax, |_| 0.0);
    ///
    /// // Valid: Xmin has EBC, Xmax has NBC
    /// assert!(ebcs.validate(&nbcs).is_ok());
    /// ```
    pub fn validate(&self, nbcs: &NaturalBcs1d) -> Result<(), StrError> {
        if self.sides[0] && nbcs.sides[0] {
            return Err("Xmin side must not have both EBC and NBC");
        }
        if self.sides[1] && nbcs.sides[1] {
            return Err("Xmax side must not have both EBC and NBC");
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
        Ok(())
    }

    /// Returns the grid node indices where essential boundary conditions are applied
    ///
    /// # Input
    ///
    /// * `grid` - Reference to the 1D grid
    ///
    /// # Returns
    ///
    /// A vector of node indices where EBCs are active. For a 1D grid:
    /// * Node 0 corresponds to Xmin
    /// * Node (nx-1) corresponds to Xmax
    pub(crate) fn get_nodes(&self, grid: &Grid1d) -> Vec<usize> {
        let mut nodes = Vec::new();
        for side in 0..2 {
            if self.sides[side] {
                let m = if side == 0 { 0 } else { grid.nx() - 1 };
                nodes.push(m);
            }
        }
        nodes
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EssentialBcs1d;
    use crate::{Grid1d, NaturalBcs1d, Side};

    #[test]
    fn new_works() {
        let ebcs = EssentialBcs1d::new();
        assert!(!ebcs.periodic_along_x);
        assert!(!ebcs.sides[0]);
        assert!(!ebcs.sides[1]);
    }

    #[test]
    fn set_periodic_works() {
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set(Side::Xmin, |_| 123.0);
        ebcs.set(Side::Xmax, |_| 123.0);
        ebcs.set_periodic(true);
        assert!(ebcs.periodic_along_x);
        assert_eq!(ebcs.sides[0], false); // Xmin EBC should be reset
        assert_eq!(ebcs.sides[1], false); // Xmax EBC should be reset
        ebcs.set_periodic(false);
        assert!(!ebcs.periodic_along_x);
    }

    #[test]
    fn set_works() {
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set_periodic(true); // should be reset
        ebcs.set(Side::Xmin, |_| 1.0);
        assert!(!ebcs.periodic_along_x);
        assert!(ebcs.sides[0]);
        assert!(!ebcs.sides[1]);
        assert_eq!((ebcs.functions[0])(0.0), 1.0);

        ebcs.set(Side::Xmax, |_| 2.0);
        assert!(ebcs.sides[0]);
        assert!(ebcs.sides[1]);
        assert_eq!((ebcs.functions[1])(0.0), 2.0);
    }

    #[test]
    fn set_homogeneous_works() {
        let mut ebcs = EssentialBcs1d::new();
        ebcs.set_periodic(true); // should be reset
        ebcs.set_homogeneous();
        assert!(!ebcs.periodic_along_x);
        assert!(ebcs.sides[0]);
        assert!(ebcs.sides[1]);
        assert_eq!((ebcs.functions[0])(123.0), 0.0);
        assert_eq!((ebcs.functions[1])(123.0), 0.0);
    }

    #[test]
    fn validate_works() {
        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();

        // 1. Missing BCs
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmin side is missing either EBC or NBC")
        );

        ebcs.set(Side::Xmin, |_| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmax side is missing either EBC or NBC")
        );

        // 2. Valid configuration (one EBC, one NBC)
        nbcs.set(Side::Xmax, |_| 0.0);
        assert_eq!(ebcs.validate(&nbcs), Ok(()));

        // 3. Both EBC and NBC on same side
        ebcs.set(Side::Xmax, |_| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Xmax side must not have both EBC and NBC")
        );

        // 4. Periodic with NBC
        let mut ebcs = EssentialBcs1d::new();
        let mut nbcs = NaturalBcs1d::new();
        ebcs.set_periodic(true);
        nbcs.set(Side::Xmin, |_| 0.0);
        assert_eq!(
            ebcs.validate(&nbcs).err(),
            Some("Periodic X does not allow NBC on Xmin or Xmax")
        );

        // 5. Periodic without NBC (Valid)
        let nbcs = NaturalBcs1d::new();
        assert_eq!(ebcs.validate(&nbcs), Ok(()));
    }

    #[test]
    fn get_nodes_works() {
        let mut ebcs = EssentialBcs1d::new();
        let grid = Grid1d::new(&[0.0, 0.5, 1.0]).unwrap(); // nodes: 0, 1, 2

        assert_eq!(ebcs.get_nodes(&grid).len(), 0);

        ebcs.set(Side::Xmin, |_| 0.0);
        assert_eq!(ebcs.get_nodes(&grid), &[0]);

        ebcs.set(Side::Xmax, |_| 0.0);
        assert_eq!(ebcs.get_nodes(&grid), &[0, 2]);
    }
}
