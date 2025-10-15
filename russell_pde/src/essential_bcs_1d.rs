use crate::{Grid1d, Side};
use std::collections::HashMap;
use std::sync::Arc;

/// Implements a handler for essential (Dirichlet) boundary conditions
///
/// This struct helps to manage essential boundary conditions (ebc) for 1D problems.
/// It holds the number of prescribed equations and the number of unknown equations.
///
/// The grid is assumed to be a regular Cartesian grid with `nx` points along x.
pub struct EssentialBcs1d<'a> {
    /// Indicates that the boundary is periodic along x (left ϕ values equal right ϕ values)
    ///
    /// If false, the left/right boundaries are zero-flux (Neumann with ∂ϕ/dx = 0)
    periodic_along_x: bool,

    /// Holds the functions to compute essential boundary conditions (ebc)
    ///
    /// The function is `f(x) -> ebc`
    ///
    /// (2) → (xmin, xmax); corresponding to the 2 sides
    functions: Vec<Arc<dyn Fn(f64) -> f64 + Send + Sync + 'a>>,

    /// Maps node to one of the two functions in `functions`
    ///
    /// length = number of nodes with essential boundary conditions (prescribed nodes)
    node_to_function: HashMap<usize, usize>,
}

impl<'a> EssentialBcs1d<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        EssentialBcs1d {
            periodic_along_x: false,
            functions: vec![
                Arc::new(|_| 0.0), // xmin
                Arc::new(|_| 0.0), // xmax
            ],
            node_to_function: HashMap::new(),
        }
    }

    // --------------------------------------------------------
    // setters
    // --------------------------------------------------------

    /// Sets periodic boundary condition
    ///
    /// **Note:** Any essential boundary condition on the corresponding side will be removed.
    pub fn set_periodic(&mut self, grid: &Grid1d, along_x: bool) {
        self.periodic_along_x = along_x;
        if along_x {
            self.node_to_function.remove(&grid.node_xmin());
            self.node_to_function.remove(&grid.node_xmax());
        }
    }

    /// Sets essential (Dirichlet) boundary condition
    ///
    /// The function is `f(x) -> ebc`
    ///
    /// **Note:** Any periodic boundary condition on the corresponding side will be removed.
    ///
    /// # Panics
    ///
    /// A panic may occur if an invalid side is provided for a 1D grid. It must be
    /// either `Side::Xmin` or `Side::Xmax`.
    pub fn set(&mut self, grid: &Grid1d, side: Side, f: impl Fn(f64) -> f64 + Send + Sync + 'a) {
        match side {
            Side::Xmin => {
                self.periodic_along_x = false;
                self.functions[0] = Arc::new(f);
                self.node_to_function.insert(grid.node_xmin(), 0);
            }
            Side::Xmax => {
                self.periodic_along_x = false;
                self.functions[1] = Arc::new(f);
                self.node_to_function.insert(grid.node_xmax(), 1);
            }
            _ => panic!("EssentialBcs1d::set: Invalid side for 1D grid"),
        };
    }

    /// Sets homogeneous boundary conditions (i.e., zero essential values at the borders)
    ///
    /// **Note:** Periodic boundary conditions will be removed.
    pub fn set_homogeneous(&mut self, grid: &Grid1d) {
        self.periodic_along_x = false;
        self.node_to_function.clear();
        self.functions = vec![
            Arc::new(|_| 0.0), // xmin
            Arc::new(|_| 0.0), // xmax
        ];
        self.node_to_function.insert(grid.node_xmin(), 0);
        self.node_to_function.insert(grid.node_xmax(), 1);
    }

    /// Indicates whether the boundary conditions are periodic along x
    pub fn is_periodic_along_x(&self) -> bool {
        self.periodic_along_x
    }

    /// Returns the prescribed value for the given node
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of bounds.
    pub fn get_prescribed_value(&self, m: usize, x: f64) -> f64 {
        let index = self.node_to_function.get(&m).unwrap();
        (self.functions[*index])(x)
    }

    /// Returns the list of nodes with essential boundary conditions (prescribed nodes)
    pub fn get_p_list(&self) -> Vec<usize> {
        self.node_to_function.keys().copied().collect()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EssentialBcs1d;
    use crate::{Grid1d, Side};

    const LEF: f64 = 1.0;
    const RIG: f64 = 2.0;

    #[test]
    fn new_default_and_set_periodic_work() {
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.functions.len(), 2);
        assert_eq!(ebcs.node_to_function.len(), 0);

        // Test default functions return zero
        for i in 0..2 {
            assert_eq!((ebcs.functions[i])(0.0), 0.0);
            assert_eq!((ebcs.functions[i])(1.5), 0.0);
        }

        ebcs.set_periodic(&grid, true);
        assert_eq!(ebcs.periodic_along_x, true);

        ebcs.set_periodic(&grid, false);
        assert_eq!(ebcs.periodic_along_x, false);
    }

    #[test]
    fn all_functionality_works() {
        // --- default: no essential boundary conditions ---
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();
        assert_eq!(ebcs.periodic_along_x, false);
        assert_eq!(ebcs.functions.len(), 2);
        assert_eq!(ebcs.node_to_function.len(), 0);

        // -- set essential boundary conditions on both sides --
        let lef = |_| LEF;
        let rig = |_| RIG;

        ebcs.set(&grid, Side::Xmin, lef);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0]); // left boundary node

        ebcs.set(&grid, Side::Xmax, rig);
        let mut res: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        res.sort();
        assert_eq!(res, vec![0, 3]); // both boundary nodes

        // --- check getters ---
        assert_eq!(ebcs.is_periodic_along_x(), false);
    }

    #[test]
    fn set_homogeneous_works() {
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // First set some non-homogeneous conditions
        ebcs.set(&grid, Side::Xmin, |_| 10.0);
        ebcs.set_periodic(&grid, true);
        assert_eq!(ebcs.periodic_along_x, true);

        // Set homogeneous - should clear periodic and set both boundaries to zero
        ebcs.set_homogeneous(&grid);

        // Check that periodic flag is cleared
        assert_eq!(ebcs.periodic_along_x, false);

        // Check that both boundary nodes are prescribed
        let mut prescribed_nodes: Vec<_> = ebcs.node_to_function.keys().copied().collect();
        prescribed_nodes.sort();
        assert_eq!(prescribed_nodes, vec![0, 3]); // left and right boundaries

        // Check that both functions return 0.0
        for i in 0..2 {
            assert_eq!((ebcs.functions[i])(1.5), 0.0);
        }

        // Check prescribed values at boundaries
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // left boundary
        assert_eq!(ebcs.get_prescribed_value(3, 3.0), 0.0); // right boundary
    }

    #[test]
    fn get_prescribed_value_works() {
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set boundary conditions with coordinate-dependent functions
        ebcs.set(&grid, Side::Xmin, |x| x * x); // left: f(x) = x²
        ebcs.set(&grid, Side::Xmax, |x| 2.0 * x + 1.0); // right: f(x) = 2x + 1

        // Test left boundary (x = 0)
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // 0² = 0

        // Test right boundary (x = 3)
        assert_eq!(ebcs.get_prescribed_value(3, 3.0), 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn get_p_list_works() {
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Initially no prescribed nodes
        assert_eq!(ebcs.get_p_list().len(), 0);

        // Set left boundary
        ebcs.set(&grid, Side::Xmin, |_| 1.0);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0]);

        // Add right boundary
        ebcs.set(&grid, Side::Xmax, |_| 2.0);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 3]);
    }

    #[test]
    fn periodic_boundary_conditions_work() {
        // 1D grid: 0---1---2---3
        let grid = Grid1d::new_uniform(0.0, 3.0, 4).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set some essential BCs first
        ebcs.set(&grid, Side::Xmin, |_| 1.0);
        ebcs.set(&grid, Side::Xmax, |_| 2.0);
        assert_eq!(ebcs.node_to_function.len(), 2); // both boundaries

        // Set periodic along x - should remove xmin/xmax prescribed nodes
        ebcs.set_periodic(&grid, true);
        assert!(ebcs.is_periodic_along_x());

        // Should have removed all boundary nodes
        let p_list = ebcs.get_p_list();
        assert_eq!(p_list.len(), 0);

        // Set non-periodic again and add BCs back
        ebcs.set(&grid, Side::Xmin, |_| 1.0);
        ebcs.set_periodic(&grid, false);
        assert!(!ebcs.is_periodic_along_x());

        // Should still have the left boundary BC
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0]);
    }

    #[test]
    fn essential_bc_overrides_periodic() {
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set periodic along x
        ebcs.set_periodic(&grid, true);
        assert!(ebcs.is_periodic_along_x());

        // Set essential BC on left - should disable periodic along x
        ebcs.set(&grid, Side::Xmin, |_| 5.0);
        assert!(!ebcs.is_periodic_along_x());

        // Set periodic again
        ebcs.set_periodic(&grid, true);
        assert!(ebcs.is_periodic_along_x());

        // Set essential BC on right - should disable periodic along x
        ebcs.set(&grid, Side::Xmax, |_| 10.0);
        assert!(!ebcs.is_periodic_along_x());
    }

    #[test]
    fn single_side_boundary_conditions() {
        // Test setting BCs on individual sides
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Only left boundary
        ebcs.set(&grid, Side::Xmin, |x| x);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0]); // left node

        // Clear and set only right boundary
        ebcs = EssentialBcs1d::new();
        ebcs.set(&grid, Side::Xmax, |x| 2.0 * x);
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![2]); // right node
    }

    #[test]
    fn complex_boundary_functions() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set complex functions on each boundary
        use std::f64::consts::PI;

        ebcs.set(&grid, Side::Xmin, |x| (PI * x).sin()); // sin(πx) on left
        ebcs.set(&grid, Side::Xmax, |x| (PI * x).cos()); // cos(πx) on right

        // Test prescribed values
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // sin(0) = 0
        assert_eq!(ebcs.get_prescribed_value(2, 1.0), (PI).cos()); // cos(π) = -1
        assert!((ebcs.get_prescribed_value(2, 1.0) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn boundary_conditions_on_small_grid() {
        // Test with minimal 2-node 1D grid
        // 0---1
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        ebcs.set_homogeneous(&grid);

        // Both nodes should be prescribed (boundary nodes only in 1D)
        let mut p_list = ebcs.get_p_list();
        p_list.sort();
        assert_eq!(p_list, vec![0, 1]);

        // All values should be zero
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0);
        assert_eq!(ebcs.get_prescribed_value(1, 1.0), 0.0);
    }

    #[test]
    fn updating_boundary_conditions() {
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set initial BC
        ebcs.set(&grid, Side::Xmin, |_| 1.0);
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 1.0);

        // Update with new function
        ebcs.set(&grid, Side::Xmin, |x| 2.0 * x);
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // 2 * 0 = 0

        // Add right boundary and then clear with homogeneous
        ebcs.set(&grid, Side::Xmax, |_| 5.0);

        ebcs.set_homogeneous(&grid);

        // All should now be zero
        for &node in &[0, 2] {
            // boundary nodes
            let x = if node == 0 { 0.0 } else { 2.0 };
            assert_eq!(ebcs.get_prescribed_value(node, x), 0.0);
        }
    }

    #[test]
    fn periodic_and_essential_conditions() {
        let grid = Grid1d::new_uniform(0.0, 4.0, 5).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set periodic along x
        ebcs.set_periodic(&grid, true);
        assert!(ebcs.is_periodic_along_x());

        // Should have no prescribed nodes when periodic
        assert_eq!(ebcs.get_p_list().len(), 0);

        // Set essential BC - should disable periodic
        ebcs.set(&grid, Side::Xmin, |x| x);
        assert!(!ebcs.is_periodic_along_x());

        // Should now have prescribed node
        let p_list = ebcs.get_p_list();
        assert_eq!(p_list, vec![0]);

        // Test prescribed value
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // f(0) = 0
    }

    #[test]
    #[should_panic]
    fn get_prescribed_value_panics_on_non_prescribed_node() {
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set only left boundary
        ebcs.set(&grid, Side::Xmin, |_| 1.0);

        // Try to get prescribed value for non-prescribed node (should panic)
        let _ = ebcs.get_prescribed_value(1, 1.0); // interior node
    }

    #[test]
    #[should_panic(expected = "EssentialBcs1d::set: Invalid side for 1D grid")]
    fn set_panics_on_invalid_side_ymin() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Try to set Y boundary on 1D grid (should panic)
        ebcs.set(&grid, Side::Ymin, |_| 1.0);
    }

    #[test]
    #[should_panic(expected = "EssentialBcs1d::set: Invalid side for 1D grid")]
    fn set_panics_on_invalid_side_ymax() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 2).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Try to set Y boundary on 1D grid (should panic)
        ebcs.set(&grid, Side::Ymax, |_| 1.0);
    }

    #[test]
    fn coordinate_dependent_functions_precision() {
        let grid = Grid1d::new_uniform(0.0, 1.0, 101).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set precise mathematical functions
        ebcs.set(&grid, Side::Xmin, |x| x * x * x); // x³ on left
        ebcs.set(&grid, Side::Xmax, |x| (x * std::f64::consts::PI).sin()); // sin(πx) on right

        // Test at specific coordinates
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 0.0); // 0³ = 0

        let expected_right = (1.0 * std::f64::consts::PI).sin(); // sin(π) = 0
        assert!((ebcs.get_prescribed_value(100, 1.0) - expected_right).abs() < 1e-15);
    }

    #[test]
    fn function_closure_captures() {
        let grid = Grid1d::new_uniform(0.0, 2.0, 3).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Test that closures can capture variables
        let multiplier = 3.0;
        let offset = 1.5;

        ebcs.set(&grid, Side::Xmin, move |x| multiplier * x + offset);
        ebcs.set(&grid, Side::Xmax, move |x| multiplier * x - offset);

        // Test captured values
        assert_eq!(ebcs.get_prescribed_value(0, 0.0), 1.5); // 3*0 + 1.5
        assert_eq!(ebcs.get_prescribed_value(2, 2.0), 4.5); // 3*2 - 1.5
    }

    #[test]
    fn mixed_boundary_types() {
        let grid = Grid1d::new_uniform(-1.0, 1.0, 5).unwrap();
        let mut ebcs = EssentialBcs1d::new();

        // Set different function types on each side
        ebcs.set(&grid, Side::Xmin, |_| 5.0); // constant on left
        ebcs.set(&grid, Side::Xmax, |x| x * x); // quadratic on right

        let p_list = ebcs.get_p_list();
        assert_eq!(p_list.len(), 2);

        // Test different function behaviors
        assert_eq!(ebcs.get_prescribed_value(0, -1.0), 5.0); // constant
        assert_eq!(ebcs.get_prescribed_value(4, 1.0), 1.0); // 1² = 1
    }
}
