use std::collections::HashMap;

/// Implements a tool to handle the equation numbering such as unknown and prescribed equations due to the essential boundary conditions.
///
/// Example:
///
/// ```text
///       | GLOBAL ID        | UNKNOWN ID | PRESCRIBED ID
///       | (e)              | (iu)       | (ip)
/// ------|------------------|------------|--------------
///       | p = 0 prescribed |            | 0
///       | u = 1            | 0          |
///       | u = 2            | 1          |
///       | p = 3 prescribed |            | 1
///       | u = 4            | 2          |
///       | u = 5            | 3          |
/// ------|------------------|------------|--------------
/// TOTAL | 6 equations      | 4 unknown  | 2 prescribed
/// ```
///
/// # Notation
///
/// * `neq`: total number of equations (= nu + np)
/// * `nu`: number of unknown equations
/// * `np`: number of prescribed equations (with essential boundary condition values)
/// * `e`: global index of an equation (0 ≤ e ≤ neq)
/// * `u`: global index of an unknown equation
/// * `p`: global index of a prescribed equation
/// * `iu`: local index of an unknown equation (0 ≤ iu < nu)
/// * `ip`: local index of a prescribed equation (0 ≤ ip < np)
pub struct EquationHandler {
    /// Holds the total number of equations (= nu + np)
    neq: usize,

    /// Maps the global ID of a prescribed equation (p) to some external ID
    ///
    /// For example, the external ID may be the function to calculate the essential boundary condition value.
    ///
    /// length = np
    p_to_external_id: HashMap<usize, usize>,

    /// Flags the prescribed equations
    ///
    /// length = neq
    is_prescribed: Vec<bool>,

    /// Maps the global equation ID (e) to the local unknown ID (iu)
    ///
    /// If the equation is prescribed, the value is set to `usize::MAX`.
    ///
    /// length = neq
    e_to_iu: Vec<usize>,

    /// Maps the global equation ID (e) to the local prescribed ID (ip)
    ///
    /// If the equation is unknown, the value is set to `usize::MAX`.
    ///
    /// length = neq
    e_to_ip: Vec<usize>,

    /// Holds the sorted global ID of the unknown equations (u)
    ///
    /// length = nu
    u_sorted: Vec<usize>,

    /// Holds the sorted global ID of the prescribed equations (p)
    p_sorted: Vec<usize>,
}

impl EquationHandler {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `neq` - total number of equations
    ///
    /// # Note
    ///
    /// Initially, all equations are considered unknown. To set prescribed equations,
    /// use the [EquationHandler::recompute] method.
    pub fn new(neq: usize) -> Self {
        let all: Vec<_> = (0..neq).collect(); // initially, all are unknown
        EquationHandler {
            neq,
            p_to_external_id: HashMap::new(),
            is_prescribed: vec![false; neq],
            e_to_iu: all.clone(),
            e_to_ip: vec![usize::MAX; neq],
            u_sorted: all,
            p_sorted: Vec::new(),
        }
    }

    /// Recomputes the internal arrays
    ///
    /// # Arguments
    ///
    /// * `p_list` - list of global IDs of the prescribed equations. The list holds the global ID
    ///   and the external ID; `(p, external_id)`. The external ID may be used to identify the
    ///   function to compute the essential boundary condition value.
    ///
    /// # Panics
    ///
    /// Panics if any prescribed equation index is out of bounds.
    pub fn recompute(&mut self, p_list: &[(usize, usize)]) {
        self.p_to_external_id.clear();
        for (p, external_id) in p_list {
            if *p >= self.neq {
                panic!("prescribed equation index is out of bounds");
            }
            self.p_to_external_id.insert(*p, *external_id);
        }
        self.u_sorted.clear();
        self.p_sorted.clear();
        let mut iu = 0;
        let mut ip = 0;
        for e in 0..self.neq {
            if self.p_to_external_id.contains_key(&e) {
                self.is_prescribed[e] = true;
                self.e_to_iu[e] = usize::MAX;
                self.e_to_ip[e] = ip;
                self.p_sorted.push(e);
                ip += 1;
            } else {
                self.is_prescribed[e] = false;
                self.e_to_iu[e] = iu;
                self.e_to_ip[e] = usize::MAX;
                self.u_sorted.push(e);
                iu += 1;
            }
        }
    }

    /// Returns the total number of equations
    ///
    /// Equals to `num_prescribed + num_unknown`.
    pub fn neq(&self) -> usize {
        self.neq
    }

    /// Returns the number of unknown equations
    pub fn nu(&self) -> usize {
        self.u_sorted.len()
    }

    /// Returns the number of prescribed equations
    pub fn np(&self) -> usize {
        self.p_sorted.len()
    }

    /// Indicates whether a node has a prescribed value or not
    ///
    /// # Panics
    ///
    /// Panics if the global equation ID is out of bounds.
    pub fn is_prescribed(&self, e: usize) -> bool {
        self.is_prescribed[e]
    }

    /// Returns the local index of the unknown equation
    ///
    /// # Panics
    ///
    /// Panics if the global equation ID does not correspond to an unknown equation.
    pub fn iu(&self, e: usize) -> usize {
        if self.e_to_iu[e] == usize::MAX {
            panic!("global equation ID does not correspond to an unknown equation");
        }
        self.e_to_iu[e]
    }

    /// Returns the local index of the prescribed equation
    ///
    /// # Panics
    ///
    /// Panics if the global equation ID does not correspond to a prescribed equation.
    pub fn ip(&self, e: usize) -> usize {
        if self.e_to_ip[e] == usize::MAX {
            panic!("global equation ID does not correspond to a prescribed equation");
        }
        self.e_to_ip[e]
    }

    /// Returns the external ID of a prescribed equation
    ///
    /// # Panics
    ///
    /// Panics if the global equation ID is not prescribed.
    pub fn external_id(&self, e: usize) -> usize {
        *self.p_to_external_id.get(&e).unwrap()
    }

    /// Returns an access to the (sorted) indices of the unknown equations
    pub fn unknown(&self) -> &Vec<usize> {
        &self.u_sorted
    }

    /// Returns an access to the (sorted) indices of the prescribed equations
    pub fn prescribed(&self) -> &Vec<usize> {
        &self.p_sorted
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EquationHandler;

    #[test]
    fn new_creates_correct_initial_state() {
        let neq = 6;
        let handler = EquationHandler::new(neq);

        assert_eq!(handler.neq(), 6);
        assert_eq!(handler.nu(), 6); // all unknown initially
        assert_eq!(handler.np(), 0); // no prescribed initially

        // Check that all equations are unknown initially
        for e in 0..neq {
            assert!(!handler.is_prescribed(e));
            assert_eq!(handler.iu(e), e);
        }

        // Check sorted lists
        assert_eq!(handler.unknown(), &vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(handler.prescribed(), &Vec::<usize>::new());
    }

    #[test]
    fn new_handles_edge_cases() {
        // Single equation
        let handler = EquationHandler::new(1);
        assert_eq!(handler.neq(), 1);
        assert_eq!(handler.nu(), 1);
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.unknown(), &vec![0]);

        // Zero equations (edge case)
        let handler = EquationHandler::new(0);
        assert_eq!(handler.neq(), 0);
        assert_eq!(handler.nu(), 0);
        assert_eq!(handler.np(), 0);
        assert!(handler.unknown().is_empty());
        assert!(handler.prescribed().is_empty());
    }

    #[test]
    fn recompute_works_with_prescribed_equations() {
        let mut handler = EquationHandler::new(6);

        // Set equations 0 and 3 as prescribed with external IDs 10 and 30
        let p_list = &[(0, 10), (3, 30)];
        handler.recompute(p_list);

        // Check counts
        assert_eq!(handler.neq(), 6);
        assert_eq!(handler.nu(), 4); // equations 1, 2, 4, 5
        assert_eq!(handler.np(), 2); // equations 0, 3

        // Check prescribed flags
        assert!(handler.is_prescribed(0));
        assert!(!handler.is_prescribed(1));
        assert!(!handler.is_prescribed(2));
        assert!(handler.is_prescribed(3));
        assert!(!handler.is_prescribed(4));
        assert!(!handler.is_prescribed(5));

        // Check unknown mappings
        assert_eq!(handler.iu(1), 0); // global 1 -> local unknown 0
        assert_eq!(handler.iu(2), 1); // global 2 -> local unknown 1
        assert_eq!(handler.iu(4), 2); // global 4 -> local unknown 2
        assert_eq!(handler.iu(5), 3); // global 5 -> local unknown 3

        // Check prescribed mappings
        assert_eq!(handler.ip(0), 0); // global 0 -> local prescribed 0
        assert_eq!(handler.ip(3), 1); // global 3 -> local prescribed 1

        // Check sorted lists
        assert_eq!(handler.unknown(), &vec![1, 2, 4, 5]);
        assert_eq!(handler.prescribed(), &vec![0, 3]);
    }

    #[test]
    fn recompute_handles_all_prescribed() {
        let mut handler = EquationHandler::new(3);

        // Set all equations as prescribed
        let p_list = &[(0, 100), (1, 200), (2, 300)];
        handler.recompute(p_list);

        assert_eq!(handler.nu(), 0);
        assert_eq!(handler.np(), 3);

        for e in 0..3 {
            assert!(handler.is_prescribed(e));
            assert_eq!(handler.ip(e), e);
        }

        assert!(handler.unknown().is_empty());
        assert_eq!(handler.prescribed(), &vec![0, 1, 2]);
    }

    #[test]
    fn recompute_handles_all_unknown() {
        let mut handler = EquationHandler::new(4);

        // First set some prescribed
        let p_list = &[(1, 10), (3, 30)];
        handler.recompute(p_list);
        assert_eq!(handler.np(), 2);

        // Then clear all prescribed (empty list)
        handler.recompute(&[]);

        assert_eq!(handler.nu(), 4);
        assert_eq!(handler.np(), 0);

        for e in 0..4 {
            assert!(!handler.is_prescribed(e));
            assert_eq!(handler.iu(e), e);
        }

        assert_eq!(handler.unknown(), &vec![0, 1, 2, 3]);
        assert!(handler.prescribed().is_empty());
    }

    #[test]
    fn recompute_handles_duplicate_prescriptions() {
        let mut handler = EquationHandler::new(4);

        // Include same equation multiple times (should use last external_id)
        let p_list = &[(1, 10), (1, 20), (3, 30)];
        handler.recompute(p_list);

        assert_eq!(handler.nu(), 2); // equations 0, 2
        assert_eq!(handler.np(), 2); // equations 1, 3

        assert_eq!(handler.unknown(), &vec![0, 2]);
        assert_eq!(handler.prescribed(), &vec![1, 3]);

        // External ID should be the last one specified
        assert_eq!(handler.p_to_external_id[&1], 20);
        assert_eq!(handler.p_to_external_id[&3], 30);
    }

    #[test]
    #[should_panic(expected = "prescribed equation index is out of bounds")]
    fn recompute_panics_on_invalid_indices() {
        let mut handler = EquationHandler::new(3);

        // Try to set equation beyond bounds
        let p_list = &[(0, 10), (5, 50)]; // 5 is out of bounds
        handler.recompute(p_list);
    }

    #[test]
    #[should_panic(expected = "prescribed equation index is out of bounds")]
    fn recompute_panics_on_max_usize() {
        let mut handler = EquationHandler::new(3);

        // Try with maximum usize
        let p_list = &[(usize::MAX, 10)];
        handler.recompute(p_list);
    }

    #[test]
    #[should_panic]
    fn is_prescribed_panics_on_out_of_bounds() {
        let handler = EquationHandler::new(3);
        let _ = handler.is_prescribed(3); // out of bounds
    }

    #[test]
    #[should_panic]
    fn is_prescribed_panics_on_large_index() {
        let handler = EquationHandler::new(3);
        let _ = handler.is_prescribed(100); // out of bounds
    }

    #[test]
    #[should_panic(expected = "global equation ID does not correspond to an unknown equation")]
    fn iu_panics_on_prescribed_equation() {
        let mut handler = EquationHandler::new(4);

        // Set equation 1 as prescribed
        handler.recompute(&[(1, 10)]);

        // Valid unknown equations work
        assert_eq!(handler.iu(0), 0);
        assert_eq!(handler.iu(2), 1);
        assert_eq!(handler.iu(3), 2);

        // Prescribed equation should panic
        let _ = handler.iu(1);
    }

    #[test]
    #[should_panic(expected = "global equation ID does not correspond to a prescribed equation")]
    fn ip_panics_on_unknown_equation() {
        let mut handler = EquationHandler::new(4);

        // Set equations 0 and 2 as prescribed
        handler.recompute(&[(0, 10), (2, 20)]);

        // Valid prescribed equations work
        assert_eq!(handler.ip(0), 0);
        assert_eq!(handler.ip(2), 1);

        // Unknown equation should panic
        let _ = handler.ip(1);
    }

    #[test]
    #[should_panic(expected = "global equation ID does not correspond to a prescribed equation")]
    fn ip_panics_on_another_unknown_equation() {
        let mut handler = EquationHandler::new(4);

        // Set equations 0 and 2 as prescribed
        handler.recompute(&[(0, 10), (2, 20)]);

        // Another unknown equation should panic
        let _ = handler.ip(3);
    }

    #[test]
    fn documentation_example_works() {
        // Test the example from the documentation
        let mut handler = EquationHandler::new(6);

        // Set equations 0 and 3 as prescribed (p)
        let p_list = &[(0, 100), (3, 200)];
        handler.recompute(p_list);

        // Verify the mapping from the documentation table:
        // GLOBAL ID | UNKNOWN ID | PRESCRIBED ID
        // 0 (p)     |            | 0
        // 1 (u)     | 0          |
        // 2 (u)     | 1          |
        // 3 (p)     |            | 1
        // 4 (u)     | 2          |
        // 5 (u)     | 3          |

        assert_eq!(handler.neq(), 6);
        assert_eq!(handler.nu(), 4);
        assert_eq!(handler.np(), 2);

        // Check prescribed equations
        assert!(handler.is_prescribed(0));
        assert!(!handler.is_prescribed(1));
        assert!(!handler.is_prescribed(2));
        assert!(handler.is_prescribed(3));
        assert!(!handler.is_prescribed(4));
        assert!(!handler.is_prescribed(5));

        // Check unknown mappings (global -> local unknown)
        assert_eq!(handler.iu(1), 0);
        assert_eq!(handler.iu(2), 1);
        assert_eq!(handler.iu(4), 2);
        assert_eq!(handler.iu(5), 3);

        // Check prescribed mappings (global -> local prescribed)
        assert_eq!(handler.ip(0), 0);
        assert_eq!(handler.ip(3), 1);

        // Check sorted lists
        assert_eq!(handler.unknown(), &vec![1, 2, 4, 5]);
        assert_eq!(handler.prescribed(), &vec![0, 3]);
    }

    #[test]
    fn complex_recompute_sequence() {
        let mut handler = EquationHandler::new(8);

        // Stage 1: Set some prescribed equations
        handler.recompute(&[(1, 10), (4, 40), (7, 70)]);
        assert_eq!(handler.nu(), 5);
        assert_eq!(handler.np(), 3);
        assert_eq!(handler.unknown(), &vec![0, 2, 3, 5, 6]);
        assert_eq!(handler.prescribed(), &vec![1, 4, 7]);

        // Stage 2: Change prescribed equations completely
        handler.recompute(&[(0, 5), (2, 25), (3, 35), (6, 65)]);
        assert_eq!(handler.nu(), 4);
        assert_eq!(handler.np(), 4);
        assert_eq!(handler.unknown(), &vec![1, 4, 5, 7]);
        assert_eq!(handler.prescribed(), &vec![0, 2, 3, 6]);

        // Stage 3: Remove all prescribed
        handler.recompute(&[]);
        assert_eq!(handler.nu(), 8);
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.unknown(), &vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(handler.prescribed().is_empty());

        // Verify all are unknown
        for e in 0..8 {
            assert!(!handler.is_prescribed(e));
            assert_eq!(handler.iu(e), e);
        }
    }

    #[test]
    fn external_id_mapping_works() {
        let mut handler = EquationHandler::new(5);

        // Use different external IDs
        let p_list = &[(1, 100), (3, 300), (4, 400)];
        handler.recompute(p_list);

        // Check that external IDs are stored correctly
        assert_eq!(handler.p_to_external_id[&1], 100);
        assert_eq!(handler.p_to_external_id[&3], 300);
        assert_eq!(handler.p_to_external_id[&4], 400);

        // Non-prescribed equations shouldn't be in the map
        assert!(!handler.p_to_external_id.contains_key(&0));
        assert!(!handler.p_to_external_id.contains_key(&2));

        // Update with different mapping
        handler.recompute(&[(0, 50), (2, 250)]);

        // Old mappings should be cleared
        assert!(!handler.p_to_external_id.contains_key(&1));
        assert!(!handler.p_to_external_id.contains_key(&3));
        assert!(!handler.p_to_external_id.contains_key(&4));

        // New mappings should be present
        assert_eq!(handler.p_to_external_id[&0], 50);
        assert_eq!(handler.p_to_external_id[&2], 250);
    }

    #[test]
    fn consistency_checks() {
        let mut handler = EquationHandler::new(10);

        // Set some prescribed equations
        handler.recompute(&[(2, 20), (5, 50), (8, 80)]);

        // Verify that nu + np = neq
        assert_eq!(handler.nu() + handler.np(), handler.neq());

        // Verify that all equations are either unknown or prescribed, but not both
        let mut all_equations = handler.unknown().clone();
        all_equations.extend(handler.prescribed());
        all_equations.sort();

        let expected: Vec<_> = (0..10).collect();
        assert_eq!(all_equations, expected);

        // Verify no overlap between unknown and prescribed
        for &u in handler.unknown() {
            assert!(!handler.prescribed().contains(&u));
        }
        for &p in handler.prescribed() {
            assert!(!handler.unknown().contains(&p));
        }

        // Verify mapping consistency
        for &u in handler.unknown() {
            assert!(!handler.is_prescribed(u));
            // iu should work for unknown equations
            handler.iu(u);
        }
        for &p in handler.prescribed() {
            assert!(handler.is_prescribed(p));
            // ip should work for prescribed equations
            handler.ip(p);
        }
    }

    #[test]
    fn sorted_lists_are_actually_sorted() {
        let mut handler = EquationHandler::new(10);

        // Set prescribed equations in non-sorted order
        handler.recompute(&[(7, 70), (2, 20), (9, 90), (1, 10), (5, 50)]);

        // Unknown list should be sorted
        let unknown = handler.unknown();
        assert_eq!(unknown, &vec![0, 3, 4, 6, 8]);
        assert!(unknown.windows(2).all(|w| w[0] < w[1]));

        // Prescribed list should be sorted
        let prescribed = handler.prescribed();
        assert_eq!(prescribed, &vec![1, 2, 5, 7, 9]);
        assert!(prescribed.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn external_id_works_correctly() {
        let mut handler = EquationHandler::new(6);

        // Set equations with specific external IDs
        let p_list = &[(0, 100), (3, 200), (5, 300)];
        handler.recompute(p_list);

        // Check that external IDs can be retrieved correctly
        assert_eq!(handler.external_id(0), 100);
        assert_eq!(handler.external_id(3), 200);
        assert_eq!(handler.external_id(5), 300);
    }

    #[test]
    #[should_panic]
    fn external_id_panics_on_unknown_equation() {
        let mut handler = EquationHandler::new(6);

        // Set equations with specific external IDs
        let p_list = &[(0, 100), (3, 200), (5, 300)];
        handler.recompute(p_list);

        // Unknown equation should panic when asking for external ID
        let _ = handler.external_id(1);
    }

    #[test]
    #[should_panic]
    fn external_id_panics_on_another_unknown_equation() {
        let mut handler = EquationHandler::new(6);

        // Set equations with specific external IDs
        let p_list = &[(0, 100), (3, 200), (5, 300)];
        handler.recompute(p_list);

        // Another unknown equation should panic
        let _ = handler.external_id(2);
    }

    #[test]
    #[should_panic]
    fn external_id_panics_on_third_unknown_equation() {
        let mut handler = EquationHandler::new(6);

        // Set equations with specific external IDs
        let p_list = &[(0, 100), (3, 200), (5, 300)];
        handler.recompute(p_list);

        // Third unknown equation should panic
        let _ = handler.external_id(4);
    }

    #[test]
    fn external_id_updates_correctly_after_recompute() {
        let mut handler = EquationHandler::new(5);

        // Initial configuration
        handler.recompute(&[(1, 100), (3, 300)]);
        assert_eq!(handler.external_id(1), 100);
        assert_eq!(handler.external_id(3), 300);

        // Update with different external IDs
        handler.recompute(&[(1, 150), (3, 350), (4, 450)]);
        assert_eq!(handler.external_id(1), 150); // updated
        assert_eq!(handler.external_id(3), 350); // updated
        assert_eq!(handler.external_id(4), 450); // new

        // Clear all prescribed equations
        handler.recompute(&[]);
        // Now all calls to external_id should panic
    }

    #[test]
    #[should_panic]
    fn external_id_panics_after_clearing_prescribed() {
        let mut handler = EquationHandler::new(5);

        // Set some prescribed equations
        handler.recompute(&[(1, 100), (3, 300)]);

        // Clear all prescribed equations
        handler.recompute(&[]);

        // Should panic since equation 1 is no longer prescribed
        let _ = handler.external_id(1);
    }

    #[test]
    fn external_id_handles_duplicate_equations_correctly() {
        let mut handler = EquationHandler::new(4);

        // Set same equation multiple times with different external IDs
        let p_list = &[(1, 100), (1, 200), (1, 300), (3, 400)];
        handler.recompute(p_list);

        // Should use the last external ID for equation 1
        assert_eq!(handler.external_id(1), 300);
        assert_eq!(handler.external_id(3), 400);
    }

    #[test]
    fn external_id_works_with_zero_external_ids() {
        let mut handler = EquationHandler::new(4);

        // Test with external ID of 0 (valid value)
        handler.recompute(&[(0, 0), (2, 0), (3, 100)]);

        assert_eq!(handler.external_id(0), 0);
        assert_eq!(handler.external_id(2), 0);
        assert_eq!(handler.external_id(3), 100);
    }

    #[test]
    #[should_panic]
    fn external_id_panics_for_unknown_with_zero_ids() {
        let mut handler = EquationHandler::new(4);

        // Test with external ID of 0 (valid value)
        handler.recompute(&[(0, 0), (2, 0), (3, 100)]);

        // Unknown equation should still panic
        let _ = handler.external_id(1);
    }

    #[test]
    fn external_id_works_with_large_external_ids() {
        let mut handler = EquationHandler::new(3);

        // Test with large external IDs
        let large_id = usize::MAX;
        handler.recompute(&[(0, large_id), (2, large_id - 1)]);

        assert_eq!(handler.external_id(0), large_id);
        assert_eq!(handler.external_id(2), large_id - 1);
    }

    #[test]
    fn external_id_consistency_with_other_methods() {
        let mut handler = EquationHandler::new(6);

        // Set prescribed equations with external IDs
        let p_list = &[(0, 10), (2, 20), (4, 40)];
        handler.recompute(p_list);

        // For each prescribed equation, external_id should work
        for &p in handler.prescribed() {
            let _external_id = handler.external_id(p); // should not panic
            assert!(handler.is_prescribed(p));
            let _ip = handler.ip(p); // should not panic
        }

        // For each unknown equation, external_id should panic (tested separately)
        for &u in handler.unknown() {
            assert!(!handler.is_prescribed(u));
            let _iu = handler.iu(u); // should not panic
        }

        // Verify specific external ID values match what was set
        assert_eq!(handler.external_id(0), 10);
        assert_eq!(handler.external_id(2), 20);
        assert_eq!(handler.external_id(4), 40);
    }

    #[test]
    fn external_id_comprehensive_example() {
        let mut handler = EquationHandler::new(8);

        // Complex scenario: multiple boundary condition types
        let bc_dirichlet_left = 1;
        let bc_dirichlet_right = 2;
        let bc_neumann_top = 3;
        let bc_robin_bottom = 4;

        let p_list = &[
            (0, bc_dirichlet_left),  // left boundary
            (1, bc_dirichlet_left),  // left boundary
            (6, bc_dirichlet_right), // right boundary
            (7, bc_dirichlet_right), // right boundary
            (2, bc_neumann_top),     // top boundary
            (5, bc_robin_bottom),    // bottom boundary
        ];

        handler.recompute(p_list);

        // Check boundary condition types
        assert_eq!(handler.external_id(0), bc_dirichlet_left);
        assert_eq!(handler.external_id(1), bc_dirichlet_left);
        assert_eq!(handler.external_id(6), bc_dirichlet_right);
        assert_eq!(handler.external_id(7), bc_dirichlet_right);
        assert_eq!(handler.external_id(2), bc_neumann_top);
        assert_eq!(handler.external_id(5), bc_robin_bottom);

        // Interior nodes should be unknown
        for &interior in &[3, 4] {
            assert!(!handler.is_prescribed(interior));
        }

        // Verify counts
        assert_eq!(handler.np(), 6); // 6 prescribed equations
        assert_eq!(handler.nu(), 2); // 2 unknown equations (interior nodes)
    }

    #[test]
    fn external_id_empty_system_edge_case() {
        let mut handler = EquationHandler::new(0);

        // Empty system - recompute with empty list should work
        handler.recompute(&[]);

        // No equations to query external IDs for
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.nu(), 0);
    }

    #[test]
    fn external_id_single_equation_system() {
        let mut handler = EquationHandler::new(1);

        // Single equation as prescribed
        handler.recompute(&[(0, 42)]);
        assert_eq!(handler.external_id(0), 42);

        // Switch to unknown - external_id should now panic
        handler.recompute(&[]);
        // external_id(0) would panic now, but we test this separately
    }

    #[test]
    #[should_panic]
    fn external_id_panics_after_switching_to_unknown() {
        let mut handler = EquationHandler::new(1);

        // Single equation as prescribed
        handler.recompute(&[(0, 42)]);

        // Switch to unknown
        handler.recompute(&[]);

        // Should panic since equation 0 is no longer prescribed
        let _ = handler.external_id(0);
    }

    #[test]
    fn large_system_performance() {
        let neq = 10000;
        let mut handler = EquationHandler::new(neq);

        // Set every 10th equation as prescribed
        let p_list: Vec<_> = (0..neq).step_by(10).map(|i| (i, i * 10)).collect();
        handler.recompute(&p_list);

        assert_eq!(handler.neq(), 10000);
        assert_eq!(handler.np(), 1000); // every 10th
        assert_eq!(handler.nu(), 9000); // remaining

        // Check a few mappings
        assert!(handler.is_prescribed(0));
        assert!(!handler.is_prescribed(1));
        assert!(handler.is_prescribed(10));
        assert!(!handler.is_prescribed(11));

        // Check that unknown list has correct size and is sorted
        assert_eq!(handler.unknown().len(), 9000);
        assert!(handler.unknown().windows(2).all(|w| w[0] < w[1])); // sorted

        // Check that prescribed list has correct size and is sorted
        assert_eq!(handler.prescribed().len(), 1000);
        assert!(handler.prescribed().windows(2).all(|w| w[0] < w[1])); // sorted

        // Test external_id for some prescribed equations
        assert_eq!(handler.external_id(0), 0);
        assert_eq!(handler.external_id(10), 100);
        assert_eq!(handler.external_id(20), 200);
    }
}
