use std::collections::HashSet;

/// Implements a tool to handle the equation numbering such as unknown and prescribed equations due to the essential boundary conditions.
///
/// ## Essential boundary conditions (EBC) handling
///
/// Two approaches are considered for handling the essential boundary conditions:
///
/// 1. System partitioning strategy (SPS)
/// 2. Lagrange multipliers method (LMM)
///
/// ### Approach 1: System partitioning strategy (SPS)
///
/// Consider the following partitioning of the vectors `a` and `f` and the matrix `K`:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ K̄   Ǩ │ │ ̄a │   │ f̄ │
/// │       │ │   │ = │   │
/// │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
/// └       ┘ └   ┘   └   ┘
///     K       a       f
/// ```
///
/// where `ā` (a-bar) is a reduced vector containing only the unknown values (i.e., non-EBC nodes), and `ǎ` (a-check)
/// is a reduced vector containing only the prescribed values (i.e., EBC nodes). `f̄` and `f̌` are the associated reduced
/// right-hand side vectors. The `K̄` (K-bar) matrix is the reduced discrete Laplacian operator and `Ǩ` (K-check) is a
/// *correction* matrix. The `Ḵ` (K-underline) and `K̰` (K-under-tilde) matrices are often not needed.
///
/// Thus, the linear system to be solved is:
///
/// ```text
/// K̄ ā = f̄ - Ǩ ǎ
/// ```
///
/// If needed, the other right-hand side values can be post-calculated by means of
///
/// ```text
/// f̌ = Ḵ ā + K̰ ǎ
/// ```
///
/// ### Approach 2: Lagrange multipliers method (LMM)
///
/// The LMM consists of augmenting the original linear system with additional equations:
///
/// ```text
/// ┌       ┐ ┌   ┐   ┌   ┐
/// │ K  Cᵀ │ │ a │   │ f │
/// │       │ │   │ = │   │
/// │ C  0  │ │ ℓ │   │ ǎ │
/// └       ┘ └   ┘   └   ┘
///     M       A       F
/// ```
///
/// where `ℓ` is the vector of Lagrange multipliers, `C` is the constraints matrix, and `ǎ` is the vector of
/// prescribed values at EBC nodes. The constraints matrix `C` has a row for each EBC (prescribed) node and a column
/// for every node. Each row in `C` has a single `1` at the column corresponding to the EBC node, and `0`s elsewhere.
///
/// ## Constants and definitions
///
/// This struct helps managing the indices associated with `ā` (a-bar; unknown) and `ǎ` (a-check; prescribed).
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
/// Notation:
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
    /// * `p_list` - list of global IDs of the prescribed equations (may have duplicates).
    ///
    /// # Panics
    ///
    /// Panics if any prescribed equation index is out of bounds.
    pub fn recompute(&mut self, p_list: &[usize]) {
        let mut p_set = HashSet::new();
        for p in p_list {
            if *p >= self.neq {
                panic!("prescribed equation index is out of bounds");
            }
            p_set.insert(*p);
        }
        self.u_sorted.clear();
        self.p_sorted.clear();
        let mut iu = 0;
        let mut ip = 0;
        for e in 0..self.neq {
            if p_set.contains(&e) {
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

    /// Indicates whether a node has an unknown value or not
    ///
    /// # Panics
    ///
    /// Panics if the global equation ID is out of bounds.
    pub fn is_unknown(&self, e: usize) -> bool {
        !self.is_prescribed[e]
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

        // Set equations 0 and 3 as prescribed
        let p_list = &[0, 3];
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
        let p_list = &[0, 1, 2];
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
        let p_list = &[1, 3];
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

        // Include same equation multiple times (duplicates should be ignored)
        let p_list = &[1, 1, 3, 1];
        handler.recompute(p_list);

        assert_eq!(handler.nu(), 2); // equations 0, 2
        assert_eq!(handler.np(), 2); // equations 1, 3

        assert_eq!(handler.unknown(), &vec![0, 2]);
        assert_eq!(handler.prescribed(), &vec![1, 3]);
    }

    #[test]
    #[should_panic(expected = "prescribed equation index is out of bounds")]
    fn recompute_panics_on_invalid_indices() {
        let mut handler = EquationHandler::new(3);

        // Try to set equation beyond bounds
        let p_list = &[0, 5]; // 5 is out of bounds
        handler.recompute(p_list);
    }

    #[test]
    #[should_panic(expected = "prescribed equation index is out of bounds")]
    fn recompute_panics_on_max_usize() {
        let mut handler = EquationHandler::new(3);

        // Try with maximum usize
        let p_list = &[usize::MAX];
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
        handler.recompute(&[1]);

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
        handler.recompute(&[0, 2]);

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
        handler.recompute(&[0, 2]);

        // Another unknown equation should panic
        let _ = handler.ip(3);
    }

    #[test]
    fn documentation_example_works() {
        // Test the example from the documentation
        let mut handler = EquationHandler::new(6);

        // Set equations 0 and 3 as prescribed (p)
        let p_list = &[0, 3];
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
        handler.recompute(&[1, 4, 7]);
        assert_eq!(handler.nu(), 5);
        assert_eq!(handler.np(), 3);
        assert_eq!(handler.unknown(), &vec![0, 2, 3, 5, 6]);
        assert_eq!(handler.prescribed(), &vec![1, 4, 7]);

        // Stage 2: Change prescribed equations completely
        handler.recompute(&[0, 2, 3, 6]);
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
    fn consistency_checks() {
        let mut handler = EquationHandler::new(10);

        // Set some prescribed equations
        handler.recompute(&[2, 5, 8]);

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
        handler.recompute(&[7, 2, 9, 1, 5]);

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
    fn prescribed_set_handles_duplicates() {
        let mut handler = EquationHandler::new(4);

        // Set same equation multiple times
        let p_list = &[1, 1, 3, 1, 3];
        handler.recompute(p_list);

        // Results should be the same as if no duplicates
        assert_eq!(handler.nu(), 2); // equations 0, 2
        assert_eq!(handler.np(), 2); // equations 1, 3
        assert_eq!(handler.unknown(), &vec![0, 2]);
        assert_eq!(handler.prescribed(), &vec![1, 3]);
    }

    #[test]
    fn comprehensive_boundary_condition_example() {
        let mut handler = EquationHandler::new(8);

        // Complex scenario: multiple boundary nodes
        let boundary_nodes = &[0, 1, 6, 7, 2, 5]; // various boundary positions
        handler.recompute(boundary_nodes);

        // Check that all boundary nodes are prescribed
        for &node in boundary_nodes {
            assert!(handler.is_prescribed(node));
        }

        // Interior nodes should be unknown
        for &interior in &[3, 4] {
            assert!(!handler.is_prescribed(interior));
        }

        // Verify counts
        assert_eq!(handler.np(), 6); // 6 prescribed equations
        assert_eq!(handler.nu(), 2); // 2 unknown equations (interior nodes)

        // Check sorted lists
        assert_eq!(handler.prescribed(), &vec![0, 1, 2, 5, 6, 7]);
        assert_eq!(handler.unknown(), &vec![3, 4]);
    }

    #[test]
    fn empty_system_edge_case() {
        let mut handler = EquationHandler::new(0);

        // Empty system - recompute with empty list should work
        handler.recompute(&[]);

        // No equations to work with
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.nu(), 0);
    }

    #[test]
    fn single_equation_system() {
        let mut handler = EquationHandler::new(1);

        // Single equation as prescribed
        handler.recompute(&[0]);
        assert!(handler.is_prescribed(0));
        assert_eq!(handler.np(), 1);
        assert_eq!(handler.nu(), 0);

        // Switch to unknown
        handler.recompute(&[]);
        assert!(!handler.is_prescribed(0));
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.nu(), 1);
    }

    #[test]
    fn recompute_clears_previous_state() {
        let mut handler = EquationHandler::new(6);

        // Initial state
        handler.recompute(&[0, 2, 4]);
        assert_eq!(handler.np(), 3);

        // New state should completely replace old state
        handler.recompute(&[1, 3]);
        assert_eq!(handler.np(), 2);

        // Old prescribed equations should no longer be prescribed
        assert!(!handler.is_prescribed(0));
        assert!(!handler.is_prescribed(2));
        assert!(!handler.is_prescribed(4));

        // New prescribed equations should be prescribed
        assert!(handler.is_prescribed(1));
        assert!(handler.is_prescribed(3));
    }

    #[test]
    fn is_unknown_works() {
        let mut handler = EquationHandler::new(4);
        handler.recompute(&[1]);
        assert!(handler.is_unknown(0));
        assert!(!handler.is_unknown(1));
        assert!(handler.is_unknown(2));
        assert!(handler.is_unknown(3));
    }

    #[test]
    #[should_panic]
    fn is_unknown_panics_on_out_of_bounds() {
        let handler = EquationHandler::new(3);
        let _ = handler.is_unknown(3);
    }
}
