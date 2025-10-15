use crate::StrError;
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
    pub fn recompute(&mut self, p_list: &[(usize, usize)]) -> Result<(), StrError> {
        self.p_to_external_id.clear();
        for (p, external_id) in p_list {
            if *p >= self.neq {
                return Err("prescribed equation index out of bounds");
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
        Ok(())
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
    pub fn is_prescribed(&self, e: usize) -> Result<bool, StrError> {
        if e >= self.neq {
            Err("global equation ID is out of bounds")
        } else {
            Ok(self.is_prescribed[e])
        }
    }

    /// Returns the local index of the unknown node
    pub fn iu(&self, e: usize) -> Result<usize, StrError> {
        if self.e_to_iu[e] == usize::MAX {
            Err("global equation ID does not correspond to an unknown equation")
        } else {
            Ok(self.e_to_iu[e])
        }
    }

    /// Returns the local index of the prescribed node
    pub fn ip(&self, e: usize) -> Result<usize, StrError> {
        if self.e_to_ip[e] == usize::MAX {
            Err("global equation ID does not correspond to a prescribed equation")
        } else {
            Ok(self.e_to_ip[e])
        }
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
            assert!(!handler.is_prescribed(e).unwrap());
            assert_eq!(handler.iu(e).unwrap(), e);
            assert!(handler.ip(e).is_err());
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
        handler.recompute(p_list).unwrap();

        // Check counts
        assert_eq!(handler.neq(), 6);
        assert_eq!(handler.nu(), 4); // equations 1, 2, 4, 5
        assert_eq!(handler.np(), 2); // equations 0, 3

        // Check prescribed flags
        assert!(handler.is_prescribed(0).unwrap());
        assert!(!handler.is_prescribed(1).unwrap());
        assert!(!handler.is_prescribed(2).unwrap());
        assert!(handler.is_prescribed(3).unwrap());
        assert!(!handler.is_prescribed(4).unwrap());
        assert!(!handler.is_prescribed(5).unwrap());

        // Check unknown mappings
        assert_eq!(handler.iu(1).unwrap(), 0); // global 1 -> local unknown 0
        assert_eq!(handler.iu(2).unwrap(), 1); // global 2 -> local unknown 1
        assert_eq!(handler.iu(4).unwrap(), 2); // global 4 -> local unknown 2
        assert_eq!(handler.iu(5).unwrap(), 3); // global 5 -> local unknown 3

        // Check prescribed mappings
        assert_eq!(handler.ip(0).unwrap(), 0); // global 0 -> local prescribed 0
        assert_eq!(handler.ip(3).unwrap(), 1); // global 3 -> local prescribed 1

        // Check sorted lists
        assert_eq!(handler.unknown(), &vec![1, 2, 4, 5]);
        assert_eq!(handler.prescribed(), &vec![0, 3]);
    }

    #[test]
    fn recompute_handles_all_prescribed() {
        let mut handler = EquationHandler::new(3);

        // Set all equations as prescribed
        let p_list = &[(0, 100), (1, 200), (2, 300)];
        handler.recompute(p_list).unwrap();

        assert_eq!(handler.nu(), 0);
        assert_eq!(handler.np(), 3);

        for e in 0..3 {
            assert!(handler.is_prescribed(e).unwrap());
            assert!(handler.iu(e).is_err());
            assert_eq!(handler.ip(e).unwrap(), e);
        }

        assert!(handler.unknown().is_empty());
        assert_eq!(handler.prescribed(), &vec![0, 1, 2]);
    }

    #[test]
    fn recompute_handles_all_unknown() {
        let mut handler = EquationHandler::new(4);

        // First set some prescribed
        let p_list = &[(1, 10), (3, 30)];
        handler.recompute(p_list).unwrap();
        assert_eq!(handler.np(), 2);

        // Then clear all prescribed (empty list)
        handler.recompute(&[]).unwrap();

        assert_eq!(handler.nu(), 4);
        assert_eq!(handler.np(), 0);

        for e in 0..4 {
            assert!(!handler.is_prescribed(e).unwrap());
            assert_eq!(handler.iu(e).unwrap(), e);
            assert!(handler.ip(e).is_err());
        }

        assert_eq!(handler.unknown(), &vec![0, 1, 2, 3]);
        assert!(handler.prescribed().is_empty());
    }

    #[test]
    fn recompute_handles_duplicate_prescriptions() {
        let mut handler = EquationHandler::new(4);

        // Include same equation multiple times (should use last external_id)
        let p_list = &[(1, 10), (1, 20), (3, 30)];
        handler.recompute(p_list).unwrap();

        assert_eq!(handler.nu(), 2); // equations 0, 2
        assert_eq!(handler.np(), 2); // equations 1, 3

        assert_eq!(handler.unknown(), &vec![0, 2]);
        assert_eq!(handler.prescribed(), &vec![1, 3]);

        // External ID should be the last one specified
        assert_eq!(handler.p_to_external_id[&1], 20);
        assert_eq!(handler.p_to_external_id[&3], 30);
    }

    #[test]
    fn recompute_fails_on_invalid_indices() {
        let mut handler = EquationHandler::new(3);

        // Try to set equation beyond bounds
        let p_list = &[(0, 10), (5, 50)]; // 5 is out of bounds
        assert_eq!(
            handler.recompute(p_list).err(),
            Some("prescribed equation index out of bounds")
        );

        // Try with maximum usize
        let p_list = &[(usize::MAX, 10)];
        assert_eq!(
            handler.recompute(p_list).err(),
            Some("prescribed equation index out of bounds")
        );
    }

    #[test]
    fn is_prescribed_handles_bounds() {
        let handler = EquationHandler::new(3);

        // Valid indices
        assert!(!handler.is_prescribed(0).unwrap());
        assert!(!handler.is_prescribed(2).unwrap());

        // Invalid indices
        assert_eq!(
            handler.is_prescribed(3).err(),
            Some("global equation ID is out of bounds")
        );
        assert_eq!(
            handler.is_prescribed(100).err(),
            Some("global equation ID is out of bounds")
        );
    }

    #[test]
    fn iu_handles_prescribed_equations() {
        let mut handler = EquationHandler::new(4);

        // Set equation 1 as prescribed
        handler.recompute(&[(1, 10)]).unwrap();

        // Valid unknown equations
        assert_eq!(handler.iu(0).unwrap(), 0);
        assert_eq!(handler.iu(2).unwrap(), 1);
        assert_eq!(handler.iu(3).unwrap(), 2);

        // Prescribed equation should error
        assert_eq!(
            handler.iu(1).err(),
            Some("global equation ID does not correspond to an unknown equation")
        );
    }

    #[test]
    fn ip_handles_unknown_equations() {
        let mut handler = EquationHandler::new(4);

        // Set equations 0 and 2 as prescribed
        handler.recompute(&[(0, 10), (2, 20)]).unwrap();

        // Valid prescribed equations
        assert_eq!(handler.ip(0).unwrap(), 0);
        assert_eq!(handler.ip(2).unwrap(), 1);

        // Unknown equations should error
        assert_eq!(
            handler.ip(1).err(),
            Some("global equation ID does not correspond to a prescribed equation")
        );
        assert_eq!(
            handler.ip(3).err(),
            Some("global equation ID does not correspond to a prescribed equation")
        );
    }

    #[test]
    fn documentation_example_works() {
        // Test the example from the documentation
        let mut handler = EquationHandler::new(6);

        // Set equations 0 and 3 as prescribed (p)
        let p_list = &[(0, 100), (3, 200)];
        handler.recompute(p_list).unwrap();

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
        assert!(handler.is_prescribed(0).unwrap());
        assert!(!handler.is_prescribed(1).unwrap());
        assert!(!handler.is_prescribed(2).unwrap());
        assert!(handler.is_prescribed(3).unwrap());
        assert!(!handler.is_prescribed(4).unwrap());
        assert!(!handler.is_prescribed(5).unwrap());

        // Check unknown mappings (global -> local unknown)
        assert_eq!(handler.iu(1).unwrap(), 0);
        assert_eq!(handler.iu(2).unwrap(), 1);
        assert_eq!(handler.iu(4).unwrap(), 2);
        assert_eq!(handler.iu(5).unwrap(), 3);

        // Check prescribed mappings (global -> local prescribed)
        assert_eq!(handler.ip(0).unwrap(), 0);
        assert_eq!(handler.ip(3).unwrap(), 1);

        // Check sorted lists
        assert_eq!(handler.unknown(), &vec![1, 2, 4, 5]);
        assert_eq!(handler.prescribed(), &vec![0, 3]);
    }

    #[test]
    fn complex_recompute_sequence() {
        let mut handler = EquationHandler::new(8);

        // Stage 1: Set some prescribed equations
        handler.recompute(&[(1, 10), (4, 40), (7, 70)]).unwrap();
        assert_eq!(handler.nu(), 5);
        assert_eq!(handler.np(), 3);
        assert_eq!(handler.unknown(), &vec![0, 2, 3, 5, 6]);
        assert_eq!(handler.prescribed(), &vec![1, 4, 7]);

        // Stage 2: Change prescribed equations completely
        handler.recompute(&[(0, 5), (2, 25), (3, 35), (6, 65)]).unwrap();
        assert_eq!(handler.nu(), 4);
        assert_eq!(handler.np(), 4);
        assert_eq!(handler.unknown(), &vec![1, 4, 5, 7]);
        assert_eq!(handler.prescribed(), &vec![0, 2, 3, 6]);

        // Stage 3: Remove all prescribed
        handler.recompute(&[]).unwrap();
        assert_eq!(handler.nu(), 8);
        assert_eq!(handler.np(), 0);
        assert_eq!(handler.unknown(), &vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(handler.prescribed().is_empty());

        // Verify all are unknown
        for e in 0..8 {
            assert!(!handler.is_prescribed(e).unwrap());
            assert_eq!(handler.iu(e).unwrap(), e);
            assert!(handler.ip(e).is_err());
        }
    }

    #[test]
    fn external_id_mapping_works() {
        let mut handler = EquationHandler::new(5);

        // Use different external IDs
        let p_list = &[(1, 100), (3, 300), (4, 400)];
        handler.recompute(p_list).unwrap();

        // Check that external IDs are stored correctly
        assert_eq!(handler.p_to_external_id[&1], 100);
        assert_eq!(handler.p_to_external_id[&3], 300);
        assert_eq!(handler.p_to_external_id[&4], 400);

        // Non-prescribed equations shouldn't be in the map
        assert!(!handler.p_to_external_id.contains_key(&0));
        assert!(!handler.p_to_external_id.contains_key(&2));

        // Update with different mapping
        handler.recompute(&[(0, 50), (2, 250)]).unwrap();

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
        handler.recompute(&[(2, 20), (5, 50), (8, 80)]).unwrap();

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
            assert!(!handler.is_prescribed(u).unwrap());
            assert!(handler.iu(u).is_ok());
            assert!(handler.ip(u).is_err());
        }
        for &p in handler.prescribed() {
            assert!(handler.is_prescribed(p).unwrap());
            assert!(handler.iu(p).is_err());
            assert!(handler.ip(p).is_ok());
        }
    }

    #[test]
    fn sorted_lists_are_actually_sorted() {
        let mut handler = EquationHandler::new(10);

        // Set prescribed equations in non-sorted order
        handler
            .recompute(&[(7, 70), (2, 20), (9, 90), (1, 10), (5, 50)])
            .unwrap();

        // Unknown list should be sorted
        let unknown = handler.unknown();
        assert_eq!(unknown, &vec![0, 3, 4, 6, 8]);
        assert!(unknown.windows(2).all(|w| w[0] < w[1]));

        // Prescribed list should be sorted
        let prescribed = handler.prescribed();
        assert_eq!(prescribed, &vec![1, 2, 5, 7, 9]);
        assert!(prescribed.windows(2).all(|w| w[0] < w[1]));
    }
}
