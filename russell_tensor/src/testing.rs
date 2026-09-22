use crate::Tensor2;
use crate::polar_decomp::{PolarAlgo, polar_decomp_mx};
use crate::{SQRT_2, SQRT_3, SQRT_6};
use russell_lab::{mat_approx_eq, small_mat_approx_eq, small_mat_mat_mul, small_mat_t_mat_mul, sort3};

// -----------------------------------------------------------------------------------
// Auxiliary functions
// -----------------------------------------------------------------------------------

/// Performs similarity transformation (make sure to return a symmetric matrix)
///
/// ```text
/// A = Q . L . Q^T
/// ```
pub fn similarity_transform(aa: &mut [[f64; 3]; 3], ll: &[[f64; 3]; 3], qq: &[[f64; 3]; 3]) {
    for i in 0..3 {
        for j in 0..3 {
            aa[i][j] = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    aa[i][j] += qq[i][k] * ll[k][l] * qq[j][l];
                }
            }
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let m = 0.5 * (aa[i][j] + aa[j][i]);
            aa[i][j] = m;
            aa[j][i] = m;
        }
    }
}

// -----------------------------------------------------------------------------------
// Eigen-problems testing
// -----------------------------------------------------------------------------------

/// Generates a set of tensors including all distinct, two repeated, and three repeated eigenvalues
///
/// Returns `(tensors, eigenvalues)`
/// Eigenvalues are sorted in decreasing order.
pub fn generate_tensors2() -> (Vec<Tensor2<6>>, Vec<[f64; 3]>) {
    // orthogonal matrices
    #[rustfmt::skip]
    const ROTATIONS: [[[f64; 3]; 3]; 3] = [
        [ // identity
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [ // skew
            [0.0, 1.0, 0.0], // skew
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        [ // Q rotates axes to octahedral system
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ],
    ];
    // diagonals: general, two-nearly-equal, and triple-equal
    let mut diagonals: Vec<_> = vec![
        // identity
        [1.0, 1.0, 1.0],
        // distinct
        [3.0, 1.0, 2.0],
        // d01
        [1.0, 2.0, 2.0],
        [2.0, 1.0, 2.0],
        [2.0, 2.0, 1.0],
        // d12
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        // d01
        [-2.0, -1.0, -1.0],
        [-1.0, -2.0, -1.0],
        [-1.0, -1.0, -2.0],
        // d12
        [-1.0, -2.0, -2.0],
        [-2.0, -1.0, -2.0],
        [-2.0, -2.0, -1.0],
        // d01
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        // d12
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        // d01
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        // d12
        [0.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [-1.0, -1.0, 0.0],
    ];
    for eps in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15] {
        diagonals.push([1.0, -0.5 + eps / 2.0, -0.5 - eps / 2.0]);
    }
    // generate matrices
    let mut aa_3x3 = [[0.0; 3]; 3];
    let mut ll_3x3 = [[0.0; 3]; 3];
    let mut tensors = Vec::new();
    let mut eigenvals = Vec::new();
    for ll in &diagonals {
        for qq in &ROTATIONS {
            ll_3x3[0][0] = ll[0];
            ll_3x3[1][1] = ll[1];
            ll_3x3[2][2] = ll[2];
            similarity_transform(&mut aa_3x3, &ll_3x3, qq);
            let aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
            tensors.push(aa);
            let mut l0 = ll[0];
            let mut l1 = ll[1];
            let mut l2 = ll[2];
            sort3(&mut l2, &mut l1, &mut l0); // will sort: l2 < l1 < l0
            eigenvals.push([l0, l1, l2]);
        }
    }
    (tensors, eigenvals)
}

/// Generates eigen-problem
///
/// Returns `(aa, expected_lambda, expected_proj)` sorted in decreasing order by lambda
pub fn generate_eigen_problem(l0: f64, l1: f64, l2: f64) -> (Tensor2<6>, [f64; 3], [Tensor2<6>; 3]) {
    // Q rotates axes to octahedral system
    #[rustfmt::skip]
        let qq = [
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ];
    // A = Q . L . Q^T
    let mut aa = [[0.0; 3]; 3];
    similarity_transform(&mut aa, &[[l0, 0.0, 0.0], [0.0, l1, 0.0], [0.0, 0.0, l2]], &qq);
    // expected eigenvectors
    #[rustfmt::skip]
        let n0 = [
            2.0 / SQRT_6,
            1.0 / SQRT_3,
            0.0,
        ];
    #[rustfmt::skip]
        let n1 = [
            -1.0 / SQRT_6,
             1.0 / SQRT_3,
            -1.0 / SQRT_2,
        ];
    #[rustfmt::skip]
        let n2 = [
            -1.0 / SQRT_6,
             1.0 / SQRT_3,
             1.0 / SQRT_2,
        ];
    // expected eigen-dyads
    let dyad0 = [
        [n0[0] * n0[0], n0[0] * n0[1], n0[0] * n0[2]],
        [n0[1] * n0[0], n0[1] * n0[1], n0[1] * n0[2]],
        [n0[2] * n0[0], n0[2] * n0[1], n0[2] * n0[2]],
    ];
    let dyad1 = [
        [n1[0] * n1[0], n1[0] * n1[1], n1[0] * n1[2]],
        [n1[1] * n1[0], n1[1] * n1[1], n1[1] * n1[2]],
        [n1[2] * n1[0], n1[2] * n1[1], n1[2] * n1[2]],
    ];
    let dyad2 = [
        [n2[0] * n2[0], n2[0] * n2[1], n2[0] * n2[2]],
        [n2[1] * n2[0], n2[1] * n2[1], n2[1] * n2[2]],
        [n2[2] * n2[0], n2[2] * n2[1], n2[2] * n2[2]],
    ];
    // sort eigen variables
    let lam = [l0, l1, l2];
    let dyads = [dyad0, dyad1, dyad2];
    let mut indices = [0, 1, 2];
    indices.sort_by(|&i, &j| lam[j].partial_cmp(&lam[i]).unwrap());
    let sorted_lam = [lam[indices[0]], lam[indices[1]], lam[indices[2]]];
    let sorted_dyad = [dyads[indices[0]], dyads[indices[1]], dyads[indices[2]]];
    // handle coalescent case
    let d01 = f64::abs(sorted_lam[0] - sorted_lam[1]);
    let d12 = f64::abs(sorted_lam[1] - sorted_lam[2]);
    let mut aux = [[0.0; 3]; 3];
    let proj = if d01 < 1e-14 {
        // λ0 = λ1 > λ2
        for i in 0..3 {
            for j in 0..3 {
                aux[i][j] = sorted_dyad[0][i][j] + sorted_dyad[1][i][j];
            }
        }
        [
            Tensor2::<6>::new(),
            Tensor2::<6>::from_std_matrix(&aux).unwrap(),
            Tensor2::<6>::from_std_matrix(&sorted_dyad[2]).unwrap(),
        ]
    } else if d12 < 1e-14 {
        // λ0 > λ1 = λ2
        for i in 0..3 {
            for j in 0..3 {
                aux[i][j] = sorted_dyad[1][i][j] + sorted_dyad[2][i][j];
            }
        }
        [
            Tensor2::<6>::from_std_matrix(&sorted_dyad[0]).unwrap(),
            Tensor2::<6>::from_std_matrix(&aux).unwrap(),
            Tensor2::<6>::new(),
        ]
    } else {
        // λ0 ≠ λ1 ≠ λ2
        [
            Tensor2::<6>::from_std_matrix(&sorted_dyad[0]).unwrap(),
            Tensor2::<6>::from_std_matrix(&sorted_dyad[1]).unwrap(),
            Tensor2::<6>::from_std_matrix(&sorted_dyad[2]).unwrap(),
        ]
    };
    // results
    let aa_ten = Tensor2::<6>::from_std_matrix(&aa).unwrap();
    (aa_ten, sorted_lam, proj)
}

// -----------------------------------------------------------------------------------
// Reference eigenprojector results
// -----------------------------------------------------------------------------------

/// Holds a reference matrix, eigenvalues, and eigendyads constructed from eigenvectors
pub struct ReferenceEigenDyads {
    /// input matrix
    pub aa_3x3: [[f64; 3]; 3],

    /// sorted eigenvalues
    pub ll: [f64; 3],

    /// first eigenvector
    pub n0: [f64; 3],

    /// second eigenvector
    pub n1: [f64; 3],

    /// third eigenvector
    pub n2: [f64; 3],

    /// n0 ⊗ n0 associated with λ0
    pub dyad0: [[f64; 3]; 3],

    /// n0 ⊗ n0 associated with λ1
    pub dyad1: [[f64; 3]; 3],

    /// n2 ⊗ n2 associated with λ2
    pub dyad2: [[f64; 3]; 3],
}

/// Returns reference eigen-dyads for testing
pub fn reference_eigendyads() -> (Vec<&'static str>, Vec<ReferenceEigenDyads>) {
    let mut names = Vec::new();
    let mut data = Vec::new();

    //
    // repeat 01
    //

    // input matrix
    //     ┌       ┐
    //     │ 2 1 0 │
    // A = │ 1 2 0 │
    //     │ 0 0 3 │
    //     └       ┘
    let aa_3x3 = [
        [2.0, 1.0, 0.0], // 0
        [1.0, 2.0, 0.0], // 1
        [0.0, 0.0, 3.0], // 2
    ];

    // sorted eigenvalues = {3, 3, 1} => λ0 = λ1 repeated
    let ll = [3.0, 3.0, 1.0];

    // analytical orthonormal eigenvectors
    let n0 = [0.0, 0.0, 1.0];
    let n1 = [1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];
    let n2 = [-1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];

    // n0 ⊗ n0 associated with λ0 = 3
    let dyad0 = [
        [0.0, 0.0, 0.0], // 0
        [0.0, 0.0, 0.0], // 1
        [0.0, 0.0, 1.0], // 2
    ];
    // n0 ⊗ n0 associated with λ1 = 3
    let dyad1 = [
        [0.5, 0.5, 0.0], // 0
        [0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0], // 2
    ];
    // n2 ⊗ n2 associated with λ1 = 1
    let dyad2 = [
        [0.5, -0.5, 0.0], // 0
        [-0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0],  // 2
    ];

    names.push("repeat-01");
    data.push(ReferenceEigenDyads {
        aa_3x3,
        ll,
        n0,
        n1,
        n2,
        dyad0,
        dyad1,
        dyad2,
    });

    //
    // repeat 12
    //

    // input matrix
    //     ┌       ┐
    //     │ 3 1 0 │
    // A = │ 1 3 0 │
    //     │ 0 0 2 │
    //     └       ┘
    let aa_3x3 = [
        [3.0, 1.0, 0.0], // 0
        [1.0, 3.0, 0.0], // 1
        [0.0, 0.0, 2.0], // 2
    ];

    // sorted eigenvalues = {4, 2, 2} => λ1 = λ2 repeated
    let ll = [4.0, 2.0, 2.0];

    // analytical orthonormal eigenvectors
    let n0 = [1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];
    let n1 = [-1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];
    let n2 = [0.0, 0.0, 1.0];

    // n0 ⊗ n0 associated with λ0 = 4
    let dyad0 = [
        [0.5, 0.5, 0.0], // 0
        [0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0], // 2
    ];
    // n1 ⊗ n1 associated with λ1 = 2
    let dyad1 = [
        [0.5, -0.5, 0.0], // 0
        [-0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0],  // 2
    ];
    // n2 ⊗ n2 associated with λ2 = 2
    let dyad2 = [
        [0.0, 0.0, 0.0], // 0
        [0.0, 0.0, 0.0], // 1
        [0.0, 0.0, 1.0], // 2
    ];

    names.push("repeat-12");
    data.push(ReferenceEigenDyads {
        aa_3x3,
        ll,
        n0,
        n1,
        n2,
        dyad0,
        dyad1,
        dyad2,
    });

    //
    // all distinct planar
    //

    // input matrix
    //     ┌          ┐
    //     │  3 -1  0 │
    // A = │ -1  3  0 │
    //     │  0  0  5 │
    //     └          ┘
    let aa_3x3 = [
        [3.0, -1.0, 0.0], // 0
        [-1.0, 3.0, 0.0], // 1
        [0.0, 0.0, 5.0],  // 2
    ];

    // sorted eigenvalues = {5, 4, 2} => all distinct
    let ll = [5.0, 4.0, 2.0];

    // analytical orthonormal eigenvectors
    let n0 = [0.0, 0.0, 1.0];
    let n1 = [-1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];
    let n2 = [1.0 / SQRT_2, 1.0 / SQRT_2, 0.0];

    // n0 ⊗ n0 associated with λ0 = 5
    let dyad0 = [
        [0.0, 0.0, 0.0], // 0
        [0.0, 0.0, 0.0], // 1
        [0.0, 0.0, 1.0], // 2
    ];
    // n1 ⊗ n1 associated with λ1 = 4
    let dyad1 = [
        [0.5, -0.5, 0.0], // 0
        [-0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0],  // 2
    ];
    // n2 ⊗ n2 associated with λ2 = 2
    let dyad2 = [
        [0.5, 0.5, 0.0], // 0
        [0.5, 0.5, 0.0], // 1
        [0.0, 0.0, 0.0], // 2
    ];

    names.push("all-distinct-planar");
    data.push(ReferenceEigenDyads {
        aa_3x3,
        ll,
        n0,
        n1,
        n2,
        dyad0,
        dyad1,
        dyad2,
    });

    //
    // all distinct
    //

    // input matrix
    //     ┌             ┐
    //     │  25 -10   2 │
    // A = │ -10  22  -8 │
    //     │   2  -8  16 │
    //     └             ┘
    let aa_3x3 = [
        [25.0, -10.0, 2.0],  // 1
        [-10.0, 22.0, -8.0], // 2
        [2.0, -8.0, 16.0],   // 3
    ];

    // sorted eigenvalues = {36, 18, 9} => all distinct
    let ll = [36.0, 18.0, 9.0];

    // analytical orthonormal eigenvectors
    let n0 = [2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0];
    let n1 = [-2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];
    let n2 = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];

    // n0 ⊗ n0 associated with λ0 = 5
    let dyad0 = [
        [4.0 / 9.0, -4.0 / 9.0, 2.0 / 9.0],
        [-4.0 / 9.0, 4.0 / 9.0, -2.0 / 9.0],
        [2.0 / 9.0, -2.0 / 9.0, 1.0 / 9.0],
    ];
    // n1 ⊗ n1 associated with λ1 = 4
    let dyad1 = [
        [4.0 / 9.0, 2.0 / 9.0, -4.0 / 9.0],
        [2.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0],
        [-4.0 / 9.0, -2.0 / 9.0, 4.0 / 9.0],
    ];
    // n2 ⊗ n2 associated with λ2 = 2
    let dyad2 = [
        [1.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0],
        [2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
        [2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
    ];

    names.push("all-distinct");
    data.push(ReferenceEigenDyads {
        aa_3x3,
        ll,
        n0,
        n1,
        n2,
        dyad0,
        dyad1,
        dyad2,
    });

    (names, data)
}

// -----------------------------------------------------------------------------------
// Habera-Zilian test cases
// 1. Habera M. and Zilian A. (2025) Numerically stable evaluation of closed-form
//    expressions for eigenvalues of 3×3 matrices. <https://arxiv.org/abs/2511.00292>
// -----------------------------------------------------------------------------------

pub struct HaberaZilian {
    pub names: [&'static str; 11],
    pub deltas: [f64; 10],
}

impl HaberaZilian {
    pub fn new() -> Self {
        HaberaZilian {
            names: [
                "single",
                "single_lim_J3",
                "single_lim_disc_t",
                "single_lim_disc_n",
                "single_lim_J3J2",
                "single_J3",
                "single_J3_lim_J2",
                "double",
                "double_lim_J3J2",
                "triple_J3",
                "d3",
            ],
            deltas: [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 500.0],
        }
    }

    /// Returns the prescribed diagonal of the Habera-Zilian test tensor
    pub fn diagonal(&self, name: &str, delta: f64) -> [f64; 3] {
        const A: f64 = 1.0;
        match name {
            "single" => [-A / 4.0, (1.0 * A) / 4.0, (2.0 + 2.0 * delta) * A / 4.0],
            "single_lim_J3" => [(-1.0 - delta) * A / 4.0, 0.0, (1.0 + 2.0 * delta) * A / 4.0],
            "single_lim_disc_t" => [-A, 1.0 * A, (1.0 + delta) * A],
            "single_lim_disc_n" => [0.0, (2.0 - delta) * A / 2.0, (2.0 + delta) * A / 2.0],
            "single_lim_J3J2" => [(1.0 - delta) * A, 1.0 * A, (1.0 + 2.0 * delta) * A],
            "single_J3" => [(-1.0 - delta) * A / 2.0, 0.0, (1.0 + delta) * A / 2.0],
            "single_J3_lim_J2" => [(1.0 - delta) * A, 1.0 * A, (1.0 + delta) * A],
            "double" => [(-1.0 - delta) * A, 1.0 * A, 1.0 * A],
            "double_lim_J3J2" => [1.0 * A, 1.0 * A, (1.0 + delta) * A],
            "triple_J3" => [-delta, 0.0, delta],
            "d3" => [0.0, 1.0 * A, (2.0 + delta) * A],
            _ => panic!("unknown HZ test case: {}", name),
        }
    }

    /// Generates the Habera-Zilian test tensor
    pub fn tensor(&self, name: &str, delta: f64) -> Tensor2<6> {
        let d = self.diagonal(name, delta);
        let qq_3x3 = [
            [1.0 / SQRT_2, -0.5, 0.5],
            [1.0 / SQRT_2, 0.5, -0.5],
            [0.0, 1.0 / SQRT_2, 1.0 / SQRT_2],
        ];
        let mut aa_3x3 = [[0.0; 3]; 3];
        let ll = [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]];
        similarity_transform(&mut aa_3x3, &ll, &qq_3x3);
        Tensor2::from_std_matrix(&aa_3x3).unwrap()
    }

    /// Returns ABSOLUTE tolerances to check eigenvalues
    ///
    /// Returns `(tol_idempotent, tol_reconstruction)`
    pub fn tolerances_eigenvalues(&self, name: &str, delta: f64) -> f64 {
        let default = 1e-15;
        match name {
            "single" => {
                if delta == 5e2 {
                    1e-13
                } else {
                    default
                }
            }
            "single_lim_J3" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-13
                } else {
                    default
                }
            }
            "single_lim_disc_t" => {
                if delta == 1e-12 || delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-13
                } else {
                    default
                }
            }
            "single_lim_disc_n" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "single_lim_J3J2" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "single_J3" => {
                if delta == 5e2 {
                    1e-13
                } else {
                    default
                }
            }
            "single_J3_lim_J2" => {
                if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "double" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "double_lim_J3J2" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "triple_J3" => {
                if delta == 5e2 {
                    1e-12
                } else {
                    default
                }
            }
            "d3" => {
                if delta == 5e0 {
                    1e-14
                } else if delta == 5e2 {
                    1e-13
                } else {
                    default
                }
            }
            _ => panic!("unknown HZ test case: {}", name),
        }
    }

    /// Returns ABSOLUTE tolerances to check eigenprojectors
    ///
    /// Returns `(tol_idempotent, tol_reconstruction)`
    pub fn tolerances_projectors(&self, name: &str, delta: f64) -> (f64, f64) {
        const TOL_IDEM: f64 = 1e-13;
        const TOL_RECON: f64 = 1e-12;
        let default = (TOL_IDEM, TOL_RECON);
        match name {
            "single" => default,
            "single_lim_J3" => default,
            "single_lim_disc_t" => {
                if delta == 1e-10 {
                    (TOL_IDEM, 1e-10)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-8)
                } else if delta == 1e-6 {
                    (1e-9, 1e-9)
                } else if delta == 1e-4 {
                    (1e-11, 1e-11)
                } else {
                    default
                }
            }
            "single_lim_disc_n" => {
                if delta == 1e-10 {
                    (TOL_IDEM, 1e-10)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-8)
                } else if delta == 1e-6 {
                    (1e-9, 1e-9)
                } else if delta == 1e-4 {
                    (1e-11, 1e-11)
                } else {
                    default
                }
            }
            "single_lim_J3J2" => {
                if delta == 1e-12 {
                    (TOL_IDEM, 1e-11)
                } else if delta == 1e-10 {
                    (TOL_IDEM, 1e-9)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-7)
                } else if delta == 1e-6 {
                    (1e-9, TOL_RECON)
                } else if delta == 1e-4 {
                    (1e-11, TOL_RECON)
                } else {
                    default
                }
            }
            "single_J3" => default,
            "single_J3_lim_J2" => {
                if delta == 1e-12 {
                    (TOL_IDEM, 1e-11)
                } else if delta == 1e-10 {
                    (TOL_IDEM, 1e-9)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-7)
                } else if delta == 1e-6 {
                    (1e-9, TOL_RECON)
                } else if delta == 1e-4 {
                    (1e-12, TOL_RECON)
                } else {
                    default
                }
            }
            "double" => default,
            "double_lim_J3J2" => {
                if delta == 1e-10 {
                    (TOL_IDEM, 1e-10)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-8)
                } else if delta == 1e-6 {
                    (1e-9, TOL_RECON)
                } else if delta == 1e-4 {
                    (1e-11, TOL_RECON)
                } else {
                    default
                }
            }
            "triple_J3" => {
                if delta == 1e-12 {
                    (TOL_IDEM, 1e-11)
                } else if delta == 1e-10 {
                    (TOL_IDEM, 1e-9)
                } else if delta == 1e-8 {
                    (TOL_IDEM, 1e-7)
                } else {
                    default
                }
            }
            "d3" => default,
            _ => panic!("unknown HZ test case: {}", name),
        }
    }
}

// -----------------------------------------------------------------------------------
// Polar decomposition testing
// -----------------------------------------------------------------------------------

pub struct ReferencePolarDecomp {}

impl ReferencePolarDecomp {
    /// Example 01 (Brannon, Eq. 12.39): in-plane deformation gradient;
    /// the polar rotation is a 60° rotation about the E3 axis.
    #[rustfmt::skip]
    pub fn example01() -> Tensor2<9> {
        Tensor2::<9>::from_std_matrix(&[
            [ 0.61784609690826542, -0.70889727457341833, 0.0],
            [ 0.59014083110323967,  0.13215390309173483, 0.0],
            [ 0.0,                  0.0,                 3.0],
        ]).unwrap()
    }

    /// Example 03 (McGinty, continuum mechanics dot org): fully 3-D deformation gradient.
    #[rustfmt::skip]
    pub fn example03() -> Tensor2<9> {
        Tensor2::<9>::from_std_matrix(&[
            [ 1.000,  0.495,  0.500],
            [-0.333,  1.000, -0.247],
            [ 0.959,  0.000,  1.500],
        ]).unwrap()
    }

    /// Higham & Noferini test (5.1).
    #[rustfmt::skip]
    pub fn case51() -> Tensor2<9> {
        Tensor2::<9>::from_std_matrix(&[
            [0.1, 0.2, 0.3],
            [0.1, 0.1, 0.0],
            [0.3, 0.2, 0.1],
        ]).unwrap()
    }

    /// Higham & Noferini test (5.2), for a given scale factor `y`.
    #[rustfmt::skip]
    pub fn case52(y: f64) -> Tensor2<9> {
        Tensor2::<9>::from_std_matrix(&[
            [(720.0 * y - 25.0) / 1275.0, (-650.0 * y + 300.0) / 1275.0, (710.0 * y + 300.0) / 1275.0],
            [(396.0 * y + 70.0) / 1275.0, (-145.0 * y - 840.0) / 1275.0, (178.0 * y - 840.0) / 1275.0],
            [(972.0 * y - 10.0) / 1275.0, (610.0 * y + 120.0) / 1275.0, (-529.0 * y + 120.0) / 1275.0],
        ]).unwrap()
    }

    /// Reference rotation for example 01 (60° about E3).
    pub fn example01_rotation() -> [[f64; 3]; 3] {
        [
            [0.5, -0.8660254037844386, 0.0],
            [0.8660254037844386, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Reference right stretch for example 01.
    pub fn example01_stretch() -> [[f64; 3]; 3] {
        [[0.82, -0.24, 0.0], [-0.24, 0.68, 0.0], [0.0, 0.0, 3.0]]
    }

    /// Reference rotation for example 03 (3-decimal published values).
    pub fn example03_rotation() -> [[f64; 3]; 3] {
        [[0.914, 0.377, -0.148], [-0.374, 0.926, 0.049], [0.156, 0.011, 0.988]]
    }

    /// Reference right stretch for example 03 (3-decimal published values).
    pub fn example03_stretch() -> [[f64; 3]; 3] {
        [[1.188, 0.079, 0.783], [0.079, 1.113, -0.024], [0.783, -0.024, 1.396]]
    }

    /// Exact polar factor for test 5.2 (well-conditioned case).
    pub fn case52_rotation() -> [[f64; 3]; 3] {
        [
            [139.0 / 255.0, -14.0 / 51.0, 202.0 / 255.0],
            [466.0 / 1275.0, -197.0 / 255.0, -662.0 / 1275.0],
            [962.0 / 1275.0, 146.0 / 255.0, -409.0 / 1275.0],
        ]
    }
}

// -----------------------------------------------------------------------------------
// Check helpers
// -----------------------------------------------------------------------------------

/// Checks that `A = Q · H` with `Q` orthogonal, within the given tolerance.
pub fn check_polar(a: &Tensor2<9>, q: &Tensor2<9>, h: &Tensor2<6>, tol: f64) {
    let mut am = [[0.0; 3]; 3];
    let mut qm = [[0.0; 3]; 3];
    let mut hm = [[0.0; 3]; 3];
    a.to_std_matrix_slice(&mut am);
    q.to_std_matrix_slice(&mut qm);
    h.to_std_matrix_slice(&mut hm);
    let mut qh = [[0.0; 3]; 3];
    small_mat_mat_mul(&mut qh, 1.0, &qm, &hm, 0.0, 3);
    small_mat_approx_eq(&qh, &am, tol);
    let mut qtq = [[0.0; 3]; 3];
    small_mat_t_mat_mul(&mut qtq, 1.0, &qm, &qm, 0.0, 3);
    #[rustfmt::skip]
    let ii = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    small_mat_approx_eq(&qtq, &ii, tol);
}

/// Runs both algorithms on `a` and checks that each satisfies `A = Q · H`
/// (with `Q` orthogonal) and that the two agree (the polar decomposition is
/// unique for any non-singular `A`).
pub fn check_agree(a: &Tensor2<9>) {
    // Brannon (iterative)
    let mut rb = Tensor2::<9>::new();
    let mut ub = Tensor2::<6>::new();
    let mut vb = Tensor2::<6>::new();
    polar_decomp_mx(&mut rb, &mut ub, Some(&mut vb), PolarAlgo::Iterative, a).unwrap();
    check_polar(a, &rb, &ub, 1e-13);

    // Higham & Noferini (quaternion)
    let mut qh = Tensor2::<9>::new();
    let mut hh = Tensor2::<6>::new();
    polar_decomp_mx(&mut qh, &mut hh, None, PolarAlgo::Quaternion, a).unwrap();
    check_polar(a, &qh, &hh, 1e-13);

    // The two implementations must agree
    mat_approx_eq(&rb.as_std_matrix(), &qh.as_std_matrix(), 1e-13);
    mat_approx_eq(&ub.as_std_matrix(), &hh.as_std_matrix(), 1e-13);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{generate_eigen_problem, similarity_transform};
    use crate::Tensor2;
    use crate::testing::reference_eigendyads;
    use crate::{SQRT_2, SQRT_3, SQRT_6};
    use russell_lab::{Matrix, approx_eq, array_approx_eq, mat_approx_eq, small_mat_approx_eq};

    #[test]
    fn check_similarity_transform() {
        // Q rotates axes to octahedral system
        #[rustfmt::skip]
        let qq_3x3 = [
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ];
        let l1 = 1.0;
        let l2 = 2.0;
        let l3 = 3.0;
        let ll = [[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]];
        let mut aa_3x3 = [[0.0; 3]; 3];
        // transform and check invariants
        similarity_transform(&mut aa_3x3, &ll, &qq_3x3);
        let aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
        approx_eq(aa.invariant_ii1(), l1 + l2 + l3, 1e-15);
        approx_eq(aa.invariant_ii2(), l1 * l2 + l2 * l3 + l3 * l1, 1e-14);
        approx_eq(aa.invariant_ii3(), l1 * l2 * l3, 1e-14);
        approx_eq(aa.norm(), f64::sqrt(l1 * l1 + l2 * l2 + l3 * l3), 1e-15);
        #[rustfmt::skip]
        let qqt_3x3 = [
            [ 2.0 / SQRT_6, 1.0 / SQRT_3,  0.0         ],
            [-1.0 / SQRT_6, 1.0 / SQRT_3, -1.0 / SQRT_2],
            [-1.0 / SQRT_6, 1.0 / SQRT_3,  1.0 / SQRT_2],
        ];
        // transform back and compare matrices
        let mut ll_3x3 = [[0.0; 3]; 3];
        similarity_transform(&mut ll_3x3, &aa_3x3, &qqt_3x3);
        mat_approx_eq(&Matrix::from(&ll_3x3), &ll, 1e-14);
    }

    #[test]
    fn generate_eigen_problem_works() {
        const VERBOSE: bool = true;

        let (names, data) = reference_eigendyads();
        let mut aa_rec_3x3 = [[0.0; 3]; 3];
        let mut dyad0 = [[0.0; 3]; 3];
        let mut dyad1 = [[0.0; 3]; 3];
        let mut dyad2 = [[0.0; 3]; 3];
        for i in 0..names.len() {
            let dat = &data[i];
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{}", names[i]);
            }

            // check eigendyads
            for i in 0..3 {
                for j in 0..3 {
                    dyad0[i][j] = dat.n0[i] * dat.n0[j];
                    dyad1[i][j] = dat.n1[i] * dat.n1[j];
                    dyad2[i][j] = dat.n2[i] * dat.n2[j];
                }
            }
            small_mat_approx_eq(&dyad0, &dat.dyad0, 1e-15);
            small_mat_approx_eq(&dyad1, &dat.dyad1, 1e-15);
            small_mat_approx_eq(&dyad2, &dat.dyad2, 1e-15);

            // reconstruct A using analytical eigendyads
            for i in 0..3 {
                for j in 0..3 {
                    aa_rec_3x3[i][j] =
                        dat.ll[0] * dat.dyad0[i][j] + dat.ll[1] * dat.dyad1[i][j] + dat.ll[2] * dat.dyad2[i][j];
                }
            }
            let aa_rec_3x3_mat = Matrix::from(&aa_rec_3x3);
            small_mat_approx_eq(&dat.aa_3x3, &aa_rec_3x3_mat, 1e-15);

            // generate eigen problem
            let (aa, e_ll, e_proj) = generate_eigen_problem(dat.ll[2], dat.ll[1], dat.ll[0]);
            let aa_std = aa.as_std_matrix();

            // check the trace of A
            let tr_a = aa.vec[0] + aa.vec[1] + aa.vec[2];

            // check the expected eigenvalues
            if VERBOSE {
                println!("A =\n{}", aa_std);
                println!("e_ll = {:?}", e_ll);
            }
            array_approx_eq(&e_ll, &dat.ll, 1e-15);
            let tol = if names[i] == "all-distinct" { 1e-14 } else { 1e-15 };
            approx_eq(tr_a, dat.aa_3x3[0][0] + dat.aa_3x3[1][1] + dat.aa_3x3[2][2], tol);

            // check the reconstruction
            let mut aa_rec = Tensor2::<6>::new();
            for m in 0..6 {
                aa_rec.vec[m] = e_ll[0] * e_proj[0].vec[m] + e_ll[1] * e_proj[1].vec[m] + e_ll[2] * e_proj[2].vec[m];
            }
            let aa_rec_std = aa_rec.as_std_matrix();
            if VERBOSE {
                println!("p0 =\n{}", e_proj[0].as_std_matrix());
                println!("p1 =\n{}", e_proj[1].as_std_matrix());
                println!("p2 =\n{}", e_proj[2].as_std_matrix());
            }
            mat_approx_eq(&aa_std, &aa_rec_std, 1e-15);
        }
    }
}
