use russell_sparse::*;

fn main() {
    println!("Running Mem Check");
    println!("=================");
    let res = SolverMumps::new(MUMPS_SYMMETRY_NONE, true);
    println!("{:?}", res);
    println!("done");
}
