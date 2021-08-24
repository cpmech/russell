use russell_chk;
use russell_lab;
use russell_openblas;
use russell_stat;
use russell_tensor;

pub fn print_descriptions() {
    println!("     chk: {}", russell_chk::desc());
    println!("     lab: {}", russell_lab::desc());
    println!("openblas: {}", russell_openblas::desc());
    println!("    stat: {}", russell_stat::desc());
    println!("  tensor: {}", russell_tensor::desc());
}

#[test]
fn print_descriptions_work() {
    print_descriptions();
}
