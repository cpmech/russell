use super::*;

#[test]
fn it_works() {
    let x = [100, 100, 102, 98, 77, 99, 70, 105, 98];
    assert_eq!(stat::ave(&x), 849.0 / 9.0);
}
