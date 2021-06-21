use super::*;

#[test]
fn ave_works() -> err::Result<()> {
    let x = [100, 100, 102, 98, 77, 99, 70, 105, 98];
    assert_eq!(stat::ave(&x)?, 849.0 / 9.0);
    Ok(())
}

#[test]
#[should_panic(expected = "cannot compute average of empty slice")]
fn ave_returns_error_on_empty_slice() {
    let x: [i32; 0] = [];
    panic!("{}", stat::ave(&x).unwrap_err().to_string());
}
