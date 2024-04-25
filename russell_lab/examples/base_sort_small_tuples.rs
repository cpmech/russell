use russell_lab::*;

fn main() -> Result<(), StrError> {
    // sorting slices with the standard function
    let mut u2 = vec![2.0, 1.0];
    let mut u3 = vec![3.0, 1.0, 2.0];
    let mut u4 = vec![3.0, 1.0, 4.0, 2.0];
    u2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    u3.sort_by(|a, b| a.partial_cmp(b).unwrap());
    u4.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("u2 = {:?}", u2);
    println!("u3 = {:?}", u3);
    println!("u4 = {:?}", u4);
    assert_eq!(&u2, &[1.0, 2.0]);
    assert_eq!(&u3, &[1.0, 2.0, 3.0]);
    assert_eq!(&u4, &[1.0, 2.0, 3.0, 4.0]);

    // sorting small tuples
    let mut v2 = (2.0, 1.0);
    let mut v3 = (3.0, 1.0, 2.0);
    let mut v4 = (3.0, 1.0, 4.0, 2.0);
    sort2(&mut v2);
    sort3(&mut v3);
    sort4(&mut v4);
    println!("v2 = {:?}", v2);
    println!("v3 = {:?}", v3);
    println!("v4 = {:?}", v4);
    assert_eq!(v2, (1.0, 2.0));
    assert_eq!(v3, (1.0, 2.0, 3.0));
    assert_eq!(v4, (1.0, 2.0, 3.0, 4.0));
    Ok(())
}
