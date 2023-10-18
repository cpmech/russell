use std::mem;

/// Sorts 2 values in ascending order
pub fn sort2<T>(x: &mut (T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
}

/// Sorts 3 values in ascending order
pub fn sort3<T>(x: &mut (T, T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.2, &mut x.1);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.1, &mut x.0);
    }
}

/// Sorts 4 values in ascending order
pub fn sort4<T>(x: &mut (T, T, T, T))
where
    T: PartialOrd,
{
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.1, &mut x.2);
    }
    if x.3 < x.2 {
        mem::swap(&mut x.2, &mut x.3);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
    if x.2 < x.1 {
        mem::swap(&mut x.1, &mut x.2);
    }
    if x.1 < x.0 {
        mem::swap(&mut x.0, &mut x.1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{sort2, sort3, sort4};

    #[test]
    fn sort2_works() {
        let mut x = (1, 2);
        sort2(&mut x);
        assert_eq!(x, (1, 2));

        let mut x = (2, 1);
        sort2(&mut x);
        assert_eq!(x, (1, 2));

        let mut x = (1.0, 2.0);
        sort2(&mut x);
        assert_eq!(x, (1.0, 2.0));

        let mut x = (2.0, 1.0);
        sort2(&mut x);
        assert_eq!(x, (1.0, 2.0));
    }

    #[test]
    fn sort3_works() {
        let mut x = (1, 2, 3);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));

        let mut x = (1, 3, 2);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));

        let mut x = (3, 2, 1);
        sort3(&mut x);
        assert_eq!(x, (1, 2, 3));
    }

    #[test]
    fn sort4_works() {
        let mut x = (1, 2, 3, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 1, 3, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 3, 1, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (2, 3, 4, 1);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 3, 2, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 3, 4, 2);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (3, 1, 2, 4);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (4, 1, 2, 3);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (1, 4, 2, 3);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));

        let mut x = (4, 3, 2, 1);
        sort4(&mut x);
        assert_eq!(x, (1, 2, 3, 4));
    }
}
