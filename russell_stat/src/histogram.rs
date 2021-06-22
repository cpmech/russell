#[derive(Debug)]
pub struct Histogram<T>
where
    T: Into<f64> + Copy,
{
    stations: Vec<T>,
    counts: Vec<i32>,
}

impl<T> Histogram<T>
where
    T: Into<f64> + Copy,
{
    pub fn new(stations: &[T]) -> Self {
        Histogram {
            stations: Vec::from(stations),
            counts: Vec::new(),
        }
    }

    /*
    pub fn new(first_station: T, last_station_plus_one: T) -> Self {
        Histogram{
            stations: Vec
        }
    }
    */
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_new_works() {
        let stations = [0, 1, 2, 3, 4, 5];
        let hist = Histogram::new(&stations);
        assert_eq!(hist.stations, &[0, 1, 2, 3, 4, 5])
    }
}
