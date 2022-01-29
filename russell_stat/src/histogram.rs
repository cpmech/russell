use num_traits::Num;

use crate::StrError;

pub struct Histogram<T>
where
    T: Num + Copy,
{
    stations: Vec<T>,
    counts: Vec<usize>,
}

impl<T> Histogram<T>
where
    T: Num + Copy + PartialOrd,
{
    /// Creates a new Histogram
    pub fn new(stations: &[T]) -> Result<Self, StrError> {
        if stations.len() < 2 {
            return Err("histogram must have at least 2 stations");
        }
        let nbins = stations.len() - 1;
        Ok(Histogram {
            stations: Vec::from(stations),
            counts: vec![0; nbins],
        })
    }

    /// Erase all counts
    pub fn reset(&mut self) {
        for i in 0..self.counts.len() {
            self.counts[i] = 0;
        }
    }

    /// Counts how many items fall within each bin
    pub fn count(&mut self, data: &[T]) {
        for x in data {
            if let Some(i) = self.find_bin(*x) {
                self.counts[i] += 1;
            }
        }
    }

    /// Finds which bin contains x
    fn find_bin(&self, x: T) -> Option<usize> {
        // handle values outside range
        let nstation = self.stations.len();
        if x < self.stations[0] {
            return None;
        }
        if x >= self.stations[nstation - 1] {
            return None;
        }

        // perform binary search
        let mut upper = nstation;
        let mut lower = 0;
        let mut mid;
        while upper - lower > 1 {
            mid = (upper + lower) / 2;
            if x >= self.stations[mid] {
                lower = mid
            } else {
                upper = mid
            }
        }
        Some(lower)
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Histogram;
    use crate::StrError;

    #[test]
    fn new_fails_on_wrong_input() -> Result<(), StrError> {
        assert_eq!(
            Histogram::<i32>::new(&[]).err(),
            Some("histogram must have at least 2 stations")
        );
        Ok(())
    }

    #[test]
    fn new_works() -> Result<(), StrError> {
        let stations: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let hist = Histogram::new(&stations)?;
        assert_eq!(hist.stations.len(), 6);
        assert_eq!(hist.counts.len(), 0);
        Ok(())
    }

    #[test]
    fn find_bin_works() -> Result<(), StrError> {
        let stations: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(&stations)?;

        let res = hist.find_bin(-3.3);
        assert_eq!(res, None);

        let res = hist.find_bin(7.0);
        assert_eq!(res, None);

        for i in 0..stations.len() {
            let res = hist.find_bin(stations[i]);
            if i < stations.len() - 1 {
                assert_eq!(res, Some(i));
            } else {
                assert_eq!(res, None);
            }
        }

        let res = hist.find_bin(0.5);
        assert_eq!(res, Some(0));

        let res = hist.find_bin(1.5);
        assert_eq!(res, Some(1));

        let res = hist.find_bin(2.5);
        assert_eq!(res, Some(2));

        let res = hist.find_bin(3.99999999999999);
        assert_eq!(res, Some(3));

        let res = hist.find_bin(4.999999);
        assert_eq!(res, Some(4));

        Ok(())
    }

    #[test]
    fn count_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let data = [
            0.0, 0.1, 0.2, 0.3, 0.9, // 5
            1.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.99, // 8
            2.0, 2.5, // 2
            3.0, 3.5, // 2
            4.1, 4.5, 4.9, // 3
            -3.0, -2.0, -1.0, // outside
            5.0, 6.0, 7.0, 8.0, // outside
        ];
        let stations: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        assert_eq!(hist.counts, &[5, 8, 2, 2, 3]);

        #[rustfmt::skip]
        let data: [i32; 12]= [
            0, 0, 0, 0, // 4
            1, // 1
            2, 2, 2, // 3
            // 0
            // 0
            5, 5, // 2
            -1, 10, // outside
        ];
        let stations: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        assert_eq!(hist.counts, &[4, 1, 3, 0, 0]);
        Ok(())
    }
}
