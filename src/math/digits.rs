use num_traits::{Euclid, Zero};
use std::ops::{self, Deref};

#[derive(Debug, Clone)]
pub struct Digits<const B: u32, const HINT: usize = 8> {
    pub digs: Vec<u32>,
}

impl<const B: u32, const HINT: usize, T> From<&T> for Digits<B, HINT>
where
    T: Zero + From<u32> + Euclid + TryInto<u32> + Clone,
{
    // note: internall digits are little-endian -- the least significant digit is
    // is at index 0, etc.
    fn from(n: &T) -> Self {
        debug_assert!(B >= 2);
        let base = T::from(B);
        let mut n = n.clone();
        let mut r: T;
        // we rarely care about messing with digits unless we have at least
        // 8 digit numberes or so
        let mut digits = Vec::with_capacity(HINT);
        while !n.is_zero() {
            (n, r) = n.div_rem_euclid(&base);
            digits.push(r.try_into().ok().unwrap());
        }
        Self { digs: digits }
    }
}

impl<const B: u32, const HINT: usize> Digits<B, HINT> {
    /// Assumes the passed digits are big endian and reverses them
    pub fn new_be<V: Into<Vec<u32>>>(n: V) -> Self {
        debug_assert!(B >= 2);
        let mut digits = n.into();
        digits.reverse();
        Self { digs: digits }
    }

    /// Assumes the passed digits are little endian and keeps them as is
    pub fn new_le<V: Into<Vec<u32>>>(n: V) -> Self {
        debug_assert!(B >= 2);
        let digits = n.into();
        Self { digs: digits }
    }

    /// Reconstructs the number from the digits in the given base
    pub fn into_num<T>(&self) -> T
    where
        T: Zero + From<u32>,
        for<'a> T: ops::AddAssign<&'a T>,
        for<'a> T: ops::MulAssign<&'a T>,
    {
        let mut n = T::zero();
        let base = T::from(B);
        for &d in self.digs.iter().rev() {
            n *= &base;
            let tmp: T = d.into();
            n += &tmp;
        }
        n
    }

    /// Reconstructes a sub-run of the digits as a number
    pub fn into_num_slice<T>(&self, start: usize, end: usize) -> T
    where
        T: Zero + From<u32>,
        for<'a> T: ops::AddAssign<&'a T>,
        for<'a> T: ops::MulAssign<&'a T>,
    {
        let mut n = T::zero();
        let base = T::from(B);
        for &d in self.digs.iter().rev().skip(start).take(end - start) {
            n *= &base;
            let tmp: T = d.into();
            n += &tmp;
        }
        n
    }

    pub fn nth_digit(&self, n: usize) -> Option<u32> {
        if n < self.digs.len() {
            Some(self.digs[self.digs.len() - 1 - n])
        } else {
            None
        }
    }

    /// Most significant digit (MSD) of the number represented by the digits.
    pub fn msd(&self) -> u32 {
        if let Some(&d) = self.digs.last() {
            d
        } else {
            0
        }
    }
}

impl<const B: u32, const HINT: usize> Deref for Digits<B, HINT> {
    type Target = [u32];
    fn deref(&self) -> &Self::Target {
        &self.digs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_digits() {
        let d: Digits<10> = Digits::from(&12345u32);
        // we expect the digits to be in reverse order
        assert_eq!(d.digs, vec![5, 4, 3, 2, 1]);
        let d: Digits<2> = Digits::from(&13u32);
        assert_eq!(d.digs, vec![1, 0, 1, 1]);
    }

    #[test]
    fn test_into_num() {
        let d: Digits<10> = Digits::from(&12345u32);
        assert_eq!(d.into_num::<u32>(), 12345);
        let d: Digits<2> = Digits::from(&13u32);
        assert_eq!(d.into_num::<u64>(), 13);
        let d: Digits<16> = Digits::from(&0u32);
        assert_eq!(d.into_num::<rug::Integer>(), 0);
    }

    #[test]
    fn test_into_num_slice() {
        let d: Digits<10> = Digits::from(&12345u32);
        assert_eq!(d.into_num_slice::<u32>(0, 5), 12345);
        assert_eq!(d.into_num_slice::<u32>(1, 4), 234);
        let d: Digits<2> = Digits::from(&0b01101u32);
        assert_eq!(d.into_num_slice::<u64>(0, 8), 0b1101);
        assert_eq!(d.into_num_slice::<u64>(1, 3), 0b10);
    }
}
