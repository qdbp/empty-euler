use std::ops::AddAssign;

use num_traits::Zero;

pub mod kreg;
pub mod lin;

/// Cumulative summation iterator adapter.
/// The summation is index-inclusive, i.e. the first output is the first input.
pub struct Summatory<I, T> {
    src: I,
    sum: T,
}

impl<I, T> Summatory<I, T>
where
    T: Zero,
{
    pub fn new(src: I) -> Self {
        Self {
            src,
            sum: T::zero(),
        }
    }
}

impl<I, T> Iterator for Summatory<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
    for<'z> T: AddAssign<&'z T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.src.next().map(|x| {
            self.sum += &x;
            self.sum.clone()
        })
    }
}

pub trait Summable: Sized {
    fn summatory<T>(self) -> Summatory<Self, T>
    where
        Self: Iterator<Item = T>,
        T: Zero,
    {
        Summatory::new(self)
    }
}

impl<T> Summable for T {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_summatory() {
        let v = vec![1, 2, 3, 4, 5];
        let mut it = v.into_iter().summatory();
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), Some(6));
        assert_eq!(it.next(), Some(10));
        assert_eq!(it.next(), Some(15));
        assert_eq!(it.next(), None);
    }
}
