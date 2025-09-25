use rug::Integer as Int;
use std::ops::{Add, Neg};

use crate::partition::IntPartition;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An element of the symmetric group Sn
/// For many methods pertaining to cycle types, see the IntPartition struct.
pub struct S<const N: usize> {
    /// One-line notation representation of the permutation
    data: [u32; N],
}

impl<const N: usize> S<N> {
    /// Create a new permutation from an array
    pub fn new(data: [u32; N]) -> Self {
        // Check that data is a valid permutation
        let mut seen = [false; N];
        for &x in &data {
            if x as usize >= N || seen[x as usize] {
                panic!("Invalid permutation");
            }
            seen[x as usize] = true;
        }
        Self::new_unchecked(data)
    }

    pub fn identity() -> Self {
        let mut data = [0u32; N];
        data.iter_mut()
            .enumerate()
            .for_each(|(i, σi)| *σi = i as u32);
        Self::new_unchecked(data)
    }

    #[inline]
    fn new_unchecked(data: [u32; N]) -> Self {
        Self { data }
    }

    /// The output should be interpreted as cycle notation; each sub-vector is a cycle. 1-cycles are included.
    #[allow(unused)]
    fn cycle_decomposition(&self) -> Vec<Vec<usize>> {
        let mut visited = [false; N];
        let mut cycles = Vec::new();
        for i in 0..N {
            if visited[i] {
                continue;
            }
            let mut cycle = vec![];
            let mut x = i;
            while !visited[x] {
                cycle.push(x);
                visited[x] = true;
                x = self.data[x] as usize;
            }
            cycles.push(cycle);
        }
        cycles
    }

    /// Faster than cycle_decomposition if you only need the lengths of the cycles
    #[allow(unused)]
    fn cycle_type(&self) -> IntPartition {
        let mut visited = [false; N];
        let mut cycle_lengths = Vec::new();
        for i in 0..N {
            if visited[i] {
                continue;
            }
            let mut length = 0;
            let mut x = i;
            while !visited[x] {
                length += 1;
                visited[x] = true;
                x = self.data[x] as usize;
            }
            cycle_lengths.push(length);
        }
        cycle_lengths.sort_unstable();
        IntPartition::new_unchecked(cycle_lengths)
    }

    #[allow(unused)]
    fn order(&self) -> Int {
        self.cycle_type().lcm()
    }
}

impl<const N: usize> Add for S<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = [0; N];
        for (i, σi) in result.iter_mut().enumerate() {
            *σi = self.data[other.data[i] as usize];
        }
        Self::new(result)
    }
}

impl<const N: usize> Neg for S<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut result = [0u32; N];
        for i in 0..N {
            result[self.data[i] as usize] = i as u32;
        }
        Self::new(result)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn test_permutation() {
        let p1 = S::<3>::new([2, 0, 1]);
        let p2 = S::<3>::new([1, 2, 0]);
        let p3 = p1 + p2;
        assert_eq!(p3.data, [0, 1, 2]);
        let p4 = -p1;
        assert_eq!(p4.data, [1, 2, 0]);
    }

    #[test]
    fn test_cycle_decomposition() {
        let cycles = S::new([0, 1, 2]).cycle_decomposition();
        assert_eq!(cycles, vec![vec![0], vec![1], vec![2]]);

        let cycles = S::new([2, 3, 4, 1, 0]).cycle_decomposition();
        assert_eq!(cycles, vec![vec![0, 2, 4], vec![1, 3]]);

        let cycles = S::new([3, 0, 1, 2]).cycle_decomposition();
        assert_eq!(cycles, vec![vec![0, 3, 2, 1]]);
    }

    #[test]
    fn test_order() {
        // (0)(1)(2) -> 1
        let p = S::new([0, 1, 2]);
        assert_eq!(p.order(), Int::from(1));
        // (0 2 4)(1 3) -> 6
        let p = S::new([2, 3, 4, 1, 0]);
        assert_eq!(p.order(), Int::from(6));
        // (0 3 2 1) -> 4
        let p = S::new([3, 0, 1, 2]);
        assert_eq!(p.order(), Int::from(4));
    }
}
