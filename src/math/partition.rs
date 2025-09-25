use std::ops::{Deref, DerefMut};

use rug::ops::Pow;
use rug::{Complete, Integer as Int};

/// A partition of an integer into a sum of positive integers.
/// Guarantees: the parts vector is in non-decreasing order, and all parts are positive.
/// Because cycle types of Sn are integer partitions of n, many cycle-type related functions
/// live here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IntPartition {
    pub parts: Vec<u32>,
}

impl IntPartition {
    pub fn new(parts: impl Into<Vec<u32>>) -> Self {
        let mut parts = parts.into();
        parts.sort_unstable();
        assert!(parts.iter().all(|&x| x > 0));
        Self { parts }
    }

    pub fn new_unchecked(parts: Vec<u32>) -> Self {
        Self { parts }
    }

    /// The number being partitioned.
    pub fn n(&self) -> u32 {
        self.iter().sum()
    }

    /// Returns the number of elements of Sn with this cycle type.
    pub fn count_permutations_with_this_cycle_type(&self) -> Int {
        let n = self.n();
        let mut out = Int::factorial(n).complete();
        let mut m_i = 1;
        // never valid to have 0 as a part, so this is safe
        let mut i_prev = 0u32;
        for &i in &self.parts {
            if i == i_prev {
                m_i += 1;
            } else {
                out /= Int::factorial(m_i).complete();
                out /= Int::from(i).pow(m_i);
                m_i = 1;
                i_prev = i;
            }
        }
        out
    }

    /// Returns true if all parts are distinct.
    pub fn is_distinct(&self) -> bool {
        self.windows(2).all(|w| w[0] != w[1])
    }

    /// Returns the least common multiple of the parts.
    pub fn lcm(&self) -> Int {
        let mut out = Int::from(1);
        for cl in &self.parts {
            out.lcm_u_mut(*cl);
        }
        out
    }

    /// Returns the greatest common divisor of the parts.
    pub fn gcd(&self) -> u32 {
        // guaranteed to be the largest
        let mut out = Int::from(self.parts[self.parts.len() - 1]);
        for cl in &self.parts[..self.parts.len() - 1] {
            out.gcd_u_mut(*cl);
        }
        // guaranteed to fit in u32
        out.to_u32_wrapping()
    }

    /// Uses the AccelAsc algorithm to visit all partitions of n in ascending order.
    pub fn visit_asc(n: u32, mut f: impl FnMut(&Self)) {
        if n == 0 {
            f(&Self::new_unchecked(vec![]));
            return;
        }

        let mut a = Self::new_unchecked(vec![0u32; n as usize]);
        let mut k = 1usize;
        let mut y: u32 = n - 1;

        while k != 0 {
            let mut x = a[k - 1] + 1;
            k -= 1;
            while 2 * x <= y {
                a[k] = x;
                y -= x;
                k += 1;
            }
            let t = k + 1;
            while x <= y {
                a[k] = x;
                a[t] = y;
                a.parts.truncate(t + 1);
                f(&a);
                x += 1;
                y -= 1;
            }
            y += x - 1;
            a[k] = y + 1;
            a.parts.truncate(k + 1);
            f(&a);
        }
    }

    /// Generates all partitions of n in ascending order.
    pub fn gen_acc(n: u32) -> Vec<Self> {
        let mut out = vec![];
        Self::visit_asc(n, |p| out.push(p.clone()));
        out
    }

    /// Uses the AccelAscDistinct algorithm to visit all partitions of n into distinct parts in ascending order.
    pub fn visit_distinct_asc(n: u32, mut f: impl FnMut(&Self)) {
        if n == 0 {
            f(&Self::new_unchecked(vec![]));
            return;
        }

        let mut a = Self::new_unchecked(vec![0u32; (n as usize).max(2usize)]);
        let len = a.parts.len();
        a[0] = 0;
        a[1] = n;
        let mut k = 1usize;

        while k != 0 {
            let mut y = a[k] - 1;
            k -= 1;
            let mut x = a[k] + 1;

            while 2 * x + 3 <= y {
                a[k] = x;
                x += 1;
                y -= x;
                k += 1;
            }

            let l = k + 1;
            while x < y {
                a[k] = x;
                a[l] = y;

                unsafe {
                    a.parts.set_len(l + 1);
                }
                f(&a);
                unsafe {
                    a.parts.set_len(len);
                }
                x += 1;
                y -= 1;
            }

            a[k] = x + y;
            unsafe {
                a.parts.set_len(k + 1);
            }
            f(&a);
            unsafe {
                a.parts.set_len(len);
            }
        }
    }

    pub fn gen_distinct_acc(n: u32) -> Vec<Self> {
        let mut out = vec![];
        Self::visit_distinct_asc(n, |p| out.push(p.clone()));
        out
    }
}

impl Deref for IntPartition {
    type Target = [u32];
    fn deref(&self) -> &Self::Target {
        &self.parts
    }
}

impl DerefMut for IntPartition {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_acc() {
        let parts = IntPartition::gen_acc(5);
        assert_eq!(parts.len(), 7);
        assert_eq!(parts[0], IntPartition::new_unchecked(vec![1, 1, 1, 1, 1]));
        assert_eq!(parts[1], IntPartition::new_unchecked(vec![1, 1, 1, 2]));
        assert_eq!(parts[2], IntPartition::new_unchecked(vec![1, 1, 3]));
        assert_eq!(parts[3], IntPartition::new_unchecked(vec![1, 2, 2]));
        assert_eq!(parts[4], IntPartition::new_unchecked(vec![1, 4]));
        assert_eq!(parts[5], IntPartition::new_unchecked(vec![2, 3]));
        assert_eq!(parts[6], IntPartition::new_unchecked(vec![5]));
    }
    #[test]
    fn test_distinct_acc_0() {
        assert_eq!(
            IntPartition::gen_distinct_acc(0),
            vec![IntPartition::new_unchecked(vec![])]
        );
    }
    #[test]
    fn test_distinct_acc_1() {
        assert_eq!(
            IntPartition::gen_distinct_acc(1),
            vec![IntPartition::new_unchecked(vec![1])]
        );
    }

    #[test]
    fn test_distinct_acc_2() {
        assert_eq!(
            IntPartition::gen_distinct_acc(2),
            vec![IntPartition::new_unchecked(vec![2])]
        );
    }

    #[test]
    fn test_distinct_acc_7() {
        let parts = IntPartition::gen_distinct_acc(8);
        assert_eq!(parts[0], IntPartition::new_unchecked(vec![1, 2, 5]));
        assert_eq!(parts[1], IntPartition::new_unchecked(vec![1, 3, 4]));
        assert_eq!(parts[2], IntPartition::new_unchecked(vec![1, 7]));
        assert_eq!(parts[3], IntPartition::new_unchecked(vec![2, 6]));
        assert_eq!(parts[4], IntPartition::new_unchecked(vec![3, 5]));
        assert_eq!(parts[5], IntPartition::new_unchecked(vec![8]));
    }

    // [1,2,6], [1,3,5], [1,8], [2,3,4], [2,7], [3,6], [4,5], [9].
    #[test]
    fn test_distinct_acc_9() {
        let parts = IntPartition::gen_distinct_acc(9);
        assert_eq!(parts.len(), 8);
        assert_eq!(parts[0], IntPartition::new_unchecked(vec![1, 2, 6]));
        assert_eq!(parts[1], IntPartition::new_unchecked(vec![1, 3, 5]));
        assert_eq!(parts[2], IntPartition::new_unchecked(vec![1, 8]));
        assert_eq!(parts[3], IntPartition::new_unchecked(vec![2, 3, 4]));
        assert_eq!(parts[4], IntPartition::new_unchecked(vec![2, 7]));
        assert_eq!(parts[5], IntPartition::new_unchecked(vec![3, 6]));
        assert_eq!(parts[6], IntPartition::new_unchecked(vec![4, 5]));
        assert_eq!(parts[7], IntPartition::new_unchecked(vec![9]));
    }
}
