use itertools::Itertools;
use ndarray::{s, Array1, Array2};
use num_traits::Euclid;
use rug::Integer as Int;

use crate::math::util::int;

type Mat = Array2<Int>;

/// a k-regular sequence
/// Suppose a_{n*m + r} = Σ{j=0..=t} C_rj a_{m+j} + d_r, 0 <= r < n
///
/// If we define s(m) = [a_m, a_{m+1}, ..., a_{m+t}]^T
/// Then there exists a set of matrices M_r for r ∈ {0..n-1} such that
/// s(m*n + r) = M_r * s(m) + β_r
///
/// Then we can compute s(x) for any x using the decomposition of x in base n.
/// Summatory functions also follow a standard form if we know M_r and β_r.
///
/// For now only the basic case where j ∈ {0..<n} is supported.
/// References
pub struct KReg {
    basis: usize,
    mats: Vec<Mat>,
    seed: Vec<Int>,
    // needed for summatory functions, where the block matrix formulation introducdes
    // a single-element lag.
    at_offset: usize,
}

impl KReg {
    /// Constructs the internal transition matrices from the recurrent representation:
    ///
    /// C and d are the matrix and vector such that our defining equations are of the form
    /// a_{nm + r} = ΣC_rj a_{m+j} + d_r, 0 <= r < n
    ///
    /// The seed is the initial values of the sequence. The minimum number of seed values is
    /// a property of the recurrence; construction will panic if the seed is insufficient
    /// to define all terms for the given C and d.
    ///
    /// We expect C to have shape (=N, <N), and d to have length N
    pub fn from_rec<T: Clone + Into<Int>>(c: Array2<T>, d: Array1<T>, seed: &[T]) -> Self {
        let n = c.shape()[0];
        let t = c.shape()[1] - 1;
        let L = n + t;

        assert!(c.shape()[0] == d.shape()[0]);

        let c: Array2<Int> = c.mapv(Into::into);
        let d: Array1<Int> = d.mapv(Into::into);

        // + 1 for the affine part
        let mut mats = Vec::<Mat>::with_capacity(n);
        let mut seed = seed.iter().cloned().map_into().collect_vec();

        // need to extend the seed to at least length t, just apply the recurrence directly here
        for ix in seed.len()..L {
            // seq_ix is the sequence-space index. residue computations involving r use this index,
            // not the vector index, which is just a storage detail.
            let row = ix % n;
            let base_ix = ix / n;
            let mut next = int(0);
            for (j, c_jr) in c.row(row).iter().enumerate() {
                // this check is not just an optimization: we need this to avoid out of bounds
                if c_jr.is_zero() {
                    continue;
                }
                next += c_jr * &seed[base_ix + j];
                next += &d[row];
            }
            seed.push(next);
        }
        seed.push(int(1)); // homogeneous part

        // the conversion of the C matrices into M for this simple case is routine
        // coordinate shuffling.
        for r in 0..n {
            mats.push(Array2::<Int>::zeros((L + 1, L + 1)));
            let Mr = mats.last_mut().unwrap();
            for row in 0..L {
                let row_seq_ix = row;
                let m_col = (r + row_seq_ix) / n;
                let c_row = (r + row_seq_ix) % n;

                let mut target = Mr.slice_mut(s![row, m_col..=(m_col + t)]);
                target.assign(&c.slice(s![c_row, ..]));

                Mr[(row, L)] = d[c_row].clone();
            }
            Mr[(L, L)] = Int::from(1);
        }
        Self {
            basis: n,
            mats,
            seed,
            at_offset: 0,
        }
    }

    /// Computes a_n for the given n
    pub fn at(&self, n: usize) -> Int {
        let mut x = n + self.at_offset;
        let mut r: usize;
        let mut mats = vec![];
        let mut rs = vec![];
        while x > 0 {
            (x, r) = x.div_rem_euclid(&self.basis);
            mats.push(&self.mats[r]);
            rs.push(r);
        }
        mats.reverse();
        let mut s = self.seed.clone();

        for m in mats {
            // can't use .dot because LinalgScalar : Copy lmao
            let mut next_s = vec![int(0); s.len()];
            for i in 0..s.len() {
                for j in 0..s.len() {
                    next_s[i] += &m[(i, j)] * &s[j];
                }
            }
            s = next_s;
        }

        s[0].clone()
    }
    /// Partial sums of k-regular sequences are also k-regular sequences.
    ///
    /// This method constructs the k-regular sequence `seq` to the summatory function such that
    /// seq.at(n) = Σ{j=0..=n} a_j, where a_j is the original sequence.
    pub fn summatory(&self) -> Self {
        // we need to construct matrices A, B_r, C_r as follows
        let w = self.mats[0].shape()[0];
        let C = self.mats.iter().cloned().reduce(|a, b| a + b).unwrap();
        let mut Bs = vec![Mat::zeros((w, w))];
        // exclude the last one, since we need an exclusive sum
        for mat in self.mats[0..self.mats.len() - 1].iter() {
            let next = mat + &Bs[Bs.len() - 1];
            Bs.push(next);
        }

        // matrix layout for Mnew_r is:
        // [ C  B_r ]
        // [ 0  M_r ]
        let new_mats = (0..self.mats.len())
            .map(|i| {
                let mut m = Mat::zeros((2 * w, 2 * w));
                m.slice_mut(s![0..w, 0..w]).assign(&C);
                m.slice_mut(s![0..w, w..(2 * w)]).assign(&Bs[i]);
                m.slice_mut(s![w..(2 * w), w..(2 * w)])
                    .assign(&self.mats[i]);
                m
            })
            .collect_vec();

        let new_seed = vec![int(0); w]
            .into_iter()
            .chain(self.seed.iter().cloned())
            .collect();

        Self {
            basis: self.basis,
            mats: new_mats.to_vec(),
            seed: new_seed,
            at_offset: self.at_offset + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::util::int;

    use super::*;
    use ndarray::array;
    #[test]
    fn test_kreg_init_homogeneous() {
        // Example from Allouche-Shallit, Theorem 16.1.2
        // a_{2n} = a_n
        // a_{2n+1} = a_n + a_{n+1}
        let c = array![[1, 0], [1, 1]];
        let d = array![0, 0];
        let seed = vec![0, 1]; // a_0 = 0, a_1 = 1
        let kreg = KReg::from_rec(c, d, &seed);
        // Check the matrices -- homogeneous form
        let m0_expected = array![[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]];
        let m1_expected = array![[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]];
        assert_eq!(kreg.mats[0], m0_expected);
        assert_eq!(kreg.mats[1], m1_expected);
        // Check the seed
        assert_eq!(kreg.seed, vec![int(0), int(1), int(1), int(1)]);
    }

    #[test]
    fn test_kreg_init_general() {
        // Modified:
        // a_{2n} = a_n + 3
        // a_{2n+1} = a_n + a_{n+1}
        let c = array![[1, 0], [1, 1]];
        let d = array![3, 0];

        // this is sufficient -- the recurrence shall be extended twice
        let seed = vec![0, 1]; // a_0 = 0, a_1 = 1
        let kreg = KReg::from_rec(c, d, &seed);

        assert_eq!(kreg.basis, 2);

        // Check the matrices -- homogeneous form
        let m0_expected = array![[1, 0, 0, 3], [1, 1, 0, 0], [0, 1, 0, 3], [0, 0, 0, 1]];
        let m1_expected = array![[1, 1, 0, 0], [0, 1, 0, 3], [0, 1, 1, 0], [0, 0, 0, 1]];
        assert_eq!(kreg.mats[0], m0_expected);
        assert_eq!(kreg.mats[1], m1_expected);

        // Check the seed
        assert_eq!(kreg.seed, vec![int(0), int(1), int(4), int(1)]);
    }

    #[test]
    fn test_kreg_at() {
        // Example from Allouche-Shallit, Theorem 16.1.2
        // a_{2n} = a_n
        // a_{2n+1} = a_n + a_{n+1}
        let c = array![[1, 0], [1, 1]];
        let d = array![0, 0];
        let seed = vec![0, 1]; // a_0 = 0, a_1 = 1
        let mut expected = vec![0, 1];
        for j in 2..20 {
            let val = if j % 2 == 0 {
                expected[j / 2]
            } else {
                expected[j / 2] + expected[j / 2 + 1]
            };
            expected.push(val);
        }
        let kreg = KReg::from_rec(c, d, &seed);
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(kreg.at(i), int(e), "mismatch at i={}", i);
        }
    }

    #[test]
    fn test_summatory() {
        // Example from Allouche-Shallit, Theorem 16.1.2
        // a_{2n} = a_n
        // a_{2n+1} = a_n + a_{n+1}
        let c = array![[1, 0], [1, 1]];
        let d = array![0, 0];
        let seed = vec![0, 1]; // a_0 = 0, a_1 = 1
        let mut expected = vec![0, 1];
        for j in 2..20 {
            let val = if j % 2 == 0 {
                expected[j / 2]
            } else {
                expected[j / 2] + expected[j / 2 + 1]
            };
            expected.push(val);
        }
        let summatory_expected = expected
            .iter()
            .scan(0, |state, &x| {
                *state += x;
                Some(*state)
            })
            .collect_vec();
        let kreg = KReg::from_rec(c, d, &seed);
        let sum_kreg = kreg.summatory();

        for (i, &e) in summatory_expected.iter().enumerate() {
            assert_eq!(sum_kreg.at(i), int(e), "mismatch at i={}", i);
        }
    }
}
