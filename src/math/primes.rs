// Most of this module is going to be a shameless ripoff of
// labmath (https://pypi.org/project/labmath/)

// def primegen(limit=inf):
// # We don't sieve 2, so we ought to be able to get sigificant savings by halving the length of the sieve.
// # But the tiny extra computation involved in that seems to exceed the savings.
// yield from takewhile(lambda x: x < limit, (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47))
// pl, pg = [3,5,7], primegen()
// for p in pl: next(pg)
// n = next(pg); nn = n*n
// while True:
//     n = next(pg)
//     ll, nn = nn, n*n
//     sl = (nn - ll)
//     sieve = bytearray([True]) * sl
//     for p in pl:
//         k = (-ll) % p
//         sieve[k::p] = bytearray([False]) * ((sl-k)//p + 1)
//     if nn > limit: break                                            # TODO bring this condition up to the while statement
//     yield from compress(range(ll,ll+sl,2), sieve[::2])
//     pl.append(n)
// yield from takewhile(lambda x: x < limit, compress(range(ll,ll+sl,2), sieve[::2]))

use fixedbitset::FixedBitSet;
use num_traits::{Euclid, Zero};
const INIT_PRIMES: &[usize] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
const BASE_PRIMES: &[usize] = &[3, 5, 7];

pub trait PG:
    Euclid
    + Zero
    + From<usize>
    + Copy
    + std::ops::Mul<Output = Self>
    + std::ops::Sub<Output = Self>
    + TryInto<usize>
{
}
impl<
    T: Euclid
        + Zero
        + From<usize>
        + Copy
        + std::ops::Mul<Output = Self>
        + std::ops::Sub<Output = Self>
        + TryInto<usize>,
> PG for T
{
}

struct PrimeGenSieve<T: PG> {
    pl: Vec<usize>,
    // we persist per-prime indices to only do the modulo computation
    // once per prime
    ks: Vec<usize>,
    pg: Box<PrimeGen<T>>,
    ll: T,
    n: T,
    nn: T,
    sieve_vec: FixedBitSet,
    ix: usize,
}

impl<T: PG> PrimeGenSieve<T> {
    pub fn new() -> Self {
        let mut pg = Box::new(PrimeGen::<T>::new());
        for _ in 0..BASE_PRIMES.len() {
            let _ = pg.next();
        }
        let n0 = pg.next().unwrap();
        // we don't know the segment end, but do know the segment start
        let ll0 = n0 * n0;

        let ks0: Vec<usize> = BASE_PRIMES
            .iter()
            .map(|&p| Self::compute_init_k(p.into(), ll0))
            .collect();

        let mut dummy = PrimeGenSieve {
            pl: BASE_PRIMES.to_vec().into_iter().map(|x| x.into()).collect(),
            ks: ks0,
            pg,
            ll: ll0,
            nn: ll0, // dummy value to be instarotated
            n: n0,   // dummy value to be instarotated
            ix: 0,
            // the fields below are dummies and are only filled on the first rotate
            sieve_vec: FixedBitSet::with_capacity(0),
        };

        dummy.rotate_segments();
        dummy
    }

    fn compute_init_k(p: T, ll: T) -> usize {
        let p_usize: usize = p.try_into().ok().unwrap();
        let mut k = (p - (ll % p)).try_into().ok().unwrap() % p_usize;
        // if we have an odd naive index that means our starting point was even
        // so we go to the next one. look ma, no branches!
        k += (k % 2) * p_usize;
        k / 2
    }

    fn rotate_segments(&mut self) {
        self.ll = self.nn;
        self.n = self.pg.next().unwrap();
        self.nn = self.n * self.n;

        // unlike in the original primegen we do implement the compression by 2
        let sl_usize = (self.nn - self.ll).try_into().ok().unwrap() / 2;
        self.sieve_vec = FixedBitSet::with_capacity(sl_usize);

        // do the sieving! differences from labmmath: 1 = composite, we've culled events
        // TODO wheel30 this mofo
        let bits = self.sieve_vec.as_mut_slice();
        let word_bits = usize::BITS as usize;
        for i in 0..self.ks.len() {
            let mut k = self.ks[i];
            let p_usize: usize = self.pl[i];
            while k < sl_usize {
                let w = k / word_bits;
                let b = k % word_bits;
                unsafe {
                    *bits.get_unchecked_mut(w) |= 1usize << b;
                }
                k += p_usize;
            }
            self.ks[i] = k; // store the next index for this prime
        }
    }

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        while self.ix < self.sieve_vec.len() && self.sieve_vec[self.ix] {
            self.ix += 1;
        }
        if self.ix >= self.sieve_vec.len() {
            return None;
        }
        let res = self.ll + T::from(self.ix * 2);
        self.ix += 1;
        Some(res)
    }

    fn finalize_segment(&mut self) {
        let sl_usize = (self.nn - self.ll).try_into().ok().unwrap() / 2;
        // adjust our prime indices to be valid for the next segment
        for i in 0..self.ks.len() {
            self.ks[i] -= sl_usize;
        }
        // push the next prime and compute its starting index.
        // self.nn will be the ll of the next segment, so we use it
        self.pl.push(self.n.try_into().ok().unwrap());
        self.ks.push(Self::compute_init_k(self.n, self.nn));
        self.ix = 0;
    }
}

enum PrimegenState<T: PG> {
    Init(usize),
    Segment(PrimeGenSieve<T>),
}

/// More or less a line-for-line re-implementation of labmath's primegen
/// we'll tweak and benchmark it later
pub struct PrimeGen<T: PG> {
    state: PrimegenState<T>,
}

impl<T: PG> PrimeGen<T> {
    pub fn new() -> Self {
        PrimeGen {
            state: PrimegenState::Init(0),
        }
    }
}

impl<T: PG> Default for PrimeGen<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PG> Iterator for PrimeGen<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                PrimegenState::Init(ix) => {
                    if *ix >= INIT_PRIMES.len() {
                        let segment = PrimeGenSieve::new();
                        self.state = PrimegenState::Segment(segment);
                        continue;
                    } else {
                        let out = INIT_PRIMES[*ix].into();
                        *ix += 1; // increment the index
                        return Some(out);
                    }
                }
                PrimegenState::Segment(sieve) => match sieve.next() {
                    Some(prime) => {
                        return Some(prime);
                    }
                    None => {
                        sieve.finalize_segment();
                        sieve.rotate_segments();
                        continue;
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_primegen() {
        let pg: Vec<usize> = PrimeGen::<usize>::new().take(100).collect();
        let cmp: Vec<usize> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
            181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
            277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
            383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479,
            487, 491, 499, 503, 509, 521, 523, 541,
        ];
        assert_eq!(pg, cmp);
    }
}
