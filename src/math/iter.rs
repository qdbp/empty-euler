#[derive(Clone, Debug)]
pub struct LazyVec<I, T> {
    src: I,
    buf: Vec<T>,
}

impl<I, T> LazyVec<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn new(src: I) -> Self {
        Self {
            src,
            buf: Vec::new(),
        }
    }

    pub fn with_capacity(src: I, cap: usize) -> Self {
        Self {
            src,
            buf: Vec::with_capacity(cap),
        }
    }

    /// Ensure at least `n` elements are cached. Returns false if source ends.
    pub fn ensure_len(&mut self, n: usize) -> bool {
        while self.buf.len() < n {
            match self.src.next() {
                Some(x) => self.buf.push(x),
                None => return false,
            }
        }
        true
    }

    /// Random access that grows on demand.
    #[inline(always)]
    pub fn at(&mut self, i: usize) -> Option<&T> {
        if self.ensure_len(i + 1) {
            Some(&self.buf[i])
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn demand(&mut self, i: usize) -> &T {
        self.at(i).unwrap()
    }

    /// Current cached prefix as a slice (no growth).
    pub fn cached(&self) -> &[T] {
        &self.buf
    }

    /// Grow to exactly `n` and return that prefix.
    pub fn view_to(&mut self, n: usize) -> Option<&[T]> {
        if self.ensure_len(n) {
            Some(&self.buf[..n])
        } else {
            None
        }
    }
}

impl<I, T: Clone> LazyVec<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn at_cloned(&mut self, i: usize) -> Option<T> {
        self.at(i).cloned()
    }

    pub fn demand_cloned(&mut self, i: usize) -> T {
        self.demand(i).clone()
    }
}
