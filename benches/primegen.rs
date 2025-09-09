use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pe_lib::math::primes::PrimeGen;

fn bench_first_10m_primes(c: &mut Criterion) {
    c.bench_function("first_10m_primes_32_collect", |b| {
        b.iter(|| {
            let v: Vec<u32> = PrimeGen::<u32>::new().take(10_000_000).collect();
            black_box(v);
        })
    });

    // lower allocation noise: consume without collect
    c.bench_function("first_10m_primes_u32_iter", |b| {
        b.iter(|| {
            let mut pg = PrimeGen::<u32>::new();
            for _ in 0..10_000_000 {
                black_box(pg.next().unwrap());
            }
        })
    });

    // lower allocation noise: consume without collect
    c.bench_function("first_10m_primes_rug_iter", |b| {
        b.iter(|| {
            let mut pg = PrimeGen::<rug::Integer>::new();
            for _ in 0..10_000_000 {
                black_box(pg.next().unwrap());
            }
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)              // tune as desired
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_first_10m_primes
);
criterion_main!(benches);
