use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pe_lib::sieve::{DiscriminatedUnion, linear_sieve, linear_sieve_completely_multiplicative};

fn bench_sieve(c: &mut Criterion) {
    c.bench_function("first_1m_factorizations", |b| {
        b.iter(|| {
            for pvec in
                linear_sieve_completely_multiplicative::<_, DiscriminatedUnion, _>(1_000_000, |p| {
                    vec![p]
                })
            {
                black_box(pvec);
            }
        })
    });

    c.bench_function("first_100m_totient", |b| {
        b.iter(|| {
            for totient in linear_sieve(100_000_000, |p| p - 1, |_, p, f_i| p * f_i) {
                black_box(totient);
            }
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = bench_sieve
);
criterion_main!(benches);
