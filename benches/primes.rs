use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_factoring_1m(c: &mut Criterion) {
    c.bench_function("100k_divisors", |b| {
        b.iter(|| {
            for n in (666..=1_337_000_000u64).step_by(13370) {
                black_box(pe_lib::math::int::divisors(n));
            }
        })
    });

    c.bench_function("100k_factorint", |b| {
        b.iter(|| {
            for n in (666..=1_337_000_000u64).step_by(13370) {
                black_box(pe_lib::math::int::Factored::new(n));
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
    targets = bench_factoring_1m
);
criterion_main!(benches);
