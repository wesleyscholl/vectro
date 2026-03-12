//! Algorithm throughput benchmarks: INT8, NF4, HNSW.
//!
//! Run with:
//!   cargo bench -p vectro_lib --bench simd_bench
//!
//! Throughput is measured in elements/second.  Divide by D to get vec/s.
//! Phase-17 targets (PLAN.md):
//!   INT8 encode: ≥ 12M vec/s @ n=100K, d=768 (≈ 9.2 Gelem/s)
//!   NF4 encode:  ≥  2M vec/s @ d=768
//!   HNSW recall@10: reported by `recall_at_k_bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use vectro_lib::quant::{int8, nf4};
use vectro_lib::index::hnsw::HnswIndex;

fn make_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.0013_f32).sin()).collect())
        .collect()
}

/// INT8 encode throughput at benchmark scale.
fn bench_int8_throughput(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("int8_throughput");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_batch_n1000_d768", |b| {
        b.iter(|| int8::encode_batch(black_box(&vecs)))
    });
    group.finish();
}

/// NF4 encode throughput.
fn bench_nf4_throughput(c: &mut Criterion) {
    const N: usize = 1_000;
    const D: usize = 768;
    let vecs = make_vecs(N, D);

    let mut group = c.benchmark_group("nf4_throughput");
    group.throughput(Throughput::Elements((N * D) as u64));
    group.bench_function("encode_batch_n1000_d768", |b| {
        b.iter(|| nf4::encode_batch(black_box(&vecs)))
    });
    group.finish();
}

/// HNSW search throughput (index pre-built outside the timed loop).
fn bench_hnsw_search(c: &mut Criterion) {
    const N: usize = 2_000;
    const D: usize = 64;
    let vecs = make_vecs(N, D);
    let query = vecs[0].clone();

    let mut idx = HnswIndex::new(8, 40);
    idx.add_batch(&vecs);

    let mut group = c.benchmark_group("hnsw_search");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("search_k10_ef50_n2000_d64", |b| {
        b.iter(|| idx.search(black_box(&query), 10, 50))
    });
    group.finish();
}

criterion_group!(benches, bench_int8_throughput, bench_nf4_throughput, bench_hnsw_search);
criterion_main!(benches);
