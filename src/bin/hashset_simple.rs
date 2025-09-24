use std::time::Instant;

use log::LevelFilter;
use oram::hashset::OramHashSet;
use oram::{Oram, path_oram::LINEAR_TIME_ORAM_CUTOFF};
use rand::{CryptoRng, Rng, RngCore, distr::StandardUniform, rng};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

const ARRAY_SIZE: u64 = 4096;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

fn benchmark_lookups<R: RngCore + CryptoRng>(oram_hash_set: &mut OramHashSet<u64>, rng: &mut R) {
    for _ in 0..20 {
        let search_val = rng.random::<u64>();

        let start = Instant::now();

        let found = oram_hash_set.contains(search_val, rng).unwrap();

        let duration = start.elapsed();

        println!("Got {:?} in {:?}", found, duration);
    }
}

fn main() {
    SimpleLogger::init(LevelFilter::Trace, simplelog::Config::default()).unwrap();

    let mut rng = rng();

    let mut oram_hash_set = OramHashSet::<u64>::new(ARRAY_SIZE, &mut rng).unwrap();

    let values = (&mut rng)
        .sample_iter(StandardUniform)
        .take((ARRAY_SIZE / 4) as usize)
        .collect::<Vec<u64>>();

    let start = Instant::now();

    for value in &values {
        oram_hash_set.insert(*value, &mut rng).unwrap();
    }

    let duration = start.elapsed();

    println!("Initialized ORAM hashset in {:?}", duration);

    println!("ORAM on:");

    benchmark_lookups(&mut oram_hash_set, &mut rng);

    println!("ORAM off:");

    oram_hash_set.array.turn_off().unwrap();

    benchmark_lookups(&mut oram_hash_set, &mut rng);

    let start = Instant::now();

    oram_hash_set.array.turn_on(&mut rng).unwrap();

    let duration = start.elapsed();

    println!("Turned on in {:?}", duration);
}
