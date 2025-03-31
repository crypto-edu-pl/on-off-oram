use std::{
    cmp::{max, min, Ordering},
    time::Instant,
};

use rand::{distributions::Standard, rngs::OsRng, CryptoRng, Rng, RngCore};
use static_assertions::const_assert;

use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Address, DefaultOram, Oram};

const ARRAY_SIZE: u64 = 4096;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

fn benchmark_searches<O: Oram<V = u64>, R: RngCore + CryptoRng>(oram_array: &mut O, rng: &mut R) {
    for _ in 0..20 {
        let search_val = rng.gen::<u64>();

        let mut l = 0;
        let mut r = ARRAY_SIZE - 1;
        let mut found = None;

        let start = Instant::now();

        // Run for a fixed number of iterations to prevent leakage from the number of iterations and keep execution times consistent
        for _ in 0..ARRAY_SIZE.next_power_of_two().ilog2() {
            let mid = (l + r) / 2;
            let val = oram_array.read(mid, rng).unwrap();

            match val.cmp(&search_val) {
                Ordering::Less => l = min(mid + 1, r),
                Ordering::Greater => r = max(mid - 1, l),
                Ordering::Equal => found = Some(mid),
            }
        }
        let duration = start.elapsed();

        println!("Found {:?} in {:?}", found, duration);
    }
}

fn main() {
    let mut rng = OsRng;

    let mut oram_array = DefaultOram::<u64>::new(ARRAY_SIZE, &mut rng).unwrap();

    let mut values = (&mut rng)
        .sample_iter(Standard)
        .take(ARRAY_SIZE as usize)
        .collect::<Vec<u64>>();
    values.sort();

    for (i, value) in values.iter().enumerate() {
        oram_array.write(i as Address, *value, &mut rng).unwrap();
    }

    println!("ORAM on:");

    benchmark_searches(&mut oram_array, &mut rng);

    println!("ORAM off:");

    oram_array.turn_off().unwrap();

    benchmark_searches(&mut oram_array, &mut rng);

    let start = Instant::now();

    oram_array.turn_on(&mut rng).unwrap();

    let duration = start.elapsed();

    println!("Turned on in {:?}", duration);
}
