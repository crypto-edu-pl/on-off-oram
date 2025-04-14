use std::{
    cmp::{max, min, Ordering},
    time::Instant,
};

use log::LevelFilter;
use rand::{distributions::Uniform, prelude::Distribution, rngs::OsRng, CryptoRng, RngCore};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Address, Oram, OramError};

#[cfg(not(feature = "bypass_oram"))]
use oram::DefaultOram;

#[cfg(feature = "bypass_oram")]
use oram::not_really_oram::NotReallyOram;

const ARRAY_SIZE: u64 = 1 << 17;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

fn benchmark_searches<O: Oram<V = u64>, R: RngCore + CryptoRng>(
    oram_array: &mut O,
    distribution: Uniform<u64>,
    rng: &mut R,
) -> Result<Vec<Option<Address>>, OramError> {
    const N_SEARCHES: usize = 20;

    let searched_values = distribution
        .sample_iter(&mut *rng)
        .take(N_SEARCHES)
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(N_SEARCHES);

    let start = Instant::now();

    for search_val in searched_values {
        let mut l = 0;
        let mut r = ARRAY_SIZE - 1;
        let mut found = None;

        // Run for a fixed number of iterations to prevent leakage from the number of iterations and keep execution times consistent
        for _ in 0..ARRAY_SIZE.next_power_of_two().ilog2() {
            let mid = (l + r) / 2;
            let val = oram_array.read(mid, rng)?;

            match val.cmp(&search_val) {
                Ordering::Less => l = min(mid + 1, r),
                Ordering::Greater => r = max(mid - 1, l),
                Ordering::Equal => found = Some(mid),
            }
        }

        result.push(found);
    }

    let duration = start.elapsed();

    println!("Searched for {} values in {:?}", N_SEARCHES, duration);

    Ok(result)
}

fn main() {
    SimpleLogger::init(LevelFilter::Trace, simplelog::Config::default()).unwrap();

    let mut rng = OsRng;

    let start = Instant::now();

    let mut oram_array = {
        #[cfg(not(feature = "bypass_oram"))]
        {
            DefaultOram::<u64>::new(ARRAY_SIZE, &mut rng).unwrap()
        }

        #[cfg(feature = "bypass_oram")]
        {
            NotReallyOram::<u64>::new(ARRAY_SIZE).unwrap()
        }
    };

    let duration = start.elapsed();

    println!("Initialized ORAM in {:?}", duration);

    let distribution = Uniform::from(0..ARRAY_SIZE);

    let mut values = distribution
        .sample_iter(&mut rng)
        .take(ARRAY_SIZE as usize)
        .collect::<Vec<u64>>();
    values.sort();

    let start = Instant::now();

    for (i, value) in values.iter().enumerate() {
        oram_array.write(i as Address, *value, &mut rng).unwrap();
    }

    let duration = start.elapsed();

    println!("Prepared array in {:?}", duration);

    println!("ORAM on:");

    benchmark_searches(&mut oram_array, distribution, &mut rng).unwrap();

    println!("ORAM off:");

    oram_array.turn_off().unwrap();

    benchmark_searches(&mut oram_array, distribution, &mut rng).unwrap();

    let start = Instant::now();

    oram_array.turn_on(&mut rng).unwrap();

    let duration = start.elapsed();

    println!("Turned on in {:?}", duration);
}
