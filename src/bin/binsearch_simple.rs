use std::{
    cmp::{max, min, Ordering},
    iter,
    time::{Duration, Instant},
};

use log::LevelFilter;
use rand::{distr::Uniform, prelude::Distribution, rng, CryptoRng, RngCore};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

use oram::{
    bin_utils::{benchmark_stats, BenchmarkResult, BenchmarkStats},
    path_oram::LINEAR_TIME_ORAM_CUTOFF,
    Address, Oram, OramError,
};

#[cfg(not(feature = "bypass_oram"))]
use oram::DefaultOram;

#[cfg(feature = "bypass_oram")]
use oram::not_really_oram::NotReallyOram;

const ARRAY_SIZE: u64 = 1 << 17;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

const N_SEARCHES: usize = 20;

const N_BENCHMARK_REPETITIONS: u32 = 20;

fn benchmark_searches<O: Oram<V = u64>, R: RngCore + CryptoRng>(
    oram_array: &mut O,
    distribution: Uniform<u64>,
    rng: &mut R,
) -> Result<Duration, OramError> {
    let searched_values = distribution
        .sample_iter(&mut *rng)
        .take(N_SEARCHES)
        .collect::<Vec<_>>();

    let start = Instant::now();

    for search_val in searched_values {
        let mut l = 0;
        let mut r = ARRAY_SIZE - 1;
        let mut _found = None;

        // Run for a fixed number of iterations to prevent leakage from the number of iterations and keep execution times consistent
        for _ in 0..ARRAY_SIZE.next_power_of_two().ilog2() {
            let mid = (l + r) / 2;
            let val = oram_array.read(mid, rng)?;

            match val.cmp(&search_val) {
                Ordering::Less => l = min(mid + 1, r),
                Ordering::Greater => r = max(mid - 1, l),
                Ordering::Equal => _found = Some(mid),
            }
        }
    }

    Ok(start.elapsed())
}

fn main() {
    SimpleLogger::init(LevelFilter::Trace, simplelog::Config::default()).unwrap();

    let mut rng = rng();

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

    println!("Averaging over {N_BENCHMARK_REPETITIONS} repetitions");

    let results = iter::repeat_with(|| {
        let distribution = Uniform::try_from(0..ARRAY_SIZE).unwrap();

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

        let access_on_duration =
            benchmark_searches(&mut oram_array, distribution, &mut rng).unwrap();

        oram_array.turn_off().unwrap();

        let access_off_duration =
            benchmark_searches(&mut oram_array, distribution, &mut rng).unwrap();

        let turn_on_start = Instant::now();

        oram_array.turn_on(&mut rng).unwrap();

        let turn_on_duration = turn_on_start.elapsed();

        BenchmarkResult {
            access_on_duration,
            access_off_duration,
            turn_on_duration,
        }
    })
    .take(N_BENCHMARK_REPETITIONS.try_into().unwrap())
    .collect::<Vec<_>>();

    let BenchmarkStats {
        access_on_duration,
        off_total_duration,
        access_off_duration,
        turn_on_duration,
    } = benchmark_stats(&results);

    println!(
        "ON mode: searched for {N_SEARCHES} values in {:?} +- {:?}",
        access_on_duration.mean, access_on_duration.stddev
    );
    println!(
        "OFF mode: searched for {N_SEARCHES} values in {:?} +- {:?} (online accesses took {:?} +- {:?}, \
        turning on took {:?} +- {:?})",
        off_total_duration.mean,
        off_total_duration.stddev,
        access_off_duration.mean,
        access_off_duration.stddev,
        turn_on_duration.mean,
        turn_on_duration.stddev
    );
}
