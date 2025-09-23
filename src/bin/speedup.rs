use std::{iter, time::Instant};

use log::LevelFilter;
use rand::{distributions::Uniform, prelude::Distribution, random, rngs::OsRng, seq::SliceRandom};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

use oram::{
    bin_utils::{benchmark_stats, BenchmarkResult, BenchmarkStats},
    path_oram::LINEAR_TIME_ORAM_CUTOFF,
    Oram,
};

#[cfg(not(feature = "bypass_oram"))]
use oram::DefaultOram;

#[cfg(feature = "bypass_oram")]
use oram::not_really_oram::NotReallyOram;

const ARRAY_SIZE: u64 = 1 << 17;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

const N_UNIQUE_ADDRESSES: [u64; 1] = [100];

const AVERAGE_N_ACCESSES_PER_ADDR: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const N_BENCHMARK_REPETITIONS: u32 = 20;

fn benchmark_percentages<O: Oram<V = u64>, R: rand::RngCore + rand::CryptoRng>(
    oram_array: &mut O,
    rng: &mut R,
    n_unique_addresses: u64,
    average_n_accesses_per_addr: u64,
) -> Result<BenchmarkResult, oram::OramError> {
    let mut addresses = (0..n_unique_addresses)
        .chain(
            Uniform::from(0..n_unique_addresses)
                .sample_iter(&mut *rng)
                .take(
                    ((average_n_accesses_per_addr - 1) * n_unique_addresses)
                        .try_into()
                        .unwrap(),
                ),
        )
        .collect::<Vec<_>>();
    addresses.shuffle(rng);

    let access_on_start = Instant::now();

    for address in &addresses {
        let _ = oram_array.read(*address, rng)?;
    }

    let access_on_duration = access_on_start.elapsed();

    oram_array.turn_off()?;

    let access_off_start = Instant::now();

    for address in &addresses {
        let _ = oram_array.read(*address, rng)?;
    }

    let access_off_duration = access_off_start.elapsed();

    let turn_on_start = Instant::now();

    oram_array.turn_on(rng)?;

    let turn_on_duration = turn_on_start.elapsed();

    Ok(BenchmarkResult {
        access_on_duration,
        access_off_duration,
        turn_on_duration,
    })
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

    let start = Instant::now();

    for i in 0..ARRAY_SIZE {
        oram_array.write(i, random(), &mut rng).unwrap();
    }

    let duration = start.elapsed();

    println!("Prepared array in {:?}", duration);

    println!("Averaging over {N_BENCHMARK_REPETITIONS} repetitions");

    for n_unique_addresses in N_UNIQUE_ADDRESSES {
        for average_n_accesses_per_addr in AVERAGE_N_ACCESSES_PER_ADDR {
            let results = iter::repeat_with(|| {
                benchmark_percentages(
                    &mut oram_array,
                    &mut rng,
                    n_unique_addresses,
                    average_n_accesses_per_addr,
                )
                .unwrap()
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
                "ON mode: accessed {n_unique_addresses} addresses with {average_n_accesses_per_addr} accesses per addr on average in {:?} +- {:?}",
                access_on_duration.mean, access_on_duration.stddev
            );
            println!(
                "OFF mode: accessed {n_unique_addresses} addresses with {average_n_accesses_per_addr} accesses per addr \
                on average in {:?} +- {:?} (online accesses took {:?} +- {:?}, turning on took {:?} +- {:?})",
                off_total_duration.mean,
                off_total_duration.stddev,
                access_off_duration.mean,
                access_off_duration.stddev,
                turn_on_duration.mean,
                turn_on_duration.stddev
            );
        }
    }
}
