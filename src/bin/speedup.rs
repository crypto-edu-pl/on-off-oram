use std::{iter, time::Instant};

use log::LevelFilter;
use rand::{
    distr::slice::Choose,
    prelude::Distribution,
    random, rng,
    seq::{IteratorRandom, SliceRandom},
};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

use oram::{
    BlockValue, Oram,
    bin_utils::{BenchmarkResult, BenchmarkStats, benchmark_stats},
    path_oram::LINEAR_TIME_ORAM_CUTOFF,
};

#[cfg(not(feature = "bypass_oram"))]
use oram::DefaultOram;

#[cfg(feature = "bypass_oram")]
use oram::not_really_oram::NotReallyOram;

const ARRAY_SIZE: u64 = 1 << 14;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

const N_UNIQUE_ADDRESSES: [u64; 1] = [100];

const AVERAGE_N_ACCESSES_PER_ADDR: [u64; 11] = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

const N_BENCHMARK_REPETITIONS: u32 = 100;

fn gen_addresses<R: rand::RngCore + rand::CryptoRng>(
    rng: &mut R,
    n_unique_addresses: u64,
    average_n_accesses_per_addr: u64,
) -> Vec<u64> {
    let unique_addresses =
        (0..ARRAY_SIZE).choose_multiple(rng, n_unique_addresses.try_into().unwrap());
    let mut addresses = unique_addresses
        .iter()
        .copied()
        .chain(
            Choose::new(&unique_addresses)
                .unwrap()
                .sample_iter(&mut *rng)
                .take(
                    ((average_n_accesses_per_addr - 1) * n_unique_addresses)
                        .try_into()
                        .unwrap(),
                )
                .copied(),
        )
        .collect::<Vec<_>>();
    addresses.shuffle(rng);
    addresses
}

fn benchmark_percentages<O: Oram, R: rand::RngCore + rand::CryptoRng>(
    oram_array: &mut O,
    rng: &mut R,
    n_unique_addresses: u64,
    average_n_accesses_per_addr: u64,
) -> Result<BenchmarkResult, oram::OramError> {
    let on_addresses = gen_addresses(rng, n_unique_addresses, average_n_accesses_per_addr);

    let access_on_start = Instant::now();

    for address in &on_addresses {
        let _ = oram_array.read(*address, rng)?;
    }

    let access_on_duration = access_on_start.elapsed();

    let off_addresses = gen_addresses(rng, n_unique_addresses, average_n_accesses_per_addr);

    oram_array.turn_off()?;

    let access_off_start = Instant::now();

    for address in &off_addresses {
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

    let mut rng = rng();

    let start = Instant::now();

    let mut oram_array = {
        #[cfg(not(feature = "bypass_oram"))]
        {
            DefaultOram::<BlockValue<64>>::new(ARRAY_SIZE, &mut rng).unwrap()
        }

        #[cfg(feature = "bypass_oram")]
        {
            NotReallyOram::<BlockValue<64>>::new(ARRAY_SIZE).unwrap()
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

            // println!(
            //     "ON mode: accessed {n_unique_addresses} addresses with {average_n_accesses_per_addr} accesses per addr on average in {:?} +- {:?}",
            //     access_on_duration.mean, access_on_duration.stddev
            // );
            // println!(
            //     "OFF mode: accessed {n_unique_addresses} addresses with {average_n_accesses_per_addr} accesses per addr \
            //     on average in {:?} +- {:?} (online accesses took {:?} +- {:?}, turning on took {:?} +- {:?})",
            //     off_total_duration.mean,
            //     off_total_duration.stddev,
            //     access_off_duration.mean,
            //     access_off_duration.stddev,
            //     turn_on_duration.mean,
            //     turn_on_duration.stddev
            // );
            for stat in [
                access_on_duration,
                off_total_duration,
                access_off_duration,
                turn_on_duration,
            ] {
                print!("{}; {}; ", stat.mean.as_nanos(), stat.stddev.as_nanos());
            }
            println!();
        }
    }
}
