use std::{
    iter,
    ops::{Add, Div},
    time::{Duration, Instant},
};

use log::LevelFilter;
use rand::{distributions::Uniform, prelude::Distribution, rngs::OsRng, seq::SliceRandom};
use simplelog::SimpleLogger;
use static_assertions::const_assert;

use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Oram};

#[cfg(not(feature = "bypass_oram"))]
use oram::DefaultOram;

#[cfg(feature = "bypass_oram")]
use oram::not_really_oram::NotReallyOram;

const ARRAY_SIZE: u64 = 1 << 17;

const_assert!(ARRAY_SIZE >= LINEAR_TIME_ORAM_CUTOFF);

const N_UNIQUE_ADDRESSES: u64 = 100;

const AVERAGE_N_ACCESSES_PER_ADDR: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const N_BENCHMARK_REPETITIONS: u32 = 20;

struct BenchmarkResult {
    access_on_duration: Duration,
    access_off_duration: Duration,
    turn_on_duration: Duration,
}

impl Add for BenchmarkResult {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BenchmarkResult {
            access_on_duration: self.access_on_duration + other.access_on_duration,
            access_off_duration: self.access_off_duration + other.access_off_duration,
            turn_on_duration: self.turn_on_duration + other.turn_on_duration,
        }
    }
}

impl Div<u32> for BenchmarkResult {
    type Output = Self;

    fn div(self, rhs: u32) -> Self {
        BenchmarkResult {
            access_on_duration: self.access_on_duration / rhs,
            access_off_duration: self.access_off_duration / rhs,
            turn_on_duration: self.turn_on_duration / rhs,
        }
    }
}

struct BenchmarkStats {
    access_on_duration: Stats,
    off_total_duration: Stats,
    access_off_duration: Stats,
    turn_on_duration: Stats,
}

struct Stats {
    mean: Duration,
    stddev: Duration,
}

fn benchmark_percentages<O: Oram<V = u64>, R: rand::RngCore + rand::CryptoRng>(
    oram_array: &mut O,
    rng: &mut R,
    average_n_accesses_per_addr: u64,
) -> Result<BenchmarkResult, oram::OramError> {
    let mut addresses = (0..N_UNIQUE_ADDRESSES)
        .chain(
            Uniform::from(0..N_UNIQUE_ADDRESSES)
                .sample_iter(&mut *rng)
                .take(
                    ((average_n_accesses_per_addr - 1) * N_UNIQUE_ADDRESSES)
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

fn mean_and_standard_deviation(data: &[Duration]) -> Stats {
    let mean =
        data.iter().copied().reduce(|acc, x| acc + x).unwrap() / u32::try_from(data.len()).unwrap();
    let variance = data
        .iter()
        .map(|x| {
            let diff = (*x - mean).as_nanos();
            diff * diff
        })
        .reduce(|acc, x| acc + x)
        .unwrap()
        / u128::try_from(data.len() - 1).unwrap();
    let stddev = Duration::from_nanos(variance.isqrt().try_into().unwrap());
    Stats { mean, stddev }
}

fn benchmark_stats(results: &[BenchmarkResult]) -> BenchmarkStats {
    let access_on_durations = results
        .iter()
        .map(|x| x.access_on_duration)
        .collect::<Vec<_>>();
    let off_total_durations = results
        .iter()
        .map(|result| result.access_off_duration + result.turn_on_duration)
        .collect::<Vec<_>>();
    let access_off_durations = results
        .iter()
        .map(|x| x.access_off_duration)
        .collect::<Vec<_>>();
    let turn_on_durations = results
        .iter()
        .map(|x| x.turn_on_duration)
        .collect::<Vec<_>>();
    BenchmarkStats {
        access_on_duration: mean_and_standard_deviation(&access_on_durations),
        off_total_duration: mean_and_standard_deviation(&off_total_durations),
        access_off_duration: mean_and_standard_deviation(&access_off_durations),
        turn_on_duration: mean_and_standard_deviation(&turn_on_durations),
    }
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

    println!("Averaging over {N_BENCHMARK_REPETITIONS} repetitions");

    for average_n_accesses_per_addr in AVERAGE_N_ACCESSES_PER_ADDR {
        let results = iter::repeat_with(|| {
            benchmark_percentages(&mut oram_array, &mut rng, average_n_accesses_per_addr).unwrap()
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
            "ON mode: accessed {N_UNIQUE_ADDRESSES} addresses with {average_n_accesses_per_addr} accesses per addr on average in {:?} +- {:?}",
            access_on_duration.mean, access_on_duration.stddev
        );
        println!(
            "OFF mode: accessed {N_UNIQUE_ADDRESSES} addresses with {average_n_accesses_per_addr} accesses per addr \
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
