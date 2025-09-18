use std::time::Instant;

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

fn benchmark_percentages<O: Oram<V = u64>, R: rand::RngCore + rand::CryptoRng>(
    oram_array: &mut O,
    rng: &mut R,
) -> Result<(), oram::OramError> {
    for average_n_accesses_per_addr in AVERAGE_N_ACCESSES_PER_ADDR {
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

        println!(
            "ON mode: accessed {N_UNIQUE_ADDRESSES} addresses with {average_n_accesses_per_addr} accesses per addr on average in {access_on_duration:?}"
        );

        oram_array.turn_off()?;

        let access_off_start = Instant::now();

        for address in &addresses {
            let _ = oram_array.read(*address, rng)?;
        }

        let access_off_duration = access_off_start.elapsed();

        let turn_on_start = Instant::now();

        oram_array.turn_on(rng)?;

        let turn_on_duration = turn_on_start.elapsed();

        println!(
            "OFF mode: accessed {N_UNIQUE_ADDRESSES} addresses with {average_n_accesses_per_addr} accesses per addr \
            on average in {:?} (online accesses took {access_off_duration:?}, turning on took {turn_on_duration:?})",
            access_off_duration + turn_on_duration
        );
    }

    Ok(())
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

    benchmark_percentages(&mut oram_array, &mut rng).unwrap();
}
