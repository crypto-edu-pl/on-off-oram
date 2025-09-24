use std::collections::HashSet;
use std::iter;
use std::time::{Duration, Instant};

use log::LevelFilter;
use oram::bin_utils::{benchmark_stats, BenchmarkResult, BenchmarkStats};
use oram::{hashset::OramHashSet, OramBlock};
use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Oram, OramError};
use rand::{rng, CryptoRng, RngCore};
use simplelog::SimpleLogger;
use static_assertions::const_assert;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

const MAX_WORD_SIZE: usize = 28;
const HASHSET_CAPACITY: u64 = 1 << 20;

const_assert!(HASHSET_CAPACITY >= LINEAR_TIME_ORAM_CUTOFF);

const N_BENCHMARK_REPETITIONS: u32 = 20;

#[derive(Default, Copy, Clone, PartialEq, Eq, Debug, Hash)]
struct DictEntry([u8; MAX_WORD_SIZE]);

impl From<String> for DictEntry {
    fn from(value: String) -> Self {
        let mut result = Self::default();
        result.0[..value.len()].copy_from_slice(value.as_bytes());
        result
    }
}

impl OramBlock for DictEntry {}

impl ConditionallySelectable for DictEntry {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut result = DictEntry::default();
        for i in 0..MAX_WORD_SIZE {
            result.0[i] = u8::conditional_select(&a.0[i], &b.0[i], choice);
        }
        result
    }
}

impl ConstantTimeEq for DictEntry {
    fn ct_eq(&self, other: &Self) -> Choice {
        let mut result = 1.into();
        for i in 0..MAX_WORD_SIZE {
            result &= self.0[i].ct_eq(&other.0[i]);
        }
        result
    }
}

fn get_words(path: &str) -> Vec<String> {
    let content = String::from_utf8(std::fs::read(path).unwrap()).unwrap();
    content
        .replace(|c: char| c.is_ascii_punctuation() && c != '\'', "")
        .to_ascii_lowercase()
        .split_ascii_whitespace()
        .map(Into::into)
        .collect()
}

fn spellcheck<R: RngCore + CryptoRng>(
    text: &[DictEntry],
    dictionary: &mut OramHashSet<DictEntry>,
    rng: &mut R,
) -> Result<Duration, OramError> {
    let start = Instant::now();

    for word in text {
        dictionary.contains(*word, rng)?;
    }

    Ok(start.elapsed())
}

fn main() {
    SimpleLogger::init(LevelFilter::Trace, simplelog::Config::default()).unwrap();

    let body_parts = get_words("data/body_parts.txt")
        .into_iter()
        .map(DictEntry::from)
        .collect::<Vec<_>>();

    let mut dictionary_entries = HashSet::new();
    dictionary_entries.extend(&body_parts);

    let mut rng = rng();

    let start = Instant::now();

    let mut dictionary = OramHashSet::<DictEntry>::new(HASHSET_CAPACITY, &mut rng).unwrap();

    let duration = start.elapsed();

    println!("Initialized ORAM in {:?}", duration);

    // Prepare the dictionary content

    let start = Instant::now();

    for entry in dictionary_entries {
        dictionary.insert(entry, &mut rng).unwrap();
    }

    let duration = start.elapsed();

    println!("Prepared dictionary in {:?}", duration);

    println!("Averaging over {N_BENCHMARK_REPETITIONS} repetitions");

    let results = iter::repeat_with(|| {
        let access_on_duration = spellcheck(&body_parts, &mut dictionary, &mut rng).unwrap();

        dictionary.array.turn_off().unwrap();

        let access_off_duration = spellcheck(&body_parts, &mut dictionary, &mut rng).unwrap();

        let turn_on_start = Instant::now();

        dictionary.array.turn_on(&mut rng).unwrap();

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
        "ON mode: checked in {:?} +- {:?}",
        access_on_duration.mean, access_on_duration.stddev
    );
    println!(
        "OFF mode: checked in {:?} +- {:?} (online accesses took {:?} +- {:?}, turning on took {:?} +- {:?})",
        off_total_duration.mean,
        off_total_duration.stddev,
        access_off_duration.mean,
        access_off_duration.stddev,
        turn_on_duration.mean,
        turn_on_duration.stddev
    );
}
