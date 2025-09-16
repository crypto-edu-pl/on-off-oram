use std::collections::HashSet;
use std::time::Instant;

use log::LevelFilter;
use oram::{hashset::OramHashSet, OramBlock};
use oram::{path_oram::LINEAR_TIME_ORAM_CUTOFF, Oram, OramError};
use rand::{rngs::OsRng, CryptoRng, RngCore};
use simplelog::SimpleLogger;
use static_assertions::const_assert;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

const MAX_WORD_SIZE: usize = 28;
const HASHSET_CAPACITY: u64 = 1 << 20;

const_assert!(HASHSET_CAPACITY >= LINEAR_TIME_ORAM_CUTOFF);

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
) -> Result<Vec<bool>, OramError> {
    let mut result = Vec::with_capacity(text.len());

    let start = Instant::now();

    for word in text {
        result.push(dictionary.contains(*word, rng)?);
    }

    let duration = start.elapsed();

    println!("Checked in {:?}", duration);

    Ok(result)
}

fn main() {
    SimpleLogger::init(LevelFilter::Trace, simplelog::Config::default()).unwrap();

    let body_parts = get_words("data/body_parts.txt")
        .into_iter()
        .map(DictEntry::from)
        .collect::<Vec<_>>();

    let mut dictionary_entries = HashSet::new();
    dictionary_entries.extend(&body_parts);

    let mut rng = OsRng;

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

    println!("ORAM on:");

    spellcheck(&body_parts, &mut dictionary, &mut rng).unwrap();

    println!("ORAM off:");

    dictionary.array.turn_off().unwrap();

    spellcheck(&body_parts, &mut dictionary, &mut rng).unwrap();

    let start = Instant::now();

    dictionary.array.turn_on(&mut rng).unwrap();

    let duration = start.elapsed();

    println!("Turned on in {:?}", duration);
}
