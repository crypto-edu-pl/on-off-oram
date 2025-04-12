use std::{char::MAX, time::Instant};

use oram::Oram;
use oram::{hashset::OramHashSet, OramBlock};
use rand::{distributions::Standard, rngs::OsRng, CryptoRng, Rng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

const N_DICT_WORDS: u64 = 82765;
const MAX_WORD_SIZE: usize = 28;
const HASHSET_CAPACITY: u64 = (N_DICT_WORDS * 4).next_power_of_two();

#[derive(Default, Copy, Clone, PartialEq, Debug, Hash)]
struct DictEntry([u8; MAX_WORD_SIZE]);

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

fn main() {
    let mut rng = OsRng;

    let start = Instant::now();

    let mut dictionary = OramHashSet::<DictEntry>::new(HASHSET_CAPACITY, &mut rng).unwrap();

    let duration = start.elapsed();

    println!("Initialized ORAM in {:?}", duration);

    // Prepare the dictionary content

    let wordlist = std::fs::read("data/frequency_dictionary_en_82_765.txt").unwrap();

    let start = Instant::now();

    for word in wordlist.split(|c| *c == b'\n') {
        let mut entry = DictEntry::default();
        entry.0[..word.len()].copy_from_slice(word);
        dictionary.insert(entry, &mut rng).unwrap();
        println!("{word:?}");
    }

    let duration = start.elapsed();

    println!("Prepared dictionary in {:?}", duration);

    // println!("ORAM on:");

    // benchmark_lookups(&mut oram_hash_set, &mut rng);

    // println!("ORAM off:");

    // oram_hash_set.array.turn_off().unwrap();

    // benchmark_lookups(&mut oram_hash_set, &mut rng);

    // let start = Instant::now();

    // oram_hash_set.array.turn_on(&mut rng).unwrap();

    // let duration = start.elapsed();

    // println!("Turned on in {:?}", duration);
}
