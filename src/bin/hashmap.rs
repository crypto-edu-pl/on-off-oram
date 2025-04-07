use std::{
    hash::{BuildHasher, Hash, RandomState},
    time::Instant,
};

use oram::{DefaultOram, Oram, OramBlock, OramError};
use rand::{distributions::Standard, rngs::OsRng, CryptoRng, Rng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

const ARRAY_SIZE: u64 = 4096;

trait OramHashSetData: OramBlock + Hash + ConstantTimeEq {}
impl<V: OramBlock + Hash + ConstantTimeEq> OramHashSetData for V {}

struct OramHashSet<V: OramHashSetData> {
    array: DefaultOram<OramHashSetEntry<V>>,
    hash_builder: RandomState,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OramHashSetEntry<V: OramHashSetData> {
    data: V,
    tag: u8,
}

impl<V: OramHashSetData> Default for OramHashSetEntry<V> {
    fn default() -> Self {
        OramHashSetEntry {
            data: V::default(),
            tag: 0xff,
        }
    }
}

impl<V: OramHashSetData> ConditionallySelectable for OramHashSetEntry<V> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let data = V::conditional_select(&a.data, &b.data, choice);
        let tag = u8::conditional_select(&a.tag, &b.tag, choice);
        Self { data, tag }
    }
}

impl<V: OramHashSetData> OramBlock for OramHashSetEntry<V> {}

impl<V: OramHashSetData> OramHashSet<V> {
    const GROUP_WIDTH: u64 = 2;
    const N_CHECKED_GROUPS: u64 = 3;

    fn new<R: Rng + CryptoRng>(capacity: u64, rng: &mut R) -> Result<Self, OramError> {
        Ok(OramHashSet {
            array: DefaultOram::<OramHashSetEntry<V>>::new(capacity, rng)?,
            hash_builder: RandomState::new(),
        })
    }

    fn insert<R: Rng + CryptoRng>(&mut self, value: V, rng: &mut R) -> Result<(), OramError> {
        let hash = self.hash_builder.hash_one(value);
        let capacity = self.array.block_capacity()?;
        let mut slot_index = hash % capacity;

        let mut insert_idx = u64::MAX;

        let mut stride = 0;

        let tag = u8::try_from(hash >> (u64::BITS - 7))?;

        // TODO this could be read from ORAM using batching instead
        for _ in 0..Self::N_CHECKED_GROUPS {
            let mut entries = Vec::with_capacity(Self::GROUP_WIDTH.try_into()?);
            for i in 0..Self::GROUP_WIDTH {
                let entry = self.array.read((slot_index + i) % capacity, rng)?;
                entries.push(entry);
            }

            for (i, entry) in entries.iter().enumerate() {
                let insert_idx_not_yet_found = insert_idx.ct_eq(&u64::MAX);
                let tag_matches = entry.tag.ct_eq(&tag);
                let equal_to_inserted = entry.data.ct_eq(&value);
                insert_idx.conditional_assign(
                    &(slot_index + u64::try_from(i)? % capacity),
                    insert_idx_not_yet_found & tag_matches & equal_to_inserted,
                );
            }

            for (i, entry) in entries.iter().enumerate() {
                let insert_idx_not_yet_found = insert_idx.ct_eq(&u64::MAX);
                let tag_is_empty = entry.tag.ct_eq(&0xff);
                insert_idx.conditional_assign(
                    &(slot_index + u64::try_from(i)? % capacity),
                    insert_idx_not_yet_found & tag_is_empty,
                );
            }

            stride += Self::GROUP_WIDTH;
            slot_index = (slot_index + stride) % capacity;
        }

        assert_ne!(insert_idx, u64::MAX);

        self.array
            .write(insert_idx, OramHashSetEntry { data: value, tag }, rng)?;

        Ok(())
    }

    fn contains<R: Rng + CryptoRng>(&mut self, value: V, rng: &mut R) -> Result<bool, OramError> {
        let hash = self.hash_builder.hash_one(value);
        let capacity = self.array.block_capacity()?;
        let mut slot_index = hash % capacity;

        let mut found = Choice::from(0);

        let mut stride = 0;

        let tag = u8::try_from(hash >> (u64::BITS - 7))?;

        for _ in 0..Self::N_CHECKED_GROUPS {
            let mut entries = Vec::with_capacity(Self::GROUP_WIDTH.try_into()?);
            for i in 0..Self::GROUP_WIDTH {
                let entry = self.array.read((slot_index + i) % capacity, rng)?;
                entries.push(entry);
            }

            for entry in entries.iter() {
                let tag_matches = entry.tag.ct_eq(&tag);
                let equal_to_searched = entry.data.ct_eq(&value);
                found |= tag_matches & equal_to_searched;
            }

            stride += Self::GROUP_WIDTH;
            slot_index = (slot_index + stride) % capacity;
        }

        Ok(found.into())
    }
}

fn benchmark_lookups<R: RngCore + CryptoRng>(oram_hash_set: &mut OramHashSet<u64>, rng: &mut R) {
    for _ in 0..20 {
        let search_val = rng.gen::<u64>();

        let start = Instant::now();

        let found = oram_hash_set.contains(search_val, rng).unwrap();

        let duration = start.elapsed();

        println!("Got {:?} in {:?}", found, duration);
    }
}

fn main() {
    let mut rng = OsRng;

    let mut oram_hash_set = OramHashSet::<u64>::new(ARRAY_SIZE, &mut rng).unwrap();

    let values = (&mut rng)
        .sample_iter(Standard)
        .take((ARRAY_SIZE / 4) as usize)
        .collect::<Vec<u64>>();

    let start = Instant::now();

    for value in &values {
        oram_hash_set.insert(*value, &mut rng).unwrap();
    }

    let duration = start.elapsed();

    println!("Initialized ORAM hashset in {:?}", duration);

    println!("ORAM on:");

    benchmark_lookups(&mut oram_hash_set, &mut rng);

    println!("ORAM off:");

    oram_hash_set.array.turn_off().unwrap();

    benchmark_lookups(&mut oram_hash_set, &mut rng);

    let start = Instant::now();

    oram_hash_set.array.turn_on(&mut rng).unwrap();

    let duration = start.elapsed();

    println!("Turned on in {:?}", duration);
}
