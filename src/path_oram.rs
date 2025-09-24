// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! An implementation of Path ORAM.

#[cfg(not(feature = "full_reconstruction"))]
use std::mem;
use std::{
    cmp::{self, min},
    collections::HashMap,
};

#[cfg(any(
    not(feature = "do_not_cache_block_values"),
    feature = "direct_accesses_in_off_mode"
))]
use std::collections::hash_map;

#[cfg(feature = "exact_locations_in_position_map")]
use std::slice::SliceIndex;

use rand::{CryptoRng, Rng, RngCore};

#[cfg(feature = "batched_turning_on")]
use static_assertions::const_assert;
#[cfg(not(feature = "full_reconstruction"))]
use subtle::ConstantTimeEq;
#[cfg(feature = "exact_locations_in_position_map")]
use subtle::{ConditionallySelectable, ConstantTimeLess};

use super::{position_map::PositionMap, stash::ObliviousStash};
use crate::{
    Address, BlockSize, BucketSize, Oram, OramBlock, OramError, OramMode, RecursionCutoff,
    StashSize,
    bucket::{BlockMetadata, Bucket, PositionBlock},
    linear_time_oram::LinearTimeOram,
    utils::{CompleteBinaryTreeIndex, TreeHeight},
};

#[cfg(feature = "direct_accesses_in_off_mode")]
use crate::bucket::PathOramBlock;

#[cfg(feature = "exact_locations_in_position_map")]
use crate::{stash::StashEntry, utils::TreeIndex};

/// The default cutoff size in blocks
/// below which `PathOram` uses a linear position map instead of a recursive one.
pub const DEFAULT_RECURSION_CUTOFF: RecursionCutoff = 1 << 6;

/// The parameter "Z" from the Path ORAM literature that sets the number of blocks per bucket; typical values are 3 or 4.
/// Here we adopt the more conservative setting of 4.
pub const DEFAULT_BLOCKS_PER_BUCKET: BucketSize = 4;

/// The default number of positions stored per position block.
pub const DEFAULT_POSITIONS_PER_BLOCK: BlockSize = 8;

/// The default number of overflow blocks that the Path ORAM stash (and recursive stashes) can store.
pub const DEFAULT_STASH_OVERFLOW_SIZE: StashSize = 40;

/// The cutoff size in blocks below which `DefaultOram` is simply a linear time ORAM.
pub const LINEAR_TIME_ORAM_CUTOFF: RecursionCutoff = 1 << 6;

/// A doubly oblivious Path ORAM.
///
/// ## Parameters
///
/// - Block type `V`: the type of elements stored by the ORAM.
/// - Bucket size `Z`: the number of blocks per Path ORAM bucket.
///     Must be at least 2. Typical values are 3, 4, or 5.
///     Along with the overflow size, this value affects the probability
///     of stash overflow (see below) and should be set with care.
/// - Positions per block `AB`:
///     The number of positions stored in each block of the recursive position map ORAM.
///     Must be a power of two and must be at least 2 (otherwise the recursion will not terminate).
///     Otherwise, can be freely tuned for performance.
///     Larger `AB` means fewer levels of recursion but higher costs for accessing each level.
/// - Recursion threshold: the maximum number of position blocks that will be stored in a recursive Path ORAM.
///     Below this value, the position map will be a linear scanning ORAM.
///     Can be freely tuned for performance.
///     A larger values means fewer levels of recursion, but a more expensive base position map.
/// - Overflow size: The number of blocks that the stash can store between ORAM accesses without overflowing.
///     Along with the bucket size, this value affects the probability of stash overflow (see below)
///     and should be set with care.
///
/// ## Security
///
/// ORAM operations are guaranteed to be oblivious, *unless* the stash overflows.
/// In this case, the stash will grow, which reveals that the overflow occurred.
/// This is a violation of obliviousness, but a mild one in several ways.
/// The stash overflow is very likely to reset to empty after the overflow,
/// and stash overflows are isolated events. It is not at all obvious
/// how an attacker might use a stash overflow to infer properties of the access pattern.
///
/// That said, it is best to choose parameters so that the stash does not ever overflow.
/// With Z = 4, experiments from the [original Path ORAM paper](https://eprint.iacr.org/2013/280.pdf)
/// indicate that the probability of overflow is independent of the number N of blocks stored,
/// and that setting SO = 40 is enough to reduce this probability to below 2^{-50} (Figure 3).
/// The authors conservatively estimate that setting SO = 89 suffices for 2^{-80} overflow probability.
/// The choice Z = 3 is also popular, although the probability of overflow is less well understood.
#[derive(Debug)]
pub struct PathOram<V: OramBlock, const Z: BucketSize, const AB: BlockSize> {
    /// The underlying untrusted memory that the ORAM is obliviously accessing on behalf of its client.
    physical_memory: Vec<Bucket<V, Z>>,
    /// The Path ORAM stash.
    stash: ObliviousStash<V>,
    /// The Path ORAM position map.
    position_map: PositionMap<AB, Z>,
    /// The height of the Path ORAM tree data structure.
    height: TreeHeight,
    /// Current mode.
    mode: OramMode,
    /// The set of blocks accessed in off mode. Their paths will be evicted when ORAM is turned on.
    ///
    /// This is not an oblivious data structure, but we use it only in off mode so it's fine.
    #[cfg(feature = "direct_accesses_in_off_mode")]
    blocks_accessed_in_off_mode: HashMap<Address, BlockLocation>,
    #[cfg(all(
        not(feature = "direct_accesses_in_off_mode"),
        not(feature = "do_not_cache_block_values"),
    ))]
    blocks_accessed_in_off_mode: HashMap<Address, V>,
    #[cfg(all(
        not(feature = "direct_accesses_in_off_mode"),
        feature = "do_not_cache_block_values",
    ))]
    blocks_accessed_in_off_mode: HashMap<Address, ()>,
}

#[cfg(feature = "direct_accesses_in_off_mode")]
#[derive(Debug, Copy, Clone)]
enum BlockLocation {
    OramTree { bucket: usize, offset: usize },
    Stash { offset: usize },
    Dummy,
}

/// An `Oram` suitable for most use cases, with reasonable default choices of parameters.
#[derive(Debug)]
pub struct DefaultOram<V: OramBlock>(DefaultOramBackend<V>);

#[derive(Debug)]
enum DefaultOramBackend<V: OramBlock> {
    Path(PathOram<V, DEFAULT_BLOCKS_PER_BUCKET, DEFAULT_POSITIONS_PER_BLOCK>),
    Linear(LinearTimeOram<V>),
}

impl<V: OramBlock> Oram for DefaultOram<V> {
    type V = V;

    fn block_capacity(&self) -> Result<Address, OramError> {
        match &self.0 {
            DefaultOramBackend::Path(p) => p.block_capacity(),
            DefaultOramBackend::Linear(l) => l.block_capacity(),
        }
    }

    fn access<R: rand::RngCore + CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        index: Address,
        callback: F,
        rng: &mut R,
    ) -> Result<Self::V, OramError> {
        match &mut self.0 {
            DefaultOramBackend::Path(p) => p.access(index, callback, rng),
            DefaultOramBackend::Linear(l) => l.access(index, callback, rng),
        }
    }

    fn batch_access<R: RngCore + CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        callbacks: &[(Address, F)],
        rng: &mut R,
    ) -> Result<Vec<Self::V>, OramError> {
        match &mut self.0 {
            DefaultOramBackend::Path(p) => p.batch_access(callbacks, rng),
            DefaultOramBackend::Linear(l) => l.batch_access(callbacks, rng),
        }
    }

    fn turn_on<R: RngCore + CryptoRng>(&mut self, rng: &mut R) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Path(p) => p.turn_on(rng),
            DefaultOramBackend::Linear(l) => l.turn_on(rng),
        }
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Path(p) => p.turn_off(),
            DefaultOramBackend::Linear(l) => l.turn_off(),
        }
    }

    fn turn_on_without_evicting(&mut self) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Path(p) => p.turn_on_without_evicting(),
            DefaultOramBackend::Linear(l) => l.turn_on_without_evicting(),
        }
    }

    fn mode(&self) -> OramMode {
        match &self.0 {
            DefaultOramBackend::Path(p) => p.mode(),
            DefaultOramBackend::Linear(l) => l.mode(),
        }
    }
}

impl<V: OramBlock> DefaultOram<V> {
    /// Returns a new ORAM mapping addresses `0 <= address < block_capacity` to default `V` values.
    ///
    /// # Errors
    ///
    /// If `block_capacity` is not a power of two, returns an `InvalidConfigurationError`.
    pub fn new<R: Rng + CryptoRng>(
        block_capacity: Address,
        rng: &mut R,
    ) -> Result<Self, OramError> {
        if block_capacity < LINEAR_TIME_ORAM_CUTOFF {
            Ok(Self(DefaultOramBackend::Linear(LinearTimeOram::new(
                block_capacity,
            )?)))
        } else {
            let max_batch_size = {
                #[cfg(feature = "batched_turning_on")]
                {
                    u64::from(block_capacity.ilog2()) * u64::try_from(DEFAULT_BLOCKS_PER_BUCKET)?
                }

                #[cfg(not(feature = "batched_turning_on"))]
                {
                    1
                }
            };
            Ok(Self(DefaultOramBackend::Path(PathOram::<
                V,
                DEFAULT_BLOCKS_PER_BUCKET,
                DEFAULT_POSITIONS_PER_BLOCK,
            >::new_with_parameters(
                block_capacity,
                rng,
                DEFAULT_STASH_OVERFLOW_SIZE,
                DEFAULT_RECURSION_CUTOFF,
                max_batch_size,
            )?)))
        }
    }
}

impl<V: OramBlock, const Z: BucketSize, const AB: BlockSize> PathOram<V, Z, AB> {
    /// Returns a new `PathOram` mapping addresses `0 <= address < block_capacity` to default `V` values,
    /// with a stash overflow size of `overflow_size` blocks, and a recursion cutoff of `recursion_cutoff`.
    /// (See [`PathOram`]) for a description of these parameters).
    ///
    /// # Errors
    ///
    /// Returns an `InvalidConfigurationError` in the following cases.
    ///
    /// - `block_capacity` is 0, 1, or is not a power of two.
    /// - `AB` is 0, 1, or is not a power of two.
    /// - `Z` is 0 or 1.
    /// - `recursion_cutoff` is 0.
    /// - `overflow_size` is 0.
    pub fn new_with_parameters<R: Rng + CryptoRng>(
        block_capacity: Address,
        rng: &mut R,
        overflow_size: StashSize,
        recursion_cutoff: RecursionCutoff,
        max_batch_size: u64,
    ) -> Result<Self, OramError> {
        log::info!("PathOram::new(capacity = {})", block_capacity,);

        if !block_capacity.is_power_of_two() | (block_capacity <= 1) {
            return Err(OramError::InvalidConfigurationError {
                parameter_name: "ORAM capacity".to_string(),
                parameter_value: block_capacity.to_string(),
            });
        }

        if Z <= 1 {
            return Err(OramError::InvalidConfigurationError {
                parameter_name: "Bucket size Z".to_string(),
                parameter_value: Z.to_string(),
            });
        }

        if recursion_cutoff == 0 {
            return Err(OramError::InvalidConfigurationError {
                parameter_name: "Recursion cutoff".to_string(),
                parameter_value: recursion_cutoff.to_string(),
            });
        }

        if overflow_size == 0 {
            return Err(OramError::InvalidConfigurationError {
                parameter_name: "Overflow size".to_string(),
                parameter_value: overflow_size.to_string(),
            });
        }

        let number_of_nodes = block_capacity;

        let height: u64 = (block_capacity.ilog2() - 1).into();

        let path_size = u64::try_from(Z)? * (height + 1);
        let stash = ObliviousStash::new(path_size, overflow_size, max_batch_size, block_capacity)?;

        // physical_memory holds `block_capacity` buckets, each storing up to Z blocks.
        // The number of leaves is `block_capacity` / 2, which the original Path ORAM paper's experiments
        // found was sufficient to keep the stash size small with high probability.
        let mut physical_memory = Vec::new();
        physical_memory.resize(usize::try_from(number_of_nodes)?, Bucket::<V, Z>::default());

        // Initialize a new position map,
        // and initialize its entries to random leaf indices.
        let mut position_map =
            PositionMap::new(block_capacity, rng, overflow_size, recursion_cutoff)?;

        let first_leaf_index: u64 = 2u64.pow(height.try_into()?);
        let last_leaf_index = (2 * first_leaf_index) - 1;
        let ab_address: Address = AB.try_into()?;

        let num_address_blocks = if block_capacity % ab_address == 0 {
            block_capacity / ab_address
        } else {
            block_capacity / ab_address + 1
        };
        for block_index in 0..num_address_blocks {
            let mut data = [BlockMetadata::default(); AB];
            for metadata in &mut data {
                metadata.assigned_leaf = rng.random_range(first_leaf_index..=last_leaf_index);
                #[cfg(feature = "exact_locations_in_position_map")]
                {
                    metadata.exact_bucket = BlockMetadata::NOT_IN_TREE;
                    metadata.exact_offset = BlockMetadata::UNINITIALIZED;
                }
            }
            let position_block = PositionBlock { data };
            position_map.write_position_block(block_index * ab_address, position_block, rng)?;
        }

        Ok(Self {
            physical_memory,
            stash,
            position_map,
            height,
            mode: OramMode::On,
            blocks_accessed_in_off_mode: HashMap::new(),
        })
    }

    #[cfg(feature = "exact_locations_in_position_map")]
    fn batch_update_position_map<
        R: Rng + CryptoRng,
        RangeT: SliceIndex<[StashEntry<V>], Output = [StashEntry<V>]>,
    >(
        &mut self,
        range: RangeT,
        rng: &mut R,
    ) -> Result<(), OramError> {
        let stash_entries = &self.stash.entries[range];

        // Create a batch of position map updates that contains exactly stash_entries.len()
        // unique updates (each for a different address).

        let mut updates = Vec::with_capacity(stash_entries.len());
        let base_dummy_address = self.block_capacity()?;

        // Put the updates resulting from the stash entries in the update array.
        // Change the address of dummy updates to a unique out-of-bounds address.
        for (i, entry) in stash_entries.iter().enumerate() {
            let is_in_tree = entry.exact_bucket.ct_ne(&BlockMetadata::NOT_IN_TREE);
            let exact_offset =
                u64::conditional_select(&i.try_into()?, &entry.exact_offset, is_in_tree);

            let metadata = BlockMetadata {
                assigned_leaf: entry.block.position,
                exact_bucket: entry.exact_bucket,
                exact_offset,
            };

            let is_dummy = entry.block.ct_is_dummy();
            let address = Address::conditional_select(
                &entry.block.address,
                &(base_dummy_address + Address::try_from(i)?),
                is_dummy,
            );

            updates.push((address, metadata));
        }

        self.position_map.batch_write(&updates, rng)?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn stash_occupancy(&self) -> StashSize {
        self.stash.occupancy()
    }
}

impl<V: OramBlock, const Z: BucketSize, const AB: BlockSize> Oram for PathOram<V, Z, AB> {
    type V = V;

    // REVIEW NOTE: This function has not been modified.
    fn access<R: Rng + CryptoRng, F: Fn(&V) -> V>(
        &mut self,
        address: Address,
        callback: F,
        rng: &mut R,
    ) -> Result<V, OramError> {
        match self.mode() {
            OramMode::On => {
                let new_position = CompleteBinaryTreeIndex::random_leaf(self.height, rng)?;

                let path_to_evict = {
                    #[cfg(feature = "exact_locations_in_position_map")]
                    {
                        // If the address is out of bounds (>= block_capacity), the access will return the dummy value and update nothing
                        let is_real_access = address.ct_lt(&self.block_capacity()?);

                        let metadata = self.position_map.read(address, rng)?;

                        // If this is a dummy access, we can use new_position as the path to read,
                        // since we won't assign it to anything anyway
                        TreeIndex::conditional_select(
                            &new_position,
                            &metadata.assigned_leaf,
                            is_real_access,
                        )
                    }

                    #[cfg(not(feature = "exact_locations_in_position_map"))]
                    {
                        let new_metadata = BlockMetadata {
                            assigned_leaf: new_position,
                        };
                        self.position_map
                            .write(address, new_metadata, rng)?
                            .assigned_leaf
                    }
                };

                self.stash
                    .read_from_path(&mut self.physical_memory, path_to_evict)?;

                // Scan the stash for the target block, read its value into `result`,
                // and overwrite its position (and possibly its value).
                let result = self.stash.access(address, new_position, callback);

                // Evict blocks from the stash into the path that was just read,
                // replacing them with dummy blocks.
                self.stash
                    .write_to_path(&mut self.physical_memory, path_to_evict)?;

                #[cfg(feature = "exact_locations_in_position_map")]
                {
                    #[cfg(feature = "exact_locations_in_position_map_and_batch_position_map")]
                    {
                        // Update the position map. Limit the batch size so that it fits in the position map's stash
                        for batch_begin in
                            (0..self.stash.entries.len()).step_by(self.stash.path_size.try_into()?)
                        {
                            let batch_end = min(
                                batch_begin + usize::try_from(self.stash.path_size)?,
                                self.stash.entries.len(),
                            );
                            self.batch_update_position_map(batch_begin..batch_end, rng)?;
                        }
                    }

                    #[cfg(not(feature = "exact_locations_in_position_map_and_batch_position_map"))]
                    {
                        for (i, entry) in self.stash.entries.iter().enumerate() {
                            let is_in_tree = entry.exact_bucket.ct_ne(&BlockMetadata::NOT_IN_TREE);
                            let exact_offset = u64::conditional_select(
                                &i.try_into()?,
                                &entry.exact_offset,
                                is_in_tree,
                            );

                            let new_metadata = BlockMetadata {
                                assigned_leaf: entry.block.position,
                                exact_bucket: entry.exact_bucket,
                                exact_offset,
                            };

                            // If the entry is a dummy, write to a dummy position at the end of the map so that no real
                            // position gets overwritten
                            let entry_is_dummy = entry
                                .block
                                .address
                                .ct_eq(&PathOramBlock::<V>::DUMMY_ADDRESS);
                            let entry_address = Address::conditional_select(
                                &entry.block.address,
                                &self.block_capacity()?,
                                entry_is_dummy,
                            );

                            self.position_map.write(entry_address, new_metadata, rng)?;
                        }
                    }
                }

                result
            }
            OramMode::Off => {
                // In off mode we do not perform dummy accesses
                if address >= self.block_capacity()? {
                    return Err(OramError::AddressOutOfBoundsError {
                        attempted: address,
                        capacity: self.block_capacity()?,
                    });
                }

                let result = {
                    #[cfg(feature = "direct_accesses_in_off_mode")]
                    {
                        // We are in off mode so we don't care about oblivious operations.

                        // Cache the block position - we have log N levels of recursion, so without caching
                        // we still have O(log N) overhead on every access even if we store the exact block locations
                        let block_location = match self.blocks_accessed_in_off_mode.entry(address) {
                            hash_map::Entry::Occupied(occupied) => *occupied.get(),
                            hash_map::Entry::Vacant(vacant) => {
                                let metadata = self.position_map.read(address, rng)?;

                                let block_location = {
                                    #[cfg(feature = "exact_locations_in_position_map")]
                                    {
                                        match metadata {
                                            BlockMetadata {
                                                exact_bucket: BlockMetadata::NOT_IN_TREE,
                                                exact_offset: BlockMetadata::UNINITIALIZED,
                                                ..
                                            } => BlockLocation::Dummy,
                                            BlockMetadata {
                                                exact_bucket: BlockMetadata::NOT_IN_TREE,
                                                exact_offset,
                                                ..
                                            } => BlockLocation::Stash {
                                                offset: exact_offset.try_into()?,
                                            },
                                            BlockMetadata {
                                                exact_bucket,
                                                exact_offset,
                                                ..
                                            } => BlockLocation::OramTree {
                                                bucket: exact_bucket.try_into()?,
                                                offset: exact_offset.try_into()?,
                                            },
                                        }
                                    }

                                    #[cfg(not(feature = "exact_locations_in_position_map"))]
                                    {
                                        let mut location = BlockLocation::Dummy;
                                        let mut bucket_idx =
                                            usize::try_from(metadata.assigned_leaf)?;

                                        'buckets: while bucket_idx > 0 {
                                            for (offset, block) in self.physical_memory[bucket_idx]
                                                .blocks
                                                .iter()
                                                .enumerate()
                                            {
                                                // Even though we are in off mode, we still want to use ct_eq - we must not leak
                                                // the addresses of blocks other than the accessed one
                                                if block.address.ct_eq(&address).into() {
                                                    location = BlockLocation::OramTree {
                                                        bucket: bucket_idx,
                                                        offset,
                                                    };
                                                    break 'buckets;
                                                }
                                            }
                                            bucket_idx >>= 1;
                                        }

                                        if matches!(location, BlockLocation::Dummy) {
                                            for (offset, entry) in
                                                self.stash.entries.iter().enumerate()
                                            {
                                                // Even though we are in off mode, we still want to use ct_eq - we must not leak
                                                // the addresses of blocks other than the accessed one
                                                if entry.block.address.ct_eq(&address).into() {
                                                    location = BlockLocation::Stash { offset };
                                                    break;
                                                }
                                            }
                                        }

                                        location
                                    }
                                };

                                *vacant.insert(block_location)
                            }
                        };

                        let block = match block_location {
                            BlockLocation::Stash { offset } => {
                                let stash_entry = &mut self.stash.entries[offset];
                                &mut stash_entry.block
                            }
                            BlockLocation::OramTree { bucket, offset } => {
                                let bucket = &mut self.physical_memory[bucket];
                                &mut bucket.blocks[offset]
                            }
                            BlockLocation::Dummy => &mut PathOramBlock::dummy(),
                        };

                        let result = block.value;
                        block.value = callback(&block.value);
                        result
                    }

                    #[cfg(not(feature = "direct_accesses_in_off_mode"))]
                    {
                        #[cfg(not(feature = "do_not_cache_block_values"))]
                        {
                            let value: &mut V =
                                match self.blocks_accessed_in_off_mode.entry(address) {
                                    hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                                    hash_map::Entry::Vacant(vacant) => {
                                        let metadata = self.position_map.read(address, rng)?;

                                        let result = {
                                            #[cfg(not(feature = "full_reconstruction"))]
                                            {
                                                let mut result = V::default();
                                                let mut bucket_idx =
                                                    usize::try_from(metadata.assigned_leaf)?;

                                                // We still need this to be oblivious, because the exact locations of blocks depend on the paths assigned to other blocks
                                                while bucket_idx > 0 {
                                                    for block in
                                                        &self.physical_memory[bucket_idx].blocks
                                                    {
                                                        let is_requested_block =
                                                            block.address.ct_eq(&address);
                                                        result.conditional_assign(
                                                            &block.value,
                                                            is_requested_block,
                                                        );
                                                    }
                                                    bucket_idx >>= 1;
                                                }

                                                for entry in &self.stash.entries {
                                                    let is_requested_block =
                                                        entry.block.address.ct_eq(&address);
                                                    result.conditional_assign(
                                                        &entry.block.value,
                                                        is_requested_block,
                                                    );
                                                }
                                                result
                                            }

                                            #[cfg(feature = "full_reconstruction")]
                                            'result: {
                                                // We do not need this to be oblivious, since we're going to reconstruct the full ORAM tree anyway
                                                // We search the ORAM tree from the top to take advantage of locality of references (recently accessed)
                                                // blocks are likely to be near the root)
                                                for i in (0..=self.height).rev() {
                                                    let bucket_idx =
                                                        usize::try_from(metadata.assigned_leaf)?
                                                            >> i;
                                                    for block in
                                                        &self.physical_memory[bucket_idx].blocks
                                                    {
                                                        if block.address == address {
                                                            break 'result block.value;
                                                        }
                                                    }
                                                }

                                                for entry in &self.stash.entries {
                                                    if entry.block.address == address {
                                                        break 'result entry.block.value;
                                                    }
                                                }

                                                V::default()
                                            }
                                        };

                                        vacant.insert(result)
                                    }
                                };

                            let result = *value;
                            *value = callback(value);
                            result
                        }

                        #[cfg(feature = "do_not_cache_block_values")]
                        {
                            self.blocks_accessed_in_off_mode.insert(address, ());

                            let metadata = self.position_map.read(address, rng)?;

                            let mut result = V::default();
                            let mut bucket_idx = usize::try_from(metadata.assigned_leaf)?;

                            // We still need this to be oblivious, because the exact locations of blocks depend on the paths assigned to other blocks
                            while bucket_idx > 0 {
                                for block in &mut self.physical_memory[bucket_idx].blocks {
                                    let is_requested_block = block.address.ct_eq(&address);
                                    result.conditional_assign(&block.value, is_requested_block);

                                    let new_value = callback(&block.value);
                                    block
                                        .value
                                        .conditional_assign(&new_value, is_requested_block);
                                }
                                bucket_idx >>= 1;
                            }

                            for entry in &mut self.stash.entries {
                                let is_requested_block = entry.block.address.ct_eq(&address);
                                result.conditional_assign(&entry.block.value, is_requested_block);

                                let new_value = callback(&entry.block.value);
                                entry
                                    .block
                                    .value
                                    .conditional_assign(&new_value, is_requested_block);
                            }

                            result
                        }
                    }
                };

                Ok(result)
            }
        }
    }

    fn batch_access<R: RngCore + CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        callbacks: &[(Address, F)],
        rng: &mut R,
    ) -> Result<Vec<Self::V>, OramError> {
        match self.mode() {
            OramMode::On => {
                let new_positions = CompleteBinaryTreeIndex::random_leaves(
                    callbacks.len().try_into()?,
                    self.height,
                    rng,
                )?;

                let mut paths_to_evict = {
                    #[cfg(feature = "exact_locations_in_position_map")]
                    {
                        let block_capacity = self.block_capacity()?;
                        self.position_map
                            .batch_read(
                                &callbacks
                                    .iter()
                                    .map(|(address, _)| *address)
                                    .collect::<Vec<_>>(),
                                rng,
                            )?
                            .into_iter()
                            .enumerate()
                            .map(|(i, metadata)| {
                                let is_real_access = callbacks[i].0.ct_lt(&block_capacity);
                                TreeIndex::conditional_select(
                                    &new_positions[i],
                                    &metadata.assigned_leaf,
                                    is_real_access,
                                )
                            })
                            .collect::<Vec<_>>()
                    }

                    #[cfg(not(feature = "exact_locations_in_position_map"))]
                    {
                        let mut paths = Vec::with_capacity(callbacks.len());

                        for batch_begin in
                            (0..callbacks.len()).step_by(self.stash.path_size.try_into()?)
                        {
                            let batch_end = min(
                                batch_begin + usize::try_from(self.stash.path_size)?,
                                callbacks.len(),
                            );
                            paths.extend(
                                self.position_map
                                    .batch_write(
                                        &callbacks[batch_begin..batch_end]
                                            .iter()
                                            .map(|(address, _)| *address)
                                            .zip(new_positions[batch_begin..batch_end].iter().map(
                                                |new_position| BlockMetadata {
                                                    assigned_leaf: *new_position,
                                                },
                                            ))
                                            .collect::<Vec<_>>(),
                                        rng,
                                    )?
                                    .into_iter()
                                    .map(|metadata| metadata.assigned_leaf)
                                    .collect::<Vec<_>>(),
                            );
                        }

                        paths
                    }
                };
                paths_to_evict.sort_by_key(|x| cmp::Reverse(*x));
                paths_to_evict.dedup();

                self.stash
                    .read_from_paths(&mut self.physical_memory, &paths_to_evict)?;

                let result = self.stash.batch_access(&new_positions, callbacks)?;

                self.stash
                    .write_to_paths(&mut self.physical_memory, &paths_to_evict)?;

                #[cfg(feature = "exact_locations_in_position_map")]
                {
                    // Update the position map. Limit the batch size so that it fits in the position map's stash
                    for batch_begin in
                        (0..self.stash.entries.len()).step_by(self.stash.path_size.try_into()?)
                    {
                        let batch_end = min(
                            batch_begin + usize::try_from(self.stash.path_size)?,
                            self.stash.entries.len(),
                        );
                        self.batch_update_position_map(batch_begin..batch_end, rng)?;
                    }
                }

                Ok(result)
            }
            OramMode::Off => unimplemented!("We do not generate batch accesses in off mode."),
        }
    }

    fn block_capacity(&self) -> Result<Address, OramError> {
        Ok(u64::try_from(self.physical_memory.len())?)
    }

    fn turn_on<R: RngCore + CryptoRng>(&mut self, rng: &mut R) -> Result<(), OramError> {
        self.mode = OramMode::On;
        // Evictions in the position map will happen during the reads below
        self.position_map.turn_on_without_evicting()?;

        #[cfg(not(feature = "batched_turning_on"))]
        {
            #[cfg(feature = "do_not_cache_block_values")]
            {
                for address in mem::take(&mut self.blocks_accessed_in_off_mode).keys() {
                    // Reading the block causes its path to be evicted.
                    self.read(*address, rng)?;
                }
            }

            #[cfg(not(feature = "do_not_cache_block_values"))]
            {
                #[cfg(not(feature = "full_reconstruction"))]
                {
                    for (address, value) in mem::take(&mut self.blocks_accessed_in_off_mode) {
                        self.write(address, value, rng)?;
                    }
                }

                #[cfg(feature = "full_reconstruction")]
                {
                    for address in 0..self.block_capacity()? {
                        if let Some(value) = self.blocks_accessed_in_off_mode.remove(&address) {
                            self.write(address, value, rng)?;
                        } else {
                            self.read(address, rng)?;
                        }
                    }
                }
            }
        }

        #[cfg(feature = "batched_turning_on")]
        {
            // The case with this feature disabled is not implemented
            const_assert!(cfg!(feature = "direct_accesses_in_off_mode"));

            let addresses = mem::take(&mut self.blocks_accessed_in_off_mode)
                .keys()
                .copied()
                .collect::<Vec<_>>();

            for batch_begin in (0..addresses.len()).step_by(self.stash.path_size.try_into()?) {
                let batch_end = min(
                    batch_begin + usize::try_from(self.stash.path_size)?,
                    addresses.len(),
                );
                self.batch_read(&addresses[batch_begin..batch_end], rng)?;
            }
        }

        Ok(())
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        self.mode = OramMode::Off;
        self.position_map.turn_off()
    }

    fn turn_on_without_evicting(&mut self) -> Result<(), OramError> {
        self.mode = OramMode::On;
        self.position_map.turn_on_without_evicting()
    }

    fn mode(&self) -> OramMode {
        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{bucket::*, test_utils::*};

    use rand::{SeedableRng, rngs::StdRng};

    // Test default parameters. For the small capacity used in the tests, this means a linear position map.
    create_path_oram_correctness_tests!(4, 8, 16384, 40, 1);

    // The remaining tests have RECURSION_CUTOFF = 1 in order to test the recursive position map.

    // Default parameters, but with RECURSION_CUTOFF = 1.
    create_path_oram_correctness_tests!(4, 8, 1, 40, 1);

    // Test small initial stash sizes and correct resizing of stash on overflow.
    create_path_oram_correctness_tests!(4, 8, 1, 10, 1);
    create_path_oram_correctness_tests!(4, 8, 1, 1, 1);

    // Test small and large bucket sizes.
    create_path_oram_correctness_tests!(3, 8, 1, 40, 1);
    create_path_oram_correctness_tests!(5, 8, 1, 40, 1);

    // Test small and large position map blocks.
    create_path_oram_correctness_tests!(4, 2, 1, 40, 1);
    create_path_oram_correctness_tests!(4, 64, 1, 40, 1);

    // "Running sanity checks" for the default parameters.

    // Check that the stash size stays reasonably small over the test runs.
    create_path_oram_stash_size_tests!(4, 8, 16384, 40, 1);

    // Sanity checks on the `DefaultOram` convenience wrapper.
    #[test]
    fn default_oram_linear_correctness() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut oram = DefaultOram::<BlockValue<1>>::new(64, &mut rng).unwrap();
        assert!(matches!(oram.0, DefaultOramBackend::Linear(_)));
        random_workload(&mut oram, 1000);
    }

    // This test is #[ignore]'d because it takes about 1 second to run.
    #[test]
    #[ignore]
    fn default_oram_path_correctness() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut oram = DefaultOram::<BlockValue<1>>::new(2048, &mut rng).unwrap();
        assert!(matches!(oram.0, DefaultOramBackend::Path(_)));
        random_workload(&mut oram, 1000);
    }
}
