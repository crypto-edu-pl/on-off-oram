// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! An implementation of Circuit ORAM.

use std::collections::HashMap;
#[cfg(not(feature = "full_reconstruction"))]
use std::mem;

#[cfg(not(feature = "do_not_cache_block_values"))]
use std::collections::hash_map;

use rand::{CryptoRng, RngExt};

use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
};

use super::{position_map::PositionMap, stash::ObliviousStash};
use crate::{
    Address, BlockSize, BucketSize, Oram, OramBlock, OramError, OramMode, RecursionCutoff,
    StashSize,
    bucket::{BlockMetadata, Bucket, CircuitOramBlock, PositionBlock},
    linear_time_oram::LinearTimeOram,
    utils::{CompleteBinaryTreeIndex, TreeHeight, TreeIndex},
};

/// The default cutoff size in blocks
/// below which `CircuitOram` uses a linear position map instead of a recursive one.
pub const DEFAULT_RECURSION_CUTOFF: RecursionCutoff = 1 << 6;

/// The parameter "Z" from the Circuit ORAM literature that sets the number of blocks per bucket; typical values are 3 or 4.
/// Here we adopt the more conservative setting of 4.
pub const DEFAULT_BLOCKS_PER_BUCKET: BucketSize = 4;

/// The default number of positions stored per position block.
pub const DEFAULT_POSITIONS_PER_BLOCK: BlockSize = 8;

/// The default number of overflow blocks that the Circuit ORAM stash (and recursive stashes) can store.
pub const DEFAULT_STASH_OVERFLOW_SIZE: StashSize = 40;

/// The cutoff size in blocks below which `DefaultOram` is simply a linear time ORAM.
pub const LINEAR_TIME_ORAM_CUTOFF: RecursionCutoff = 1 << 6;

/// A doubly oblivious Circuit ORAM.
///
/// ## Parameters
///
/// - Block type `V`: the type of elements stored by the ORAM.
/// - Bucket size `Z`: the number of blocks per Circuit ORAM bucket.
///     Must be at least 2. Typical values are 3, 4, or 5.
///     Along with the overflow size, this value affects the probability
///     of stash overflow (see below) and should be set with care.
/// - Positions per block `AB`:
///     The number of positions stored in each block of the recursive position map ORAM.
///     Must be a power of two and must be at least 2 (otherwise the recursion will not terminate).
///     Otherwise, can be freely tuned for performance.
///     Larger `AB` means fewer levels of recursion but higher costs for accessing each level.
/// - Recursion threshold: the maximum number of position blocks that will be stored in a recursive Circuit ORAM.
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
/// With Z = 4, experiments from the [original Circuit ORAM paper](https://eprint.iacr.org/2013/280.pdf)
/// indicate that the probability of overflow is independent of the number N of blocks stored,
/// and that setting SO = 40 is enough to reduce this probability to below 2^{-50} (Figure 3).
/// The authors conservatively estimate that setting SO = 89 suffices for 2^{-80} overflow probability.
/// The choice Z = 3 is also popular, although the probability of overflow is less well understood.
#[derive(Debug)]
pub struct CircuitOram<V: OramBlock, const Z: BucketSize, const AB: BlockSize> {
    /// The underlying untrusted memory that the ORAM is obliviously accessing on behalf of its client.
    physical_memory: Vec<Bucket<V, Z>>,
    /// The Circuit ORAM stash.
    stash: ObliviousStash<V>,
    /// The Circuit ORAM position map.
    // TODO: use a different block size?
    position_map: PositionMap<AB, Z>,
    /// The height of the Circuit ORAM tree data structure.
    height: TreeHeight,
    /// The counter used for the reverse-lexicographic deterministic path eviction.
    timestep: TreeIndex,
    /// Current mode.
    mode: OramMode,
    /// The set of blocks accessed in off mode. Their paths will be evicted when ORAM is turned on.
    ///
    /// This is not an oblivious data structure, but we use it only in off mode so it's fine.
    #[cfg(not(feature = "do_not_cache_block_values"))]
    blocks_accessed_in_off_mode: HashMap<Address, V>,
    #[cfg(feature = "do_not_cache_block_values")]
    blocks_accessed_in_off_mode: HashMap<Address, ()>,
}

/// An `Oram` suitable for most use cases, with reasonable default choices of parameters.
#[derive(Debug)]
pub struct DefaultOram<V: OramBlock>(DefaultOramBackend<V>);

#[derive(Debug)]
enum DefaultOramBackend<V: OramBlock> {
    Circuit(CircuitOram<V, DEFAULT_BLOCKS_PER_BUCKET, DEFAULT_POSITIONS_PER_BLOCK>),
    Linear(LinearTimeOram<V>),
}

impl<V: OramBlock> Oram for DefaultOram<V> {
    type V = V;

    fn block_capacity(&self) -> Result<Address, OramError> {
        match &self.0 {
            DefaultOramBackend::Circuit(p) => p.block_capacity(),
            DefaultOramBackend::Linear(l) => l.block_capacity(),
        }
    }

    fn access<R: rand::CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        index: Address,
        callback: F,
        rng: &mut R,
    ) -> Result<Self::V, OramError> {
        match &mut self.0 {
            DefaultOramBackend::Circuit(p) => p.access(index, callback, rng),
            DefaultOramBackend::Linear(l) => l.access(index, callback, rng),
        }
    }

    fn turn_on<R: CryptoRng>(&mut self, rng: &mut R) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Circuit(p) => p.turn_on(rng),
            DefaultOramBackend::Linear(l) => l.turn_on(rng),
        }
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Circuit(p) => p.turn_off(),
            DefaultOramBackend::Linear(l) => l.turn_off(),
        }
    }

    fn turn_on_without_evicting(&mut self) -> Result<(), OramError> {
        match &mut self.0 {
            DefaultOramBackend::Circuit(p) => p.turn_on_without_evicting(),
            DefaultOramBackend::Linear(l) => l.turn_on_without_evicting(),
        }
    }

    fn mode(&self) -> OramMode {
        match &self.0 {
            DefaultOramBackend::Circuit(p) => p.mode(),
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
    pub fn new<R: CryptoRng>(block_capacity: Address, rng: &mut R) -> Result<Self, OramError> {
        if block_capacity < LINEAR_TIME_ORAM_CUTOFF {
            Ok(Self(DefaultOramBackend::Linear(LinearTimeOram::new(
                block_capacity,
            )?)))
        } else {
            Ok(Self(DefaultOramBackend::Circuit(CircuitOram::<
                V,
                DEFAULT_BLOCKS_PER_BUCKET,
                DEFAULT_POSITIONS_PER_BLOCK,
            >::new_with_parameters(
                block_capacity,
                rng,
                DEFAULT_STASH_OVERFLOW_SIZE,
                DEFAULT_RECURSION_CUTOFF,
            )?)))
        }
    }
}

impl<V: OramBlock, const Z: BucketSize, const AB: BlockSize> CircuitOram<V, Z, AB> {
    /// Returns a new `CircuitOram` mapping addresses `0 <= address < block_capacity` to default `V` values,
    /// with a stash overflow size of `overflow_size` blocks, and a recursion cutoff of `recursion_cutoff`.
    /// (See [`CircuitOram`]) for a description of these parameters).
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
    pub fn new_with_parameters<R: CryptoRng>(
        block_capacity: Address,
        rng: &mut R,
        overflow_size: StashSize,
        recursion_cutoff: RecursionCutoff,
    ) -> Result<Self, OramError> {
        log::info!("CircuitOram::new(capacity = {})", block_capacity,);

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
        let stash = ObliviousStash::new(path_size, overflow_size)?;

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

        let num_address_blocks = if block_capacity.is_multiple_of(ab_address) {
            block_capacity / ab_address
        } else {
            block_capacity / ab_address + 1
        };
        for block_index in 0..num_address_blocks {
            let mut data = [BlockMetadata::default(); AB];
            for metadata in &mut data {
                metadata.assigned_leaf = rng.random_range(first_leaf_index..=last_leaf_index);
            }
            let position_block = PositionBlock { data };
            position_map.write_position_block(block_index * ab_address, position_block, rng)?;
        }

        Ok(Self {
            physical_memory,
            stash,
            position_map,
            height,
            timestep: 0,
            mode: OramMode::On,
            blocks_accessed_in_off_mode: HashMap::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn stash_occupancy(&self) -> StashSize {
        self.stash.occupancy()
    }

    fn read_and_rm(&mut self, address: Address, path_to_evict: TreeIndex) -> Result<V, OramError> {
        let mut result = V::default();
        let mut bucket_idx = usize::try_from(path_to_evict)?;

        let dummy = CircuitOramBlock::dummy();

        while bucket_idx > 0 {
            for block in &mut self.physical_memory[bucket_idx].blocks {
                let is_requested_block = block.address.ct_eq(&address);
                result.conditional_assign(&block.value, is_requested_block);
                block.conditional_assign(&dummy, is_requested_block);
            }
            bucket_idx >>= 1;
        }

        for entry in &mut self.stash.entries {
            let is_requested_block = entry.block.address.ct_eq(&address);
            result.conditional_assign(&entry.block.value, is_requested_block);
            entry.block.conditional_assign(&dummy, is_requested_block);
        }

        Ok(result)
    }

    fn evict(&mut self) -> Result<(), OramError> {
        self.evict_deterministic()
    }

    fn evict_deterministic(&mut self) -> Result<(), OramError> {
        let path_0 = self.next_eviction_path();
        let path_1 = self.next_eviction_path();

        self.evict_once_fast(path_0)?;
        self.evict_once_fast(path_1)?;

        Ok(())
    }

    /// Choose the next eviction path using the reverse-lexicographic order
    fn next_eviction_path(&mut self) -> TreeIndex {
        let path = (1 << self.height)
            + (self
                .timestep
                .reverse_bits()
                .unbounded_shr((64 - self.height) as u32));
        self.timestep = (self.timestep + 1) & ((1 << self.height) - 1);
        path
    }

    fn evict_once_fast(&mut self, path: TreeIndex) -> Result<(), OramError> {
        let deepest = self.prepare_deepest(path)?;
        let target = self.prepare_target(path, &deepest)?;

        let mut hold = CircuitOramBlock::dummy();
        let mut dest = TreeHeight::MAX;

        for i in 0..=self.height + 1 {
            let mut towrite = CircuitOramBlock::dummy();

            let is_dest = !hold.ct_is_dummy() & i.ct_eq(&dest);
            {
                towrite.conditional_assign(&hold, is_dest);
                hold.conditional_assign(&CircuitOramBlock::dummy(), is_dest);
                dest.conditional_assign(&TreeHeight::MAX, is_dest);
            }

            let bucket_idx = if i > 0 {
                path.ct_node_on_path(i - 1, self.height)
            } else {
                0
            };

            {
                let has_target = target[i as usize].0.ct_ne(&TreeHeight::MAX);
                let deepest_block = if i == 0 {
                    self.stash
                        .conditional_remove(target[i as usize].1, has_target)?
                } else {
                    self.conditional_remove(bucket_idx, target[i as usize].1, has_target)?
                };
                hold.conditional_assign(&deepest_block, has_target);
                dest.conditional_assign(&target[i as usize].0, has_target);
            }

            if i > 0 {
                self.conditional_insert(bucket_idx, towrite, is_dest)?;
            }
        }

        Ok(())
    }

    // Compared to the paper, we additionally store the stash/bucket offset of the deepest block here, so that we don't have to
    // look for it again during the eviction
    fn prepare_deepest(&self, path: TreeIndex) -> Result<Vec<(TreeHeight, u32)>, OramError> {
        let mut deepest = vec![(TreeHeight::MAX, u32::MAX); self.height as usize + 2];
        let mut src = TreeHeight::MAX;
        let mut src_offset = u32::MAX;
        let mut goal = 0;

        let (stash_deepest_level, stash_deepest_offset) =
            Self::find_block_with_deepest_level(&self.stash.entries, path, |entry| &entry.block);

        let stash_not_empty = stash_deepest_offset.ct_ne(&u32::MAX);
        src.conditional_assign(&0, stash_not_empty);
        src_offset.conditional_assign(&stash_deepest_offset, stash_not_empty);
        goal.conditional_assign(&stash_deepest_level, stash_not_empty);

        for i in 1..=self.height + 1 {
            let src_block_can_reside_in_node = !goal.ct_lt(&i);
            deepest[i as usize]
                .0
                .conditional_assign(&src, src_block_can_reside_in_node);
            deepest[i as usize]
                .1
                .conditional_assign(&src_offset, src_block_can_reside_in_node);

            let bucket_idx = path.ct_node_on_path(i - 1, self.height);
            let (bucket_deepest_level, bucket_deepest_offset) = Self::find_block_with_deepest_level(
                &self.physical_memory[bucket_idx as usize].blocks,
                path,
                |block| block,
            );

            let found_new_deepest_level = bucket_deepest_level.ct_gt(&goal);
            goal.conditional_assign(&bucket_deepest_level, found_new_deepest_level);
            src.conditional_assign(&i, found_new_deepest_level);
            src_offset.conditional_assign(&bucket_deepest_offset, found_new_deepest_level);
        }

        Ok(deepest)
    }

    // Here we operate on bucket indices instead of levels
    fn find_block_with_deepest_level<
        T,
        I: IntoIterator<Item = T>,
        F: Fn(&T) -> &CircuitOramBlock<V>,
    >(
        container: I,
        path: TreeIndex,
        get_block: F,
    ) -> (TreeHeight, u32) {
        let mut deepest_level = 0;
        let mut result_offset = u32::MAX;

        for (offset, entry) in container.into_iter().enumerate() {
            let entry_deepest_level = get_block(&entry)
                .position
                .deepest_common_ancestor_of_leaves(&path)
                .ct_level_unchecked();
            let found_new_deepest =
                !get_block(&entry).ct_is_dummy() & entry_deepest_level.ct_gt(&deepest_level);

            result_offset.conditional_assign(&(offset as u32), found_new_deepest);
            deepest_level.conditional_assign(&entry_deepest_level, found_new_deepest);
        }

        (deepest_level, result_offset)
    }

    fn prepare_target(
        &self,
        path: TreeIndex,
        deepest: &[(TreeHeight, u32)],
    ) -> Result<Vec<(TreeHeight, u32)>, OramError> {
        let mut dest = TreeHeight::MAX;
        let mut src = TreeHeight::MAX;
        let mut src_offset = u32::MAX;
        let mut target = vec![(TreeHeight::MAX, u32::MAX); self.height as usize + 2];

        for i in (0..=self.height + 1).rev() {
            let is_src = i.ct_eq(&src);
            target[i as usize].0.conditional_assign(&dest, is_src);
            target[i as usize].1.conditional_assign(&src_offset, is_src);
            dest.conditional_assign(&TreeHeight::MAX, is_src);
            src.conditional_assign(&TreeHeight::MAX, is_src);
            src_offset.conditional_assign(&u32::MAX, is_src);

            let mut has_empty_slot = Choice::from(0);
            if i > 0 {
                let bucket_idx = path.ct_node_on_path(i - 1, self.height);
                for block in &self.physical_memory[bucket_idx as usize].blocks {
                    has_empty_slot |= block.ct_is_dummy();
                }
            }

            let has_source = ((dest.ct_eq(&TreeHeight::MAX) & has_empty_slot)
                | target[i as usize].0.ct_ne(&TreeHeight::MAX))
                & deepest[i as usize].0.ct_ne(&TreeHeight::MAX);

            src.conditional_assign(&deepest[i as usize].0, has_source);
            src_offset.conditional_assign(&deepest[i as usize].1, has_source);
            dest.conditional_assign(&i, has_source);
        }

        Ok(target)
    }

    fn conditional_remove(
        &mut self,
        bucket_idx: TreeIndex,
        bucket_offset: u32,
        choice: Choice,
    ) -> Result<CircuitOramBlock<V>, OramError> {
        let mut result = CircuitOramBlock::dummy();

        for (offset, slot) in self.physical_memory[bucket_idx as usize]
            .blocks
            .iter_mut()
            .enumerate()
        {
            let should_remove = choice & (offset as u32).ct_eq(&bucket_offset);

            result.conditional_assign(slot, should_remove);
            slot.conditional_assign(&CircuitOramBlock::dummy(), should_remove);
        }

        assert!(bool::from(result.ct_is_dummy().ct_eq(&!choice)));

        Ok(result)
    }

    fn conditional_insert(
        &mut self,
        bucket_idx: TreeIndex,
        block: CircuitOramBlock<V>,
        choice: Choice,
    ) -> Result<(), OramError> {
        let mut inserted = Choice::from(0);

        for slot in &mut self.physical_memory[bucket_idx as usize].blocks {
            let is_dummy = slot.ct_is_dummy();
            let should_insert = choice & !inserted & is_dummy;

            slot.conditional_assign(&block, should_insert);

            inserted |= should_insert;
        }

        assert!(bool::from(choice.ct_eq(&inserted)));

        Ok(())
    }
}

impl<V: OramBlock, const Z: BucketSize, const AB: BlockSize> Oram for CircuitOram<V, Z, AB> {
    type V = V;

    // REVIEW NOTE: This function has not been modified.
    fn access<R: CryptoRng, F: Fn(&V) -> V>(
        &mut self,
        address: Address,
        callback: F,
        rng: &mut R,
    ) -> Result<V, OramError> {
        match self.mode() {
            OramMode::On => {
                let new_position = CompleteBinaryTreeIndex::random_leaf(self.height, rng)?;

                let path_to_evict = {
                    let new_metadata = BlockMetadata {
                        assigned_leaf: new_position,
                    };
                    self.position_map
                        .write(address, new_metadata, rng)?
                        .assigned_leaf
                };

                let result = self.read_and_rm(address, path_to_evict)?;

                self.stash
                    .add_block(address, new_position, callback(&result))?;

                self.evict()?;

                Ok(result)
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
                    #[cfg(not(feature = "do_not_cache_block_values"))]
                    {
                        let value: &mut V = match self.blocks_accessed_in_off_mode.entry(address) {
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
                                            for block in &self.physical_memory[bucket_idx].blocks {
                                                let is_requested_block =
                                                    block.address.ct_eq(&address);
                                                result.conditional_assign(
                                                    &block.value,
                                                    is_requested_block,
                                                );
                                            }
                                            bucket_idx >>= 1;
                                        }

                                        for entry in &self.stash.entries
                                            [usize::try_from(self.stash.path_size).unwrap()..]
                                        {
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
                                                usize::try_from(metadata.assigned_leaf)? >> i;
                                            for block in &self.physical_memory[bucket_idx].blocks {
                                                if block.address == address {
                                                    break 'result block.value;
                                                }
                                            }
                                        }

                                        for entry in &self.stash.entries
                                            [usize::try_from(self.stash.path_size).unwrap()..]
                                        {
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

                        for entry in &mut self.stash.entries
                            [usize::try_from(self.stash.path_size).unwrap()..]
                        {
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
                };

                Ok(result)
            }
        }
    }

    fn block_capacity(&self) -> Result<Address, OramError> {
        Ok(u64::try_from(self.physical_memory.len())?)
    }

    fn turn_on<R: CryptoRng>(&mut self, rng: &mut R) -> Result<(), OramError> {
        self.mode = OramMode::On;
        // Evictions in the position map will happen during the reads below
        self.position_map.turn_on_without_evicting()?;

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
    create_circuit_oram_correctness_tests!(4, 8, 16384, 40);

    // The remaining tests have RECURSION_CUTOFF = 1 in order to test the recursive position map.

    // Default parameters, but with RECURSION_CUTOFF = 1.
    create_circuit_oram_correctness_tests!(4, 8, 1, 40);

    // Test small initial stash sizes and correct resizing of stash on overflow.
    create_circuit_oram_correctness_tests!(4, 8, 1, 10);
    create_circuit_oram_correctness_tests!(4, 8, 1, 1);

    // Test small and large bucket sizes.
    create_circuit_oram_correctness_tests!(3, 8, 1, 40);
    create_circuit_oram_correctness_tests!(5, 8, 1, 40);

    // Test small and large position map blocks.
    create_circuit_oram_correctness_tests!(4, 2, 1, 40);
    create_circuit_oram_correctness_tests!(4, 64, 1, 40);

    // "Running sanity checks" for the default parameters.

    // Check that the stash size stays reasonably small over the test runs.
    create_circuit_oram_stash_size_tests!(4, 8, 16384, 40);

    // Sanity checks on the `DefaultOram` convenience wrapper.
    #[test]
    fn default_oram_linear_correctness() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut oram = DefaultOram::<BlockValue<1>>::new(32, &mut rng).unwrap();
        assert!(matches!(oram.0, DefaultOramBackend::Linear(_)));
        random_workload(&mut oram, 1000);
    }

    #[test]
    fn default_oram_path_correctness() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut oram = DefaultOram::<BlockValue<1>>::new(2048, &mut rng).unwrap();
        assert!(matches!(oram.0, DefaultOramBackend::Circuit(_)));
        random_workload(&mut oram, 1000);
    }
}
