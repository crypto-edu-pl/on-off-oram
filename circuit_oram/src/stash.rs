// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! A trait representing a Circuit ORAM stash.

use std::vec;

use crate::{
    Address, BucketSize, OramBlock, OramError, StashSize,
    bucket::{Bucket, CircuitOramBlock},
    utils::{CompleteBinaryTreeIndex, TreeIndex, bitonic_sort_by_keys},
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeLess};

const STASH_GROWTH_INCREMENT: usize = 10;

#[derive(Debug)]
/// A fixed-size, obliviously accessed Circuit ORAM stash data structure implemented using oblivious sorting.
pub struct ObliviousStash<V: OramBlock> {
    pub entries: Vec<StashEntry<V>>,
    pub path_size: StashSize,
    prefix_last_read_from_physical_memory: usize,
    oram_block_capacity: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct StashEntry<V: OramBlock> {
    pub block: CircuitOramBlock<V>,
}

impl<V: OramBlock> StashEntry<V> {
    pub fn dummy() -> Self {
        Self {
            block: CircuitOramBlock::<V>::dummy(),
        }
    }
}

impl<V: OramBlock> ConditionallySelectable for StashEntry<V> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let block = CircuitOramBlock::conditional_select(&a.block, &b.block, choice);
        Self { block }
    }
}

impl<V: OramBlock> ObliviousStash<V> {
    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<V: OramBlock> ObliviousStash<V> {
    pub fn new(
        path_size: StashSize,
        overflow_size: StashSize,
        oram_block_capacity: u64,
    ) -> Result<Self, OramError> {
        let num_stash_blocks: usize = (path_size + overflow_size).try_into()?;

        Ok(Self {
            entries: vec![StashEntry::<V>::dummy(); num_stash_blocks],
            path_size,
            prefix_last_read_from_physical_memory: 0,
            oram_block_capacity,
        })
    }

    pub fn write_to_path<const Z: BucketSize>(
        &mut self,
        physical_memory: &mut [Bucket<V, Z>],
        position: TreeIndex,
    ) -> Result<(), OramError> {
        let height = position.ct_depth();
        let mut level_assignments = vec![TreeIndex::MAX - 1; self.len()];
        let mut level_counts = vec![0; usize::try_from(height)? + 1];

        let mut n_dummy_blocks = 0;

        // Assign all non-dummy blocks in the stash to either the path or the overflow.
        for (i, entry) in self.entries.iter().enumerate() {
            // If `block` is a dummy, the rest of this loop iteration will be a no-op, and the values don't matter.
            let block_is_dummy = entry.block.ct_is_dummy();

            n_dummy_blocks += u64::from(bool::from(block_is_dummy));

            // Set up valid but meaningless input to the computation in case `block` is a dummy.
            let an_arbitrary_leaf: TreeIndex = 1 << height;
            let block_position = TreeIndex::conditional_select(
                &entry.block.position,
                &an_arbitrary_leaf,
                block_is_dummy,
            );

            // Assign the block to a bucket or to the overflow.
            let mut assigned = Choice::from(0);
            // Obliviously scan through the buckets from leaf to root,
            // assigning the block to the first empty bucket satisfying the invariant.
            for (level, count) in level_counts.iter_mut().enumerate().rev() {
                let level_bucket_full: Choice = count.ct_eq(&(u64::try_from(Z)?));

                let level_u64 = u64::try_from(level)?;
                let level_satisfies_invariant = block_position
                    .ct_node_on_path(level_u64, height)
                    .ct_eq(&position.ct_node_on_path(level_u64, height));

                let should_assign = level_satisfies_invariant
                    & (!level_bucket_full)
                    & (!block_is_dummy)
                    & (!assigned);
                assigned |= should_assign;

                let level_count_incremented = *count + 1;
                count.conditional_assign(&level_count_incremented, should_assign);
                level_assignments[i].conditional_assign(&level_u64, should_assign);
            }
            // If the block was not able to be assigned to any bucket, assign it to the overflow.
            level_assignments[i]
                .conditional_assign(&TreeIndex::MAX, (!assigned) & (!block_is_dummy));
        }

        // We need enough dummy blocks to be able to read a path and have a spare
        // block to handle writes to uninitialized blocks.
        let required_dummy_blocks =
            self.path_size - u64::try_from(self.prefix_last_read_from_physical_memory)? + 1;

        // Assign dummy blocks to the remaining non-full buckets until all buckets are full.
        let mut need_more_dummy_blocks: Choice = 1.into();
        let mut first_unassigned_block_index: usize = 0;
        // Unless the stash overflows, this loop will execute exactly once, and the inner `if` will not execute.
        // If the stash overflows, this loop will execute twice and the inner `if` will execute.
        // This difference in control flow will leak the fact that the stash has overflowed.
        // This is a violation of obliviousness, but the alternative is simply to fail.
        // If the stash is set large enough when the ORAM is initialized,
        // stash overflow will occur only with negligible probability.
        while need_more_dummy_blocks.into() {
            // Make a pass over the stash, assigning dummy blocks to unfilled levels in the path.
            for (i, entry) in self
                .entries
                .iter()
                .enumerate()
                .skip(first_unassigned_block_index)
            {
                let block_free = entry.block.ct_is_dummy();

                let mut assigned: Choice = 0.into();
                for (level, count) in level_counts.iter_mut().enumerate() {
                    let full = count.ct_eq(&(u64::try_from(Z)?));
                    let no_op = assigned | full | !block_free;

                    level_assignments[i].conditional_assign(&(u64::try_from(level))?, !no_op);
                    count.conditional_assign(&(*count + 1), !no_op);
                    assigned |= !no_op;
                }

                n_dummy_blocks -= u64::from(bool::from(block_free & assigned));
            }

            // Check that all levels have been filled and there are enough dummy blocks.
            need_more_dummy_blocks = 0.into();
            for count in level_counts.iter() {
                let full = count.ct_eq(&(u64::try_from(Z)?));
                need_more_dummy_blocks |= !full;
            }
            need_more_dummy_blocks |= n_dummy_blocks.ct_lt(&required_dummy_blocks);

            // If not, there must not have been enough dummy blocks remaining in the stash.
            // That is, the stash has overflowed.
            // So, extend the stash with STASH_GROWTH_INCREMENT more dummy blocks,
            // and repeat the process of trying to fill all unfilled levels with dummy blocks.
            if need_more_dummy_blocks.into() {
                first_unassigned_block_index = self.entries.len() - 1;

                self.entries.resize(
                    self.entries.len() + STASH_GROWTH_INCREMENT,
                    StashEntry::<V>::dummy(),
                );
                level_assignments.resize(
                    level_assignments.len() + STASH_GROWTH_INCREMENT,
                    TreeIndex::MAX - 1,
                );
                n_dummy_blocks += u64::try_from(STASH_GROWTH_INCREMENT)?;

                log::warn!(
                    "Stash overflow occurred. Stash resized to {} blocks.",
                    self.entries.len()
                );
            }
        }

        bitonic_sort_by_keys(&mut self.entries, &mut level_assignments);

        // Write the first Z * height blocks into slots in the tree
        for depth in 0..=height {
            let bucket_index = usize::try_from(position.ct_node_on_path(depth, height))?;
            let bucket_to_write = &mut physical_memory[bucket_index];
            for slot_number in 0..Z {
                let stash_index = (usize::try_from(depth)?) * Z + slot_number;

                bucket_to_write.blocks[slot_number] = self.entries[stash_index].block;
            }
        }

        Ok(())
    }

    pub fn access<F: Fn(&V) -> V>(
        &mut self,
        address: Address,
        new_position: TreeIndex,
        value_callback: F,
    ) -> Result<V, OramError> {
        let (result, found) = self.try_update(address, new_position, &value_callback)?;

        // If a block with address `address` is not found,
        // initialize one by writing to a reserved block in the stash,
        // which will always be a dummy block.
        let reserved_block_index = self.prefix_last_read_from_physical_memory;
        let reserved_entry = &mut self.entries[reserved_block_index];
        assert!(bool::from(reserved_entry.block.ct_is_dummy()));

        // Do not initialize the block if this is a dummy access
        let is_real_access = address.ct_lt(&self.oram_block_capacity);

        reserved_entry.block.conditional_assign(
            &CircuitOramBlock {
                value: value_callback(&result),
                address,
                position: new_position,
            },
            !found & is_real_access,
        );

        // Return the value of the found block (or the default value, if no block was found)
        Ok(result)
    }

    fn try_update<F: Fn(&V) -> V>(
        &mut self,
        address: Address,
        new_position: TreeIndex,
        value_callback: &F,
    ) -> Result<(V, Choice), OramError> {
        let mut result: V = V::default();
        let mut found: Choice = 0.into();

        // Iterate over stash, updating the block with address `address` if one exists.
        for entry in &mut self.entries {
            let is_requested_index = entry.block.address.ct_eq(&address);
            found.conditional_assign(&1.into(), is_requested_index);

            // Read current value of target block into `result`.
            result.conditional_assign(&entry.block.value, is_requested_index);
            // Write new position into target block.
            entry
                .block
                .position
                .conditional_assign(&new_position, is_requested_index);
            // If a write, write new value into target block.
            let value_to_write = value_callback(&result);
            entry
                .block
                .value
                .conditional_assign(&value_to_write, is_requested_index);
        }

        Ok((result, found))
    }

    #[cfg(test)]
    pub fn occupancy(&self) -> StashSize {
        let mut result = 0;
        for i in self.prefix_last_read_from_physical_memory..self.entries.len() {
            if !self.entries[i].block.is_dummy() {
                result += 1;
            }
        }
        result
    }

    pub fn read_from_path<const Z: crate::BucketSize>(
        &mut self,
        physical_memory: &mut [Bucket<V, Z>],
        position: TreeIndex,
    ) -> Result<(), OramError> {
        let height = position.ct_depth();

        for i in (0..(self.path_size / u64::try_from(Z)?)).rev() {
            let bucket_index = position.ct_node_on_path(i, height);
            let bucket = physical_memory[usize::try_from(bucket_index)?];
            for slot_index in 0..Z {
                self.entries[Z * (usize::try_from(i)?) + slot_index] = StashEntry {
                    block: bucket.blocks[slot_index],
                }
            }
        }

        let stash_index = self.path_size.try_into()?;
        if stash_index < self.prefix_last_read_from_physical_memory {
            self.entries[stash_index..self.prefix_last_read_from_physical_memory]
                .fill(StashEntry::dummy());
        }
        self.prefix_last_read_from_physical_memory = stash_index;

        Ok(())
    }
}
