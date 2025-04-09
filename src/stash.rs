// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! A trait representing a Path ORAM stash.

use std::{cmp, vec};

use crate::{
    bucket::{BlockMetadata, Bucket, PathOramBlock},
    utils::{bitonic_sort_by_keys, CompleteBinaryTreeIndex, TreeIndex},
    Address, BucketSize, OramBlock, OramError, StashSize,
};

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeLess};

const STASH_GROWTH_INCREMENT: usize = 10;

#[derive(Debug)]
/// A fixed-size, obliviously accessed Path ORAM stash data structure implemented using oblivious sorting.
pub struct ObliviousStash<V: OramBlock> {
    pub entries: Vec<StashEntry<V>>,
    pub path_size: StashSize,
    max_batch_size: StashSize,
    prefix_last_read_from_physical_memory: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StashEntry<V: OramBlock> {
    pub block: PathOramBlock<V>,
    pub exact_bucket: TreeIndex,
    pub exact_offset: u64,
}

impl<V: OramBlock> StashEntry<V> {
    const DUMMY_OFFSET: u64 = u64::MAX;

    pub fn dummy() -> Self {
        Self {
            block: PathOramBlock::<V>::dummy(),
            exact_bucket: BlockMetadata::NOT_IN_TREE,
            exact_offset: Self::DUMMY_OFFSET,
        }
    }
}

impl<V: OramBlock> ConditionallySelectable for StashEntry<V> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let block = PathOramBlock::conditional_select(&a.block, &b.block, choice);
        let exact_bucket = u64::conditional_select(&a.exact_bucket, &b.exact_bucket, choice);
        let exact_offset = u64::conditional_select(&a.exact_offset, &b.exact_offset, choice);
        Self {
            block,
            exact_bucket,
            exact_offset,
        }
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
        max_batch_size: StashSize,
    ) -> Result<Self, OramError> {
        // Make the stash larger - path_size * path_size - so that it can fit the paths of a batch
        // of size path_size (this means that on a single Path ORAM access we can update the position map
        // in a single top-level batch)
        // TODO think if other stash and batch sizes make more sense
        let num_stash_blocks: usize = (path_size * max_batch_size + overflow_size).try_into()?;

        Ok(Self {
            entries: vec![StashEntry::<V>::dummy(); num_stash_blocks],
            path_size,
            max_batch_size,
            prefix_last_read_from_physical_memory: 0,
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

        // We need enough dummy blocks to be able to read a maximum size batch of paths and have self.max_batch_size spare
        // blocks to handle writes to uninitialized blocks.
        let required_dummy_blocks = self.max_batch_size * self.path_size
            - u64::try_from(self.prefix_last_read_from_physical_memory)?
            + self.max_batch_size;

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
                first_unassigned_block_index =
                    self.entries.len() - usize::try_from(self.max_batch_size)?;

                self.entries.resize(
                    self.entries.len() + STASH_GROWTH_INCREMENT,
                    StashEntry::<V>::dummy(),
                );
                level_assignments.resize(
                    level_assignments.len() + STASH_GROWTH_INCREMENT,
                    TreeIndex::MAX,
                );

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

                self.entries[stash_index].exact_bucket = bucket_index.try_into()?;
                self.entries[stash_index].exact_offset = slot_number.try_into()?;
            }
        }

        Ok(())
    }

    pub fn write_to_paths<const Z: BucketSize>(
        &mut self,
        physical_memory: &mut [Bucket<V, Z>],
        positions: &[TreeIndex],
    ) -> Result<(), OramError> {
        let height = positions[0].ct_depth();
        // The assignments will be:
        // - a bucket index for blocks that get assigned to a bucket
        // - TreeIndex::MAX - 1 for dummy blocks
        // - TreeIndex::MAX for other real blocks ("the overflow")
        let mut bucket_assignments = vec![TreeIndex::MAX - 1; self.len()];

        // Fun fact: the maximal number of buckets in a subtree comprised of k paths is k * (height + 1) - (kth sorting number)
        // (the fact that it is a sorting number found using OEIS). This is because the second path has at least one node in common
        // with the first one, then the next two have at least two in common with them, then the next four at least three and so on.
        // (This is easiest to see when looking at the path as a binary representation of the leaf's index.)
        //
        // The positions (paths) are public knowledge, so we don't have to do computations that rely only on them obliviously.
        let mut bucket_counts = prepare_bucket_counts(positions);

        let mut n_dummy_blocks = 0;

        // Assign all non-dummy blocks in the stash to either a path or the overflow.
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
            for (bucket, count) in bucket_counts.iter_mut() {
                let bucket_full: Choice = count.ct_eq(&(u64::try_from(Z)?));

                let level_satisfies_invariant = bucket.ct_is_ancestor_of_other(&block_position);

                let should_assign =
                    level_satisfies_invariant & (!bucket_full) & (!block_is_dummy) & (!assigned);
                assigned |= should_assign;

                let level_count_incremented = *count + 1;
                count.conditional_assign(&level_count_incremented, should_assign);
                bucket_assignments[i].conditional_assign(bucket, should_assign);
            }
            // If the block was not able to be assigned to any bucket, assign it to the overflow.
            bucket_assignments[i]
                .conditional_assign(&TreeIndex::MAX, (!assigned) & (!block_is_dummy));
        }

        // We need enough dummy blocks to be able to read a maximum size batch of paths and have self.max_batch_size spare
        // blocks to handle writes to uninitialized blocks.
        let required_dummy_blocks = self.max_batch_size * self.path_size
            - u64::try_from(self.prefix_last_read_from_physical_memory)?
            + self.max_batch_size;

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
                for (bucket, count) in bucket_counts.iter_mut() {
                    let full = count.ct_eq(&(u64::try_from(Z)?));
                    let no_op = assigned | full | !block_free;

                    bucket_assignments[i].conditional_assign(bucket, !no_op);
                    count.conditional_assign(&(*count + 1), !no_op);
                    assigned |= !no_op;
                }

                n_dummy_blocks -= u64::from(bool::from(block_free & assigned));
            }

            // Check that all levels have been filled and there are enough dummy blocks.
            need_more_dummy_blocks = 0.into();
            for (_, count) in bucket_counts.iter() {
                let full = count.ct_eq(&(u64::try_from(Z)?));
                need_more_dummy_blocks |= !full;
            }
            need_more_dummy_blocks |= n_dummy_blocks.ct_lt(&required_dummy_blocks);

            // If not, there must not have been enough dummy blocks remaining in the stash.
            // That is, the stash has overflowed.
            // So, extend the stash with STASH_GROWTH_INCREMENT more dummy blocks,
            // and repeat the process of trying to fill all unfilled levels with dummy blocks.
            if need_more_dummy_blocks.into() {
                first_unassigned_block_index =
                    self.entries.len() - usize::try_from(self.max_batch_size)?;

                self.entries.resize(
                    self.entries.len() + STASH_GROWTH_INCREMENT,
                    StashEntry::<V>::dummy(),
                );
                bucket_assignments.resize(
                    bucket_assignments.len() + STASH_GROWTH_INCREMENT,
                    TreeIndex::MAX,
                );

                log::warn!(
                    "Stash overflow occurred. Stash resized to {} blocks.",
                    self.entries.len()
                );
            }
        }

        bitonic_sort_by_keys(&mut self.entries, &mut bucket_assignments);

        // Write the first Z * bucket_counts.len() blocks into slots in the tree
        for i in 0..bucket_counts.len() {
            for slot_number in 0..Z {
                let stash_index = i * Z + slot_number;
                let bucket_index = usize::try_from(bucket_assignments[stash_index])?;
                let bucket_to_write = &mut physical_memory[bucket_index];

                bucket_to_write.blocks[slot_number] = self.entries[stash_index].block;

                self.entries[stash_index].exact_bucket = bucket_index.try_into()?;
                self.entries[stash_index].exact_offset = slot_number.try_into()?;
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
        //
        // Note that the batch size is public information, so a direct access like this is fine.
        let reserved_block_index = self.prefix_last_read_from_physical_memory;
        let reserved_entry = &mut self.entries[reserved_block_index];
        assert!(bool::from(reserved_entry.block.ct_is_dummy()));
        reserved_entry.block.conditional_assign(
            &PathOramBlock {
                value: value_callback(&result),
                address,
                position: new_position,
            },
            !found,
        );

        // Return the value of the found block (or the default value, if no block was found)
        Ok(result)
    }

    pub fn batch_access<F: Fn(&V) -> V>(
        &mut self,
        new_positions: &[TreeIndex],
        value_callbacks: &[(Address, F)],
    ) -> Result<Vec<V>, OramError> {
        assert_eq!(new_positions.len(), value_callbacks.len());

        let mut results = Vec::with_capacity(value_callbacks.len());

        for (i, (new_position, (address, value_callback))) in
            new_positions.iter().zip(value_callbacks).enumerate()
        {
            let (result, found) = self.try_update(*address, *new_position, &value_callback)?;

            // If a block with address `address` is not found,
            // initialize one by writing to a reserved block in the stash,
            // which will always be a dummy block
            let reserved_block_index = self.prefix_last_read_from_physical_memory + i;
            let reserved_entry = &mut self.entries[reserved_block_index];
            assert!(bool::from(reserved_entry.block.ct_is_dummy()));
            reserved_entry.block.conditional_assign(
                &PathOramBlock {
                    value: value_callback(&result),
                    address: *address,
                    position: *new_position,
                },
                !found,
            );

            results.push(result);
        }

        Ok(results)
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
                    exact_bucket: BlockMetadata::NOT_IN_TREE,
                    exact_offset: StashEntry::<V>::DUMMY_OFFSET,
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

    /// `positions` must be sorted in descending order! (for this function the order could also be ascending, but for
    /// `write_to_paths` it must be descending)
    pub fn read_from_paths<const Z: crate::BucketSize>(
        &mut self,
        physical_memory: &mut [Bucket<V, Z>],
        positions: &[TreeIndex],
    ) -> Result<(), OramError> {
        debug_assert!(positions.is_sorted_by_key(cmp::Reverse));

        // FIXME: Currently the stash does not support increasing the batch size (that is, using a larger batch after
        // using a smaller one) - this in can cause real blocks to get overwritten with the newly fetched path.
        assert!(
            usize::try_from(self.path_size)? * positions.len()
                < self.prefix_last_read_from_physical_memory
        );

        // This is hacky, but deepest_common_ancestor called with an argument equal to 0 will return 0,
        // so we will correctly load from the root in the fist iteration.
        let mut prev_position = 0;
        let mut stash_index = 0;
        for position in positions {
            let deepest_common_ancestor = position.depest_common_ancestor_of_leaves(&prev_position);
            self.read_below_ancestor(
                physical_memory,
                *position,
                deepest_common_ancestor,
                &mut stash_index,
            )?;
            prev_position = *position;
        }

        if stash_index < self.prefix_last_read_from_physical_memory {
            self.entries[stash_index..self.prefix_last_read_from_physical_memory]
                .fill(StashEntry::dummy());
        }
        self.prefix_last_read_from_physical_memory = stash_index;

        Ok(())
    }

    pub fn read_below_ancestor<const Z: crate::BucketSize>(
        &mut self,
        physical_memory: &mut [Bucket<V, Z>],
        mut descendant: TreeIndex,
        ancestor: TreeIndex,
        stash_index: &mut usize,
    ) -> Result<(), OramError> {
        while descendant != ancestor {
            let bucket = physical_memory[usize::try_from(descendant)?];
            for block in bucket.blocks {
                self.entries[*stash_index] = StashEntry {
                    block,
                    exact_bucket: BlockMetadata::NOT_IN_TREE,
                    exact_offset: StashEntry::<V>::DUMMY_OFFSET,
                };
                *stash_index += 1;
            }
            descendant >>= 1;
        }

        Ok(())
    }
}

fn prepare_bucket_counts(positions: &[TreeIndex]) -> Vec<(TreeIndex, u64)> {
    let mut bucket_counts = positions.iter().map(|&pos| (pos, 0)).collect::<Vec<_>>();

    // This works because the positions are sorted
    let mut i = 0;
    while bucket_counts.last().unwrap().0 != 1 {
        let parent = bucket_counts[i].0 >> 1;
        if parent != bucket_counts.last().unwrap().0 {
            bucket_counts.push((parent, 0));
        }
        i += 1;
    }

    bucket_counts
}
