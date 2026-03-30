// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! A trait representing a Circuit ORAM stash.

use std::vec;

use crate::{Address, OramBlock, OramError, StashSize, bucket::CircuitOramBlock, utils::TreeIndex};

use subtle::{Choice, ConditionallySelectable};

#[derive(Debug)]
/// A fixed-size, obliviously accessed Circuit ORAM stash data structure implemented using oblivious sorting.
pub struct ObliviousStash<V: OramBlock> {
    pub entries: Vec<StashEntry<V>>,
    pub path_size: StashSize,
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
    pub fn new(path_size: StashSize, overflow_size: StashSize) -> Result<Self, OramError> {
        let num_stash_blocks: usize = (path_size + overflow_size).try_into()?;

        Ok(Self {
            entries: vec![StashEntry::<V>::dummy(); num_stash_blocks],
            path_size,
        })
    }

    pub fn add_block(
        &mut self,
        address: Address,
        position: TreeIndex,
        value: V,
    ) -> Result<(), OramError> {
        let mut added: Choice = 0.into();

        for entry in &mut self.entries {
            let is_dummy = entry.block.ct_is_dummy();

            entry.block.conditional_assign(
                &CircuitOramBlock {
                    value,
                    address,
                    position,
                },
                !added & is_dummy,
            );

            added |= is_dummy;
        }

        assert!(bool::from(added));

        Ok(())
    }

    pub fn conditional_remove_deepest(
        &mut self,
        choice: Choice,
    ) -> Result<CircuitOramBlock<V>, OramError> {
        todo!()
    }

    #[cfg(test)]
    pub fn occupancy(&self) -> StashSize {
        let mut result = 0;
        for i in 0..self.entries.len() {
            if !self.entries[i].block.is_dummy() {
                result += 1;
            }
        }
        result
    }
}
