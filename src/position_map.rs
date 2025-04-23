// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! A recursive Path ORAM position map data structure.

use super::path_oram::PathOram;
use crate::bucket::{BlockMetadata, PositionBlock};
use crate::{linear_time_oram::LinearTimeOram, Address, BlockSize, BucketSize, Oram};
use crate::{OramError, RecursionCutoff};
use crate::{OramMode, StashSize};
use rand::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

#[derive(Copy, Clone)]
pub struct PositionMapUpdate {
    pub address: Address,
    pub metadata: BlockMetadata,
}

impl ConditionallySelectable for PositionMapUpdate {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let address = Address::conditional_select(&a.address, &b.address, choice);
        let metadata = BlockMetadata::conditional_select(&a.metadata, &b.metadata, choice);
        PositionMapUpdate { address, metadata }
    }
}

/// A recursive Path ORAM position map data structure. `AB` is the number of addresses stored in each ORAM block.
#[derive(Debug)]
pub enum PositionMap<const AB: BlockSize, const Z: BucketSize> {
    /// A simple, linear-time `AddressOram`.
    Base(LinearTimeOram<PositionBlock<AB>>),
    /// A recursive `AddressOram` whose position map is also an `AddressOram`.
    Recursive(Box<PathOram<PositionBlock<AB>, Z, AB>>),
}
impl<const AB: BlockSize, const Z: BucketSize> PositionMap<AB, Z> {
    fn address_of_block(address: Address) -> Address {
        let block_address_bits = AB.ilog2();
        address >> block_address_bits
    }

    fn address_within_block(address: Address) -> Result<usize, OramError> {
        let block_address_bits = AB.ilog2();
        let shift: usize = (Address::BITS - block_address_bits).try_into()?;
        Ok(((address << shift) >> shift).try_into()?)
    }
}

impl<const AB: BlockSize, const Z: BucketSize> PositionMap<AB, Z> {
    pub fn write_position_block<R: RngCore + CryptoRng>(
        &mut self,
        address: Address,
        position_block: PositionBlock<AB>,
        rng: &mut R,
    ) -> Result<(), OramError> {
        let address_of_block = PositionMap::<AB, Z>::address_of_block(address);

        match self {
            PositionMap::Base(linear_oram) => {
                linear_oram.write(address_of_block, position_block, rng)?;
            }

            PositionMap::Recursive(block_oram) => {
                block_oram.write(address_of_block, position_block, rng)?;
            }
        }

        Ok(())
    }
}

impl<const AB: BlockSize, const Z: BucketSize> PositionMap<AB, Z> {
    pub fn new<R: CryptoRng + RngCore>(
        number_of_addresses: Address,
        rng: &mut R,
        overflow_size: StashSize,
        recursion_cutoff: RecursionCutoff,
    ) -> Result<Self, OramError> {
        log::info!(
            "PositionMap::new(number_of_addresses = {})",
            number_of_addresses
        );

        if (AB < 2) | (!AB.is_power_of_two()) {
            return Err(OramError::InvalidConfigurationError {
                parameter_name: "Position block size AB".to_string(),
                parameter_value: AB.to_string(),
            });
        }

        let ab_address: Address = AB.try_into()?;
        if number_of_addresses / ab_address <= recursion_cutoff {
            let mut block_capacity = number_of_addresses / ab_address;
            if number_of_addresses % ab_address > 0 {
                block_capacity += 1;
            }
            Ok(Self::Base(LinearTimeOram::new(block_capacity)?))
        } else {
            let block_capacity = number_of_addresses / ab_address;
            let max_batch_size = {
                #[cfg(any(
                    feature = "exact_locations_in_position_map_and_batch_position_map",
                    feature = "batched_turning_on"
                ))]
                {
                    u64::from(block_capacity.ilog2()) * u64::try_from(Z)?
                }

                #[cfg(not(any(
                    feature = "exact_locations_in_position_map_and_batch_position_map",
                    feature = "batched_turning_on"
                )))]
                {
                    1
                }
            };

            Ok(Self::Recursive(Box::new(PathOram::new_with_parameters(
                block_capacity,
                rng,
                overflow_size,
                recursion_cutoff,
                max_batch_size,
            )?)))
        }
    }
}

impl<const AB: BlockSize, const Z: BucketSize> Oram for PositionMap<AB, Z> {
    type V = BlockMetadata;

    fn block_capacity(&self) -> Result<Address, OramError> {
        match self {
            PositionMap::Base(linear_oram) => linear_oram.block_capacity(),
            PositionMap::Recursive(block_oram) => {
                let ab_address: Address = AB.try_into()?;
                Ok(block_oram.block_capacity()? * ab_address)
            }
        }
    }

    fn access<R: RngCore + CryptoRng, F: Fn(&BlockMetadata) -> BlockMetadata>(
        &mut self,
        address: Address,
        callback: F,
        rng: &mut R,
    ) -> Result<BlockMetadata, OramError> {
        let address_of_block = PositionMap::<AB, Z>::address_of_block(address);
        let address_within_block = PositionMap::<AB, Z>::address_within_block(address)?;

        // Not constant time - leaks the mode (which is fine)
        match self.mode() {
            OramMode::On => {
                let block_callback = |block: &PositionBlock<AB>| {
                    let mut result: PositionBlock<AB> = *block;
                    for i in 0..block.data.len() {
                        let index_matches = i.ct_eq(&address_within_block);
                        let position_to_write = callback(&block.data[i]);
                        result.data[i].conditional_assign(&position_to_write, index_matches);
                    }
                    result
                };

                match self {
                    // Base case: index into a linear-time ORAM.
                    PositionMap::Base(linear_oram) => {
                        let block = linear_oram.access(address_of_block, block_callback, rng)?;

                        // XXX: there was no access-whole-block loop for the linear base case. I think this was an error
                        // that leaked information about the block. (That was introduced in
                        // 8a75559dcc1fe5e154d162878d5286c942dc9156)
                        let mut result = BlockMetadata::default();
                        for i in 0..block.data.len() {
                            let index_matches = i.ct_eq(&address_within_block);
                            result.conditional_assign(&block.data[i], index_matches);
                        }

                        Ok(result)
                    }

                    // Recursive case:
                    // (1) split the address into an ORAM address (`address_of_block`) and an offset within the block (`address_within_block`)
                    // (2) Recursively access the block at `address_of_block`, using a callback which updates only the address of interest in that block.
                    // (3) Return the address of interest from the block.
                    PositionMap::Recursive(block_oram) => {
                        let block = block_oram.access(address_of_block, block_callback, rng)?;

                        let mut result = BlockMetadata::default();
                        for i in 0..block.data.len() {
                            let index_matches = i.ct_eq(&address_within_block);
                            result.conditional_assign(&block.data[i], index_matches);
                        }

                        Ok(result)
                    }
                }
            }
            OramMode::Off => {
                let block_callback = |block: &PositionBlock<AB>| {
                    let mut result: PositionBlock<AB> = *block;
                    result.data[address_within_block] = callback(&block.data[address_within_block]);
                    result
                };

                let block = match self {
                    PositionMap::Base(linear_oram) => {
                        linear_oram.access(address_of_block, block_callback, rng)?
                    }
                    PositionMap::Recursive(block_oram) => {
                        block_oram.access(address_of_block, block_callback, rng)?
                    }
                };
                Ok(block.data[address_within_block])
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
                let mut block_callbacks = Vec::with_capacity(callbacks.len());

                for (address, callback) in callbacks {
                    let address_of_block = PositionMap::<AB, Z>::address_of_block(*address);
                    let address_within_block =
                        PositionMap::<AB, Z>::address_within_block(*address)?;

                    let block_callback = move |block: &PositionBlock<AB>| {
                        let mut result: PositionBlock<AB> = *block;
                        for i in 0..block.data.len() {
                            let index_matches = i.ct_eq(&address_within_block);
                            let position_to_write = callback(&block.data[i]);
                            result.data[i].conditional_assign(&position_to_write, index_matches);
                        }
                        result
                    };

                    block_callbacks.push((address_of_block, block_callback));
                }

                let blocks = match self {
                    PositionMap::Base(linear_oram) => {
                        linear_oram.batch_access(&block_callbacks, rng)?
                    }
                    PositionMap::Recursive(block_oram) => {
                        block_oram.batch_access(&block_callbacks, rng)?
                    }
                };

                let mut results = Vec::with_capacity(blocks.len());

                for (block, (address, _)) in blocks.into_iter().zip(callbacks) {
                    let address_within_block =
                        PositionMap::<AB, Z>::address_within_block(*address)?;
                    let mut result = BlockMetadata::default();
                    for i in 0..block.data.len() {
                        let index_matches = i.ct_eq(&address_within_block);
                        result.conditional_assign(&block.data[i], index_matches);
                    }
                    results.push(result);
                }

                Ok(results)
            }
            OramMode::Off => unimplemented!("We do not generate batch accesses in off mode."),
        }
    }

    fn turn_on<R: RngCore + CryptoRng>(&mut self, rng: &mut R) -> Result<(), OramError> {
        match self {
            PositionMap::Base(linear_oram) => linear_oram.turn_on(rng),
            PositionMap::Recursive(block_oram) => block_oram.turn_on(rng),
        }
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        match self {
            PositionMap::Base(linear_oram) => linear_oram.turn_off(),
            PositionMap::Recursive(block_oram) => block_oram.turn_off(),
        }
    }

    fn turn_on_without_evicting(&mut self) -> Result<(), OramError> {
        match self {
            PositionMap::Base(linear_oram) => linear_oram.turn_on_without_evicting(),
            PositionMap::Recursive(block_oram) => block_oram.turn_on_without_evicting(),
        }
    }

    fn mode(&self) -> OramMode {
        match self {
            PositionMap::Base(linear_oram) => linear_oram.mode(),
            PositionMap::Recursive(block_oram) => block_oram.mode(),
        }
    }
}
