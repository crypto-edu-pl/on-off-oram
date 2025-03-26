// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! A simple linear-time implementation of Oblivious RAM.

use crate::{Address, Oram, OramBlock, OramError, OramMode};
use rand::{CryptoRng, RngCore};
use subtle::{ConstantTimeEq, ConstantTimeLess};

/// A simple ORAM that, for each access, ensures obliviousness by making a complete pass over the database,
/// reading and writing each memory location.
#[derive(Debug)]
pub struct LinearTimeOram<V: OramBlock> {
    /// The memory of the ORAM (public for benchmarking).
    pub physical_memory: Vec<V>,
    /// Current mode.
    mode: OramMode,
}

impl<V: OramBlock> LinearTimeOram<V> {
    /// Returns a new `LinearTimeOram` mapping addresses `0 <= address < block_capacity` to default `V` values.
    pub fn new(block_capacity: Address) -> Result<Self, OramError> {
        log::info!("LinearTimeOram::new(capacity = {})", block_capacity,);

        let mut physical_memory = Vec::new();
        physical_memory.resize(usize::try_from(block_capacity)?, V::default());
        Ok(Self {
            physical_memory,
            mode: OramMode::On,
        })
    }
}

impl<V: OramBlock> Oram for LinearTimeOram<V> {
    type V = V;

    fn access<R: RngCore + CryptoRng, F: Fn(&V) -> V>(
        &mut self,
        index: Address,
        callback: F,
        _: &mut R,
    ) -> Result<V, OramError> {
        let index_in_bounds: bool = index.ct_lt(&self.block_capacity()?).into();

        // This operation is not constant-time, but only leaks whether the ORAM index is well-formed or not.
        if !index_in_bounds {
            return Err(OramError::AddressOutOfBoundsError {
                attempted: index,
                capacity: self.block_capacity()?,
            });
        }

        match self.mode() {
            OramMode::On => {
                // This is a dummy value which will always be overwritten.
                let mut result = V::default();

                for i in 0..self.physical_memory.len() {
                    let entry = &self.physical_memory[i];

                    let is_requested_index = (u64::try_from(i)?).ct_eq(&index);

                    result.conditional_assign(entry, is_requested_index);

                    let potential_new_value = callback(entry);

                    self.physical_memory[i]
                        .conditional_assign(&potential_new_value, is_requested_index);
                }
                Ok(result)
            }
            OramMode::Off => {
                let index_usize = usize::try_from(index)?;

                let result = self.physical_memory[index_usize];

                self.physical_memory[index_usize] = callback(&result);

                Ok(result)
            }
        }
    }

    fn block_capacity(&self) -> Result<Address, OramError> {
        Ok(u64::try_from(self.physical_memory.len())?)
    }

    fn turn_on(&mut self) -> Result<(), OramError> {
        // No need to do anything - in on mode we always go through the entire memory, so it doesn't matter if the server
        // knows where some blocks are.
        self.mode = OramMode::On;
        Ok(())
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        self.mode = OramMode::Off;
        Ok(())
    }

    fn mode(&self) -> OramMode {
        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bucket::BlockValue, test_utils::*};

    #[test]
    fn linear_time_oram_correctness_random_workload() {
        let mut oram = LinearTimeOram::<BlockValue<1>>::new(64).unwrap();
        random_workload(&mut oram, 1000);
    }

    #[test]
    fn linear_time_oram_correctness_linear_workload() {
        let mut oram = LinearTimeOram::<BlockValue<1>>::new(64).unwrap();
        linear_workload(&mut oram, 1000);
    }
}
