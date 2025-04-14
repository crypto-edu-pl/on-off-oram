use rand::{CryptoRng, RngCore};

use crate::{Address, Oram, OramBlock, OramError, OramMode};

/// A fake "ORAM" that actually just does plain memory accesses (as a baseline for performance comparisions)
pub struct NotReallyOram<V: OramBlock> {
    physical_memory: Vec<V>,
}

impl<V: OramBlock> NotReallyOram<V> {
    /// Create a new instance
    pub fn new(block_capacity: Address) -> Result<Self, OramError> {
        Ok(Self {
            physical_memory: vec![V::default(); usize::try_from(block_capacity)?],
        })
    }
}

impl<V: OramBlock> Oram for NotReallyOram<V> {
    type V = V;

    fn block_capacity(&self) -> Result<Address, OramError> {
        Ok(self.physical_memory.len().try_into()?)
    }

    fn access<R: rand::RngCore + CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        index: Address,
        callback: F,
        _rng: &mut R,
    ) -> Result<Self::V, OramError> {
        let index = usize::try_from(index)?;
        let result = self.physical_memory[index];
        self.physical_memory[index] = callback(&result);
        Ok(result)
    }

    fn batch_access<R: RngCore + CryptoRng, F: Fn(&Self::V) -> Self::V>(
        &mut self,
        _callbacks: &[(Address, F)],
        _rng: &mut R,
    ) -> Result<Vec<Self::V>, OramError> {
        unimplemented!()
    }

    fn turn_on<R: RngCore + CryptoRng>(&mut self, _rng: &mut R) -> Result<(), OramError> {
        Ok(())
    }

    fn turn_off(&mut self) -> Result<(), OramError> {
        Ok(())
    }

    fn turn_on_without_evicting(&mut self) -> Result<(), OramError> {
        Ok(())
    }

    fn mode(&self) -> OramMode {
        OramMode::Off
    }
}
