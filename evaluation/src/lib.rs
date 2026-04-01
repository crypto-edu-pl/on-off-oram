/// Re-export one of the implemented ORAMs depending on enabled features. This is to allow evaluating different ORAMs
pub mod oram_proxy;

/// An oblivious hashset that uses ORAM for the underlying storage
pub mod hashset;

/// A fake "ORAM" that actually just does plain memory accesses (as a baseline for performance comparisions)
pub mod not_really_oram;

/// Helpers for the binaries
pub mod bin_utils;
