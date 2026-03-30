// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is dual-licensed under either the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree or the Apache
// License, Version 2.0 found in the LICENSE-APACHE file in the root directory
// of this source tree. You may select, at your option, one of the above-listed licenses.

//! Utilities.

use rand::{CryptoRng, RngExt};

use std::num::TryFromIntError;

pub(crate) type TreeIndex = u64;
pub(crate) type TreeHeight = u64;

pub(crate) trait CompleteBinaryTreeIndex
where
    Self: Sized,
{
    fn ct_node_on_path(&self, depth: TreeHeight, height: TreeHeight) -> Self;
    fn random_leaf<R: CryptoRng>(
        tree_height: TreeHeight,
        rng: &mut R,
    ) -> Result<Self, TryFromIntError>;
    fn ct_depth(&self) -> TreeHeight;
    fn is_leaf(&self, height: TreeHeight) -> bool;
}

impl CompleteBinaryTreeIndex for TreeIndex {
    // A TreeIndex can have any nonzero value.
    fn ct_node_on_path(&self, depth: TreeHeight, height: TreeHeight) -> Self {
        // We maintain the invariant that all TreeIndex values are nonzero.
        assert_ne!(*self, 0);
        // We only call this method when the receiver is a leaf.
        assert!(self.is_leaf(height));

        let shift = height - depth;
        self >> shift
    }

    fn random_leaf<R: CryptoRng>(
        tree_height: TreeHeight,
        rng: &mut R,
    ) -> Result<Self, TryFromIntError> {
        let tree_height: u32 = tree_height.try_into()?;
        let result = 2u64.pow(tree_height) + rng.random_range(0..2u64.pow(tree_height));
        // The value we've just generated is at least the first summand, which is at least 1.
        assert_ne!(result, 0);
        Ok(result)
    }

    fn ct_depth(&self) -> TreeHeight {
        // We maintain the invariant that all TreeIndex values are nonzero.
        assert_ne!(*self, 0);

        let leading_zeroes: u64 = self.leading_zeros().into();
        let index_bitlength = 64;
        index_bitlength - leading_zeroes - 1
    }

    fn is_leaf(&self, height: TreeHeight) -> bool {
        // We maintain the invariant that all TreeIndex values are nonzero.
        assert_ne!(*self, 0);

        self.ct_depth() == height
    }
}
