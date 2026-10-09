//! Dataset values, batch iteration, and stateless dataset operations.
//!
//! Donor: `oa/data/dataset.h`, `oa/data/dsSubset.h`, `oa/data/fnDataset.h`
//!
//! | C++ | Rust |
//! | --- | --- |
//! | `oa::Dataset` | [`Dataset`] |
//! | `oa::Dataset::Sample` | [`Sample`] |
//! | `oa::DataLoader` | [`DataLoader`] |
//! | `oa::DataLoaderConfig` | [`DataLoaderConfig`] |
//! | `oa::DataLoader::Batch` | [`Batch`] |
//! | `oa::DsSubset` | [`Dataset::subset`] → [`Subset`] |
//! | `oa::FnDataset::SplitResult` | [`SplitResult`] |
//! | `oa::FnDataset::shuffle` | [`shuffle`] |
//! | `oa::FnDataset::randomSplit` | [`random_split`] |
//! | `oa::FnDataset::collate` | [`collate`] |

mod dataset;

pub use dataset::{
	Batch, DataLoader, DataLoaderConfig, Dataset, Sample, SplitResult, Subset, collate, random_split,
	shuffle,
};
