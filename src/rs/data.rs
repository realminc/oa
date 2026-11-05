//! Checked host columns, OHLCV windows, datasets and device batch collation.
//!
//! Donor: `oa/data/dataset.h`, `oa/data/dsSubset.h`, `oa/data/fnDataset.h`
//!
//! | C++ | Rust |
//! | --- | --- |
//! | trade `OaSeries` / `OaDataFrame` | [`Series`] / [`DataFrame`] |
//! | trade `OaDataOhlcv` | [`OhlcvView`] |
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

mod dataframe;
mod dataset;
mod ohlcv;
#[path = "data/type.gen.rs"]
mod types;

pub use dataframe::{DataFrame, DataFrameView, Series, SeriesView, SeriesViewMut};
pub use ohlcv::{DsOhlcv, OhlcvView, OhlcvWindows, UnixTimeUnit};
pub use types::{Batch, DataLoaderConfig, OhlcvWindowConfig, SplitResult};

pub use dataset::{DataLoader, Dataset, Sample, Subset, collate, random_split, shuffle};

#[path = "data/parity.gen.rs"]
mod parity;
