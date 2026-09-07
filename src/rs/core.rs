//! Foundational semantic values and metadata.
//!
//! This is a private source-organization boundary. The crate root explicitly
//! re-exports admitted public values.

mod image;
mod matrix;

pub use image::{Format, Image};
pub use matrix::Matrix;
