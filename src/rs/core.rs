//! Foundational OA values, metadata, and failure contracts.

mod dtype;
mod error;
mod image;
mod matrix;

pub use dtype::{DType, Element};
pub use error::{Error, ErrorKind, Result};
pub use image::{Format, Image};
pub use matrix::Matrix;
