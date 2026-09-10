//! Foundational OA values, metadata, and failure contracts.

pub(crate) mod autograd;
mod dtype;
mod error;
mod image;
mod json;
mod matrix;
pub mod memory;
mod op;
pub mod operation;
pub mod vlm;

pub use dtype::{DType, Element};
pub use error::{Error, ErrorKind, Result};
pub use image::{Image, ImageFormat, ImageLayout};
pub(crate) use json::{push_format, push_json_string};
pub use matrix::Matrix;
pub(crate) use matrix::MatrixSemantic;
pub use op::{
	OpAttribute, OpAttributeKind, OpAttributeSpec, OpControlFlow, OpDTypeRule, OpDifferentiation,
	OpEffect, OpLowering, OpShapeRule, OpValueKind, OperationContract,
};
