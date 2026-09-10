//! Interpretation of image and video values.
//!
//! Pixel transformations and still-image codecs belong to [`crate::image`];
//! video codecs and frame transformations belong to [`crate::video`]. Vision
//! is the direct Rust operation namespace for the donor's `oa::FnDetection`
//! surface; no `FnDetection` type or forwarding facade is introduced.
//! Operations are published here only when they add interpretation semantics
//! and have an operation schema, executable lowering, and correctness oracle.

mod detection;

pub use detection::{
	DetectionMetricsResult, NmsConfig, NmsResult, SegmentationMetricsResult, binary_mask_counts,
	box_iou, confusion_matrix, evaluate, evaluate_segmentation, nms,
};
