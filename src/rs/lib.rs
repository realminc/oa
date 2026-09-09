//! oa - GPU-first architecture with explicit Vulkan
//!
//! Experimental Rust port / successor prototype for OA.
//!
//! The target API is organized as a language-like semantic surface:
//! - Semantic value types are re-exported at the root.
//! - Engine and, once admitted, Device and Event are runtime types.
//! - Stateless operations live in matrix, image, audio, video, vision, render,
//!   ml, and crypto modules.
//!
//! Foundational contracts live under `core`; common types are also re-exported
//! from the crate root. Runtime machinery lives under `runtime`.

pub mod runtime;

pub mod audio;
pub mod core;
pub mod crypto;
pub mod matrix;
pub mod ml;
pub mod render;
pub mod video;
pub mod vision;

pub use core::vlm;

// Re-export the common public vocabulary.
pub use audio::{Audio, AudioChannelLayout};
pub use core::{
	DType, Error, ErrorKind, Format, Image, Matrix, OpAttribute, OpAttributeKind, OpAttributeSpec,
	OpControlFlow, OpDTypeRule, OpDifferentiation, OpEffect, OpLowering, OpShapeRule, OpValueKind,
	OperationContract, Result,
};
pub use crypto::{Hash, Hasher, MerkleProof, MerkleTree, Shake128, Shake256};
pub use video::Video;

pub use runtime::{
	CapturedResourceDesc, DeviceSelection, Engine, EngineBuilder, Event, ExecutionPlan,
	ExecutionPlanDiagnostics, LogComponent, LogLevel, LogOptions, SemanticAccessMode,
	SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph, SemanticLoweringAnalysis,
	SemanticOpDesc, SemanticOpId, SemanticStorageBinding, SemanticValueAccess, SemanticValueDesc,
	SemanticValueId,
};
