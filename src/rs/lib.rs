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
//! There is no public `core` namespace. Runtime machinery lives under `runtime/`.

mod core;

pub mod error;
pub mod runtime;

pub mod audio;
pub mod crypto;
pub mod image;
pub mod matrix;
pub mod ml;
pub mod render;
pub mod video;
pub mod vision;

// Re-export public types
pub use error::{Error, Result};

pub use audio::Audio;
pub use core::{Format, Image, Matrix};
pub use video::Video;

pub use runtime::Engine;
