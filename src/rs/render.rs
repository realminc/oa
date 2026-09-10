//! Rendering, texture, scene, and presentation contracts.
//!
//! This namespace intentionally exposes no placeholder renderer or presenter.
//! Stateful rendering and presentation types are admitted only with explicit
//! engine ownership, synchronization, failure, and lifecycle behavior.

mod texture;

pub use texture::{Texture, save_texture_file, texture_from_rgba8};
