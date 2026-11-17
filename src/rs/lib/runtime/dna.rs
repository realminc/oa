//! Cross-domain compute planning, fusion and compilation.
//!
//! Owns compiler transformations and provider policy, not execution lifetime.
//! Engine owns orchestration; Device owns native resources and publication.
//! Shared semantic/executable representations live under runtime::graph.

pub(super) mod compiler;
pub(super) mod fusion;
pub(super) mod ir;
pub(super) mod kernel;
pub(super) mod provider;
pub(super) mod tuning;

pub(super) use provider::dnn::{DnnPlan, DnnPolicy};
