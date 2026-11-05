//! Cross-domain operation admission and provider selection.
//!
//! Planned OaDna home: providers are grouped by domain below `provider/` as
//! admission is proven. The `oaDna` planner document owns the contract and
//! provider snapshot.

pub(super) mod kernel;
pub(super) mod provider;

pub(super) use provider::dnn::{DnnPlan, DnnPolicy};
