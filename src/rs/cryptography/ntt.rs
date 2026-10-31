//! Device-resident cyclic NTT over checked custom prime fields below 2^62.
//!
//! Coefficients are U32 `[batch, length, 2]` limb pairs. Preparation retains
//! Engine-owned twiddle tables; transforms preserve natural coefficient order.
//! These are public-data operations, not a secret execution contract.
mod config;
#[path = "ntt/parity.gen.rs"]
mod parity;
#[path = "ntt/transform.gen.rs"]
mod transform;

pub use config::{NttConfig, NttRepresentation};
pub use transform::{NttParameters, forward, inverse, pointwise_multiply, prepare};
