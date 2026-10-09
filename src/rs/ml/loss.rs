//! ML loss functions: pointwise, cross-entropy, and RL composite objectives.

mod bce;
mod common;
mod cross_entropy;
mod dqn;
mod l1;
mod mse;
mod ppo;
mod sac;
mod smooth_l1;

pub use bce::bce;
pub use cross_entropy::{cross_entropy, cross_entropy_masked};
pub use dqn::{DqnLossConfig, DqnLossResult, dqn};
pub use l1::l1;
pub use mse::mse;
pub use ppo::{PpoLossConfig, PpoLossResult, ppo, ppo_clipped_policy, ppo_clipped_policy_backward};
pub use sac::{SacCriticLossResult, SacLossConfig, sac_actor, sac_critic};
pub use smooth_l1::smooth_l1;

pub(in crate::ml) use bce::bce_backward;
pub(in crate::ml) use cross_entropy::{cross_entropy_backward, cross_entropy_masked_backward};
pub(in crate::ml) use l1::l1_backward;
pub(in crate::ml) use mse::mse_backward;
pub(in crate::ml) use smooth_l1::smooth_l1_backward;
