//! ML loss functions: pointwise, cross-entropy, and RL composite objectives.

mod bce;
mod common;
mod cross_entropy;
mod ctc;
mod dqn;
mod l1;
mod margin;
mod mse;
mod ppo;
mod probabilistic;
mod sac;
mod smooth_l1;

pub use super::types::{
	DqnLossConfig, DqnLossResult, PpoLossConfig, PpoLossResult, SacCriticLossResult, SacLossConfig,
};
pub use bce::bce;
pub use cross_entropy::{cross_entropy, cross_entropy_masked};
pub use ctc::ctc;
pub use dqn::dqn;
pub use l1::l1;
pub use margin::{cosine_embedding, margin_ranking, multi_margin, soft_margin, triplet_margin};
pub use mse::mse;
pub use ppo::{ppo, ppo_clipped_policy, ppo_clipped_policy_backward};
pub use probabilistic::{gaussian_nll, kl_div, multi_label_soft_margin, nll, poisson_nll};
pub use sac::{sac_actor, sac_critic};
pub use smooth_l1::smooth_l1;

pub(in crate::ml) use bce::bce_backward;
pub(in crate::ml) use cross_entropy::{cross_entropy_backward, cross_entropy_masked_backward};
pub(in crate::ml) use ctc::ctc_backward;
pub(in crate::ml) use l1::l1_backward;
pub(in crate::ml) use mse::mse_backward;
pub(in crate::ml) use smooth_l1::smooth_l1_backward;
