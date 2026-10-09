//! Saved parameter-version validation before recording an adjoint.

use super::GradNode;
use crate::Result;

impl GradNode {
	pub(in crate::ml::autograd) fn validate_versions(&self) -> Result<()> {
		match self {
			GradNode::GroupedGemmM {
				weight,
				weight_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
			}
			GradNode::GroupedLinearM {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
				if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
					bias.validate_version(*bias_version)?;
				}
			}
			GradNode::Linear {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				weight.validate_version(*weight_version)?;
				if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
					bias.validate_version(*bias_version)?;
				}
			}
			GradNode::Conv2d {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
				if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
					bias.validate_version(*bias_version)?;
				}
			}
			GradNode::Conv1d {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
				if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
					bias.validate_version(*bias_version)?;
				}
			}
			GradNode::ConvTranspose1d {
				weight,
				weight_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
			}
			GradNode::ConvTranspose2d {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				if let (Some(weight), Some(weight_version)) = (weight, weight_version) {
					weight.validate_version(*weight_version)?;
				}
				if let (Some(bias), Some(bias_version)) = (bias, bias_version) {
					bias.validate_version(*bias_version)?;
				}
			}
			GradNode::Embedding {
				weight,
				weight_version,
				..
			} => weight.validate_version(*weight_version)?,
			GradNode::LayerNorm {
				weight,
				weight_version,
				bias,
				bias_version,
				..
			} => {
				weight.validate_version(*weight_version)?;
				bias.validate_version(*bias_version)?;
			}
			GradNode::BatchNorm2d {
				weight: Some(weight),
				weight_version: Some(weight_version),
				bias: Some(bias),
				bias_version: Some(bias_version),
				..
			} => {
				weight.validate_version(*weight_version)?;
				bias.validate_version(*bias_version)?;
			}
			GradNode::RmsNorm {
				weight: Some(weight),
				weight_version: Some(weight_version),
				..
			} => weight.validate_version(*weight_version)?,
			GradNode::RmsNormGated(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::ChannelNorm(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::GruCell(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::GruScan(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::RnnCell(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::RnnScan(node) => {
				if let (Some(weight), Some(version)) = (&node.weight, node.weight_version) {
					weight.validate_version(version)?;
				}
				if let (Some(bias), Some(version)) = (&node.bias, node.bias_version) {
					bias.validate_version(version)?;
				}
			}
			GradNode::Matrix(_)
			| GradNode::Operation(_)
			| GradNode::FlowLinearState { .. }
			| GradNode::FlowLinearVelocity { .. }
			| GradNode::FlowEulerStep { .. }
			| GradNode::FlowMaskedMse { .. }
			| GradNode::RmsNorm { .. }
			| GradNode::BatchNorm2d { .. }
			| GradNode::Bmm { .. }
			| GradNode::BmmNt { .. }
			| GradNode::BmmTn { .. }
			| GradNode::SplitHeads { .. }
			| GradNode::MergeHeads { .. }
			| GradNode::SoftmaxScaledMasked { .. }
			| GradNode::ScaledDotProductAttention { .. }
			| GradNode::FlashAttention { .. }
			| GradNode::MoeRouteWeights { .. }
			| GradNode::MoeGather { .. }
			| GradNode::MoeCombine { .. }
			| GradNode::Mamba3Preprocess(_)
			| GradNode::Mamba3Siso(_)
			| GradNode::Mamba3Mimo(_) => {}
		}
		Ok(())
	}
}
