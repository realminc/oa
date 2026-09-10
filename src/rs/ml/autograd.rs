//! Reverse-mode differentiation lifecycle and private operation attachments.
//!
//! [`GradientTape`] is the only public autograd owner. Concrete nodes and
//! generated attachment policy remain private implementation details paired
//! with their owning operation families.

mod loss;
mod matrix;
mod node;
mod tape;

pub use tape::GradientTape;

pub(in crate::ml) use loss::{
	record_bce, record_cross_entropy, record_l1, record_masked_cross_entropy, record_mse,
	record_smooth_l1,
};
pub(in crate::ml) use matrix::{
	record_adaptive_avg_pool_2d, record_avg_pool_2d, record_batch_norm_2d, record_bmm,
	record_bmm_nt, record_bmm_tn, record_conv_1d, record_conv_2d, record_conv_transpose_1d,
	record_conv_transpose_2d, record_elu, record_embedding, record_flash_attention, record_gelu,
	record_gru_cell, record_gru_scan, record_layer_norm, record_leaky_relu, record_linear,
	record_max_pool_2d, record_merge_heads, record_mish, record_moe_route_weights, record_relu,
	record_rms_norm, record_rnn_cell, record_rnn_scan, record_rope,
	record_scaled_dot_product_attention, record_sigmoid, record_silu, record_softmax_scaled_masked,
	record_softplus, record_split_heads, record_swiglu, record_tanh, record_upsample_2d,
};
