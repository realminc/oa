//! Private autograd attachments for ML-owned Matrix operation families.

mod attention;
mod conv;
mod embedding;
mod linear;
mod moe;
mod norm;
mod pool;
mod position;
mod recurrent;
mod upsample;

include!("matrix/activation.gen.rs");

pub(in crate::ml) use attention::{
	record_bmm, record_bmm_nt, record_bmm_tn, record_flash_attention, record_merge_heads,
	record_scaled_dot_product_attention, record_softmax_scaled_masked, record_split_heads,
};
pub(in crate::ml) use conv::{
	record_conv_1d, record_conv_2d, record_conv_transpose_1d, record_conv_transpose_2d,
};
pub(in crate::ml) use embedding::record_embedding;
pub(in crate::ml) use linear::record_linear;
pub(in crate::ml) use moe::record_moe_route_weights;
pub(in crate::ml) use norm::{record_batch_norm_2d, record_layer_norm, record_rms_norm};
pub(in crate::ml) use pool::{record_adaptive_avg_pool_2d, record_avg_pool_2d, record_max_pool_2d};
pub(in crate::ml) use position::record_rope;
pub(in crate::ml) use recurrent::{
	record_gru_cell, record_gru_scan, record_rnn_cell, record_rnn_scan,
};
pub(in crate::ml) use upsample::record_upsample_2d;
