use pyo3::prelude::*;

mod activation;
mod attention;
mod batch_norm;
mod byte_model;
mod conv;
mod dropout;
mod embedding;
mod empyrealm;
mod ffn;
mod flow;
mod gru;
mod layer_norm;
mod linear;
mod mamba3;
mod moe;
mod pool;
mod rms_norm;
mod rnn;
mod rope;
mod sequential;
mod swiglu;
mod transformer;
mod upsample;
mod utility;
mod vq;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	linear::register(module)?;
	embedding::register(module)?;
	layer_norm::register(module)?;
	rms_norm::register(module)?;
	dropout::register(module)?;
	activation::register(module)?;
	utility::register(module)?;
	rnn::register(module)?;
	gru::register(module)?;
	swiglu::register(module)?;
	ffn::register(module)?;
	attention::register(module)?;
	transformer::register(module)?;
	mamba3::register(module)?;
	moe::register(module)?;
	sequential::register(module)?;
	empyrealm::register(module)?;
	flow::register(module)?;
	conv::register(module)?;
	pool::register(module)?;
	upsample::register(module)?;
	rope::register(module)?;
	byte_model::register(module)?;
	batch_norm::register(module)?;
	vq::register(module)?;
	Ok(())
}
