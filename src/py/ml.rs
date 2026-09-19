use pyo3::prelude::*;

mod actor_critic;
mod autograd;
mod checkpoint;
mod checkpoint_fns;
mod environment;
mod nlp;
mod nn;
mod ops;
mod optim;
mod replay;
mod rl;
mod rollout;
mod tokenizer;
mod trainer;
mod trainer_data;
mod training_program;
mod training_schedule;
mod training_session;

pub(crate) use optim::{PythonAdam, PythonAdamW, PythonMuon, PythonSgd};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	ops::register(module)?;
	tokenizer::register(module)?;
	autograd::register(module)?;
	optim::register(module)?;
	checkpoint::register(module)?;
	checkpoint_fns::register(module)?;
	training_schedule::register(module)?;
	training_session::register(module)?;
	nlp::register(module)?;
	trainer_data::register(module)?;
	trainer::register(module)?;
	training_program::register(module)?;
	nn::register(module)?;
	rl::register(module)?;
	environment::register(module)?;
	replay::register(module)?;
	rollout::register(module)?;
	actor_critic::register(module)?;
	Ok(())
}
