//! ML forward, backward, and optimizer contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "ml/test_training.rs"]
mod training;

#[path = "ml/test_module.rs"]
mod module;

#[path = "ml/test_layer_norm.rs"]
mod layer_norm;

#[path = "ml/test_transformer.rs"]
mod transformer;

#[path = "ml/test_nlp.rs"]
mod nlp;

#[path = "ml/test_training_loop.rs"]
mod training_loop;
