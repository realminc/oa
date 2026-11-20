//! SDK workload and dataset contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "sdk/data/test_human_ml3d.rs"]
mod human_ml3d;

#[path = "sdk/ml/test_alm_training.rs"]
mod alm_training;

#[path = "sdk/ml/test_cart_pole.rs"]
mod cart_pole;

#[path = "sdk/ml/test_fp16_training.rs"]
mod fp16_training;

#[path = "sdk/ml/test_lunar_lander.rs"]
mod lunar_lander;

#[path = "sdk/ml/test_nlp.rs"]
mod nlp;

#[path = "sdk/ml/nn/test_alm.rs"]
mod alm;

#[path = "sdk/runtime/test_execution_plan.rs"]
mod execution_plan;
