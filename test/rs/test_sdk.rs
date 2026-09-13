//! SDK workload and dataset contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "sdk/data/test_human_ml3d.rs"]
mod human_ml3d;

#[path = "sdk/ml/test_alm_training.rs"]
mod alm_training;
