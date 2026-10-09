//! ML forward, backward, and optimizer contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "ml/test_training.rs"]
mod training;

#[path = "ml/test_loss.rs"]
mod loss;

#[path = "ml/test_metric.rs"]
mod metric;

#[path = "ml/test_module.rs"]
mod module;

#[path = "ml/test_environment.rs"]
mod environment;

#[path = "ml/test_cart_pole.rs"]
mod cart_pole;
#[path = "ml/test_lunar_lander.rs"]
mod lunar_lander;

#[path = "ml/test_advantage.rs"]
mod advantage;

#[path = "ml/test_rollout.rs"]
mod rollout;

#[path = "ml/test_replay.rs"]
mod replay;

#[path = "ml/test_policy.rs"]
mod policy;

#[path = "ml/test_ppo.rs"]
mod ppo;

#[path = "ml/test_ppo_trainer.rs"]
mod ppo_trainer;

#[path = "ml/test_dqn.rs"]
mod dqn;

#[path = "ml/test_dqn_trainer.rs"]
mod dqn_trainer;

#[path = "ml/test_sac.rs"]
mod sac;

#[path = "ml/test_sac_trainer.rs"]
mod sac_trainer;

#[path = "ml/test_bmm.rs"]
mod bmm;

#[path = "ml/test_actor_critic.rs"]
mod actor_critic;

#[path = "ml/test_nlp.rs"]
mod nlp;

#[path = "ml/test_tokenizer.rs"]
mod tokenizer;

#[path = "ml/test_it_training.rs"]
mod it_training;

#[path = "ml/test_it_rollout_training.rs"]
mod it_rollout_training;

#[path = "ml/test_callbacks.rs"]
mod callbacks;

#[path = "ml/test_schedulers.rs"]
mod schedulers;

#[path = "ml/test_model_file.rs"]
mod model_file;

#[path = "ml/test_optimizer.rs"]
mod optimizer;
#[path = "ml/test_optim_contracts.rs"]
mod optimizer_contracts;

#[path = "ml/nn/mod.rs"]
mod nn;
