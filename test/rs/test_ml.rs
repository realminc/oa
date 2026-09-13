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

#[path = "ml/test_layer_norm.rs"]
mod layer_norm;

#[path = "ml/test_batch_norm.rs"]
mod batch_norm;

#[path = "ml/test_byte.rs"]
mod byte;

#[path = "ml/test_attention.rs"]
mod attention;

#[path = "ml/test_bmm.rs"]
mod bmm;

#[path = "ml/test_moe.rs"]
mod moe;

#[path = "ml/test_conv.rs"]
mod conv;

#[path = "ml/test_rms_norm.rs"]
mod rms_norm;

#[path = "ml/test_ffn.rs"]
mod ffn;

#[path = "ml/test_flow.rs"]
mod flow;

#[path = "ml/test_gru.rs"]
mod gru;

#[path = "ml/test_rnn.rs"]
mod rnn;

#[path = "ml/test_swiglu.rs"]
mod swiglu;

#[path = "ml/test_softmax.rs"]
mod softmax;

#[path = "ml/test_pool.rs"]
mod pool;

#[path = "ml/test_upsample.rs"]
mod upsample;

#[path = "ml/test_rope.rs"]
mod rope;

#[path = "ml/test_transformer.rs"]
mod transformer;

#[path = "ml/test_transformer_module.rs"]
mod transformer_module;

#[path = "ml/test_activation.rs"]
mod activation;

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

#[path = "ml/test_vq.rs"]
mod vq;

#[path = "ml/test_alm.rs"]
mod alm;

#[path = "ml/test_mamba3.rs"]
mod mamba3;
