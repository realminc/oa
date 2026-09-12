//! Private ML operation lowering, partitioned by semantic owner.

pub(in crate::ml) mod advantage;
pub(in crate::ml) mod attention;
mod common;
pub(in crate::ml) mod conv;
pub(crate) mod environment;
pub(in crate::ml) mod flow;
pub(in crate::ml) mod loss;
pub(in crate::ml) mod matrix;
pub(in crate::ml) mod moe;
pub(in crate::ml) mod optim;
pub(in crate::ml) mod pool;
pub(in crate::ml) mod recurrent;
pub(in crate::ml) mod replay;
pub(in crate::ml) mod rollout;
pub(in crate::ml) mod ssm;
pub(in crate::ml) mod upsample;
pub(in crate::ml) mod vq;
