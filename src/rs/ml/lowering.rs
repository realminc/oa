//! Private ML operation lowering, partitioned by semantic owner.

pub(in crate::ml) mod attention;
mod common;
pub(in crate::ml) mod conv;
pub(in crate::ml) mod loss;
pub(in crate::ml) mod matrix;
pub(in crate::ml) mod moe;
pub(in crate::ml) mod optim;
pub(in crate::ml) mod pool;
pub(in crate::ml) mod recurrent;
pub(in crate::ml) mod upsample;
