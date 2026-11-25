//! Shared semantic and executable graph representations.
//! DNA owns transformations; Runtime owns their recording and replay lifecycle.

pub(in crate::runtime) mod executable;
pub(in crate::runtime) mod semantic;
