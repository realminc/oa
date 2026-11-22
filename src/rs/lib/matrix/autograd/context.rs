//! Private borrowed services supplied by the active tape.

use crate::{Matrix, Result};

/// Borrowed tape services; no recording scope, parameter policy or execution owner.
pub(crate) trait GradientContext {
	fn take_output(&mut self, output_id: u64) -> Option<Matrix>;
	fn accumulate(&mut self, value_id: u64, gradient: Matrix) -> Result<()>;
}
