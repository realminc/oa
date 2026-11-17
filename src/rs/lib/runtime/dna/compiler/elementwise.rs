//! Deterministic Slang lowering of the shared, handle-free fusion SSA.
//! The elementwise pointwise template lowers through the tile IR
//! (OA-CC-11); numerical bodies are embedded from the admitted OA scalar
//! modules.
use crate::runtime::dna::ir::elementwise::Program;
use crate::runtime::dna::ir::tile::TileProgram;
use crate::{Error, Result};

impl Program {
	/// Straight-line lowering of the same SSA consumed by the tape encoder.
	pub fn slang_source(&self, bounded: bool) -> Result<String> {
		if !self.validate() || self.input_bytes.len() + self.outputs.len() > 31 {
			return Err(Error::invalid_argument(
				"specialization IR exceeds its validated descriptor/push ABI",
			));
		}
		TileProgram::from_elementwise(self).slang_source(bounded)
	}
}
