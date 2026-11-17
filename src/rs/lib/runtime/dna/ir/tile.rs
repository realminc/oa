//! Typed tile-level IR (OA-CC-11), lowered deterministically to Slang.
//!
//! The IR separates a block program's structure — index space, descriptor
//! bindings, and ordered tile ops — from the emitted shader text. This
//! first slice carries the vocabulary the elementwise pointwise template
//! needs; shared-memory staging, barriers, subgroup ops, `dot`, row/column
//! reduce and scan arrive with the templates that consume them.

use crate::runtime::dna::ir::elementwise::{Program, Source};

/// One descriptor slot's role in the template ABI. Slot order is the
/// elementwise convention: inputs then outputs, dense push-constant
/// indices — the artifact ABI is unchanged by the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum Binding {
	Input(usize),
	Output(usize),
}

/// Scalar operand inside a tile op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum Scalar {
	/// A hoisted linear input load (`input{slot}`).
	Loaded(usize),
	/// The SSA value produced by op index `usize`.
	Value(usize),
	/// Exact f32 bit pattern.
	Constant(u32),
	/// Broadcast/gather-mapped read of input `input` through
	/// `dimensions`/`strides` of length `rank`.
	Mapped {
		input: usize,
		rank: u32,
		dimensions: [u32; 8],
		strides: [u32; 8],
	},
}

/// One ordered tile operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum TileOp {
	/// Hoisted linear load of input `slot`, referenced as `Scalar::Loaded`.
	LoadLinear { slot: usize },
	/// Row-table gather: computes `v{value}` from `table`/`indices`.
	GatherRow {
		value: usize,
		table: usize,
		indices: usize,
		rows: u32,
		width: u32,
		dtype: u32,
	},
	/// `eval_element` pointwise instruction producing `v{value}`.
	Pointwise {
		value: usize,
		opcode: u32,
		args: Vec<Scalar>,
		parameter: u32,
	},
	/// Store `v{value}` into output `slot` at the linear index.
	Store { slot: usize, value: usize },
}

/// A lowered block program: one 1-D index space, dense descriptor
/// bindings, and ordered ops. The elementwise template keeps the exact
/// incumbent emission shape — hoisted loads, per-op stores — so the
/// emitted Slang text is bit-identical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct TileProgram {
	pub element_count: u32,
	pub bindings: Vec<Binding>,
	pub ops: Vec<TileOp>,
}

impl TileProgram {
	/// Leading `Binding::Input` entries; outputs occupy the dense tail.
	pub(in crate::runtime) fn input_count(&self) -> usize {
		self
			.bindings
			.iter()
			.position(|binding| matches!(binding, Binding::Output(_)))
			.unwrap_or(self.bindings.len())
	}

	/// Lower the elementwise pointwise template into tile IR. The builder
	/// is the single owner of template emission order; validation stays
	/// with the source `Program`.
	pub(in crate::runtime) fn from_elementwise(program: &Program) -> Self {
		let bindings = (0..program.input_bytes.len())
			.map(Binding::Input)
			.chain((0..program.outputs.len()).map(Binding::Output))
			.collect();
		let mut ops = Vec::new();
		// Hoisted linear loads, in sorted slot order as the incumbent emits.
		let mut linear = std::collections::BTreeSet::new();
		for op in &program.instructions {
			if op.opcode != 4 {
				for source in &op.sources {
					if let Source::Input(slot) = source {
						linear.insert(*slot);
					}
				}
			}
		}
		ops.extend(linear.into_iter().map(|slot| TileOp::LoadLinear { slot }));
		for (index, op) in program.instructions.iter().enumerate() {
			if op.opcode == 4 {
				let [Source::Input(table), Source::Input(indices)] = op.sources[..] else {
					continue;
				};
				let [rows, width, dtype] = op.parameters[..] else {
					continue;
				};
				ops.push(TileOp::GatherRow {
					value: index,
					table,
					indices,
					rows,
					width,
					dtype,
				});
			} else {
				ops.push(TileOp::Pointwise {
					value: index,
					opcode: op.opcode,
					args: op
						.sources
						.iter()
						.map(|source| match source {
							Source::Input(slot) => Scalar::Loaded(*slot),
							Source::Value(value) => Scalar::Value(*value),
							Source::Constant(bits) => Scalar::Constant(*bits),
							Source::Broadcast {
								input,
								rank,
								dimensions,
								strides,
							} => Scalar::Mapped {
								input: *input,
								rank: *rank,
								dimensions: *dimensions,
								strides: *strides,
							},
						})
						.collect(),
					parameter: op.parameters.first().copied().unwrap_or(0),
				});
			}
			for (slot, &value) in program.outputs.iter().enumerate() {
				if value == index {
					ops.push(TileOp::Store { slot, value: index });
				}
			}
		}
		Self {
			element_count: program.element_count,
			bindings,
			ops,
		}
	}
}

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/ir_tile_unit.rs"]
mod tests;
