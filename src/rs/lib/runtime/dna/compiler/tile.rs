//! Deterministic Slang lowering of the tile IR (OA-CC-11).
//!
//! The elementwise pointwise template emits the incumbent text verbatim so
//! the IR swap is bit-identical by construction; later templates extend
//! this emitter with shared-memory, subgroup and `dot` ops.

use crate::runtime::dna::ir::tile::{Scalar, TileOp, TileProgram};
use crate::{Error, Result};

impl TileProgram {
	/// Straight-line lowering of the pointwise template consumed by the
	/// tape encoder's specialized route.
	pub(in crate::runtime) fn slang_source(&self, bounded: bool) -> Result<String> {
		if self.element_count == 0 || self.bindings.len() > 31 {
			return Err(Error::invalid_argument(
				"specialization IR exceeds its validated descriptor/push ABI",
			));
		}
		let bindings = self.bindings.len();
		let mut code = String::new();
		for module in [
			include_str!("../../../../../slang/core/math/activations.slang"),
			include_str!("../../../../../slang/core/math/elementwise.slang"),
		] {
			for line in module.lines() {
				if line == "module activations;"
					|| line == "module elementwise;"
					|| line == "import activations;"
				{
					continue;
				}
				code.push_str(line);
				code.push('\n');
			}
		}
		code.push_str("struct PushConstants {\n");
		for slot in 0..bindings {
			code.push_str(&format!("\tuint buffer{slot};\n"));
		}
		code.push_str("\tuint count;\n};\n[[vk::push_constant]] PushConstants push;\n");
		code.push_str(&format!(
			"[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{}];\n",
			if bounded {
				bindings.to_string()
			} else {
				String::new()
			}
		));
		code.push_str("[shader(\"compute\")]\n[numthreads(256, 1, 1)]\nvoid main(uint3 thread : SV_DispatchThreadID) {\n\tuint index = thread.x;\n\tif (index >= push.count) return;\n");
		for op in &self.ops {
			match op {
				TileOp::LoadLinear { slot } => {
					code.push_str(&format!(
						"\tfloat input{slot} = asfloat(storage_buffers[push.buffer{slot}].Load(index * 4u));\n"
					));
				}
				TileOp::GatherRow {
					value,
					table,
					indices,
					rows,
					width,
					dtype,
				} => {
					let read = if *dtype == 0 {
						format!(
							"(storage_buffers[push.buffer{indices}].Load((position{value} & ~3u)) >> ((position{value} & 3u) * 8u)) & 255u"
						)
					} else {
						format!("storage_buffers[push.buffer{indices}].Load(position{value} * 4u)")
					};
					code.push_str(&format!("\tuint position{value} = thread.x / {width}u;\n\tuint row{value} = {read};\n\tfloat v{value} = row{value} < {rows}u ? asfloat(storage_buffers[push.buffer{table}].Load((row{value} * {width}u + thread.x % {width}u) * 4u)) : asfloat(0x7fc00000u);\n"));
				}
				TileOp::Pointwise {
					value,
					opcode,
					args,
					parameter,
				} => {
					let mut rendered = Vec::new();
					for (operand, scalar) in args.iter().enumerate() {
						rendered.push(match scalar {
							Scalar::Loaded(slot) => format!("input{slot}"),
							Scalar::Value(source) => format!("v{source}"),
							Scalar::Constant(bits) => format!("asfloat(0x{bits:08x}u)"),
							Scalar::Mapped {
								input,
								rank,
								dimensions,
								strides,
							} => {
								let name = format!("mapped{value}_{operand}");
								code.push_str(&format!("\tuint {name} = 0u;\n"));
								let mut divisor = 1u32;
								for axis in (0..*rank as usize).rev() {
									code.push_str(&format!(
										"\t{name} += ((thread.x / {divisor}u) % {}u) * {}u;\n",
										dimensions[axis], strides[axis]
									));
									divisor *= dimensions[axis];
								}
								format!("asfloat(storage_buffers[push.buffer{input}].Load({name} * 4u))")
							}
						});
					}
					let expression = format!(
						"eval_element({}u, {}, {}, asfloat(0x{:08x}u))",
						opcode,
						rendered[0],
						rendered.get(1).cloned().unwrap_or_else(|| "0.0f".into()),
						parameter
					);
					code.push_str(&format!("\tfloat v{value} = {expression};\n"));
				}
				TileOp::Store { slot, value } => {
					code.push_str(&format!(
						"\tstorage_buffers[push.buffer{}].Store(thread.x * 4u, asuint(v{value}));\n",
						self.input_count() + slot
					));
				}
			}
		}
		code.push_str("}\n");
		Ok(code)
	}
}
