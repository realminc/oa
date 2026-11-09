//! Deterministic Slang lowering of the shared, handle-free fusion SSA.
//! Numerical bodies are embedded from the admitted OA scalar modules.
use crate::runtime::dna::elementwise::{Program, Source};
use crate::{Error, Result};

impl Program {
	/// Straight-line lowering of the same SSA consumed by the tape encoder.
	pub fn slang_source(&self, bounded: bool) -> Result<String> {
		if !self.validate() || self.input_bytes.len() + self.outputs.len() > 31 {
			return Err(Error::invalid_argument(
				"specialization IR exceeds its validated descriptor/push ABI",
			));
		}
		let bindings = self.input_bytes.len() + self.outputs.len();
		let mut code = String::new();
		// Package only these two audited entry-point-free sources in memory.
		for module in [
			include_str!("../../../../slang/core/math/activations.slang"),
			include_str!("../../../../slang/core/math/elementwise.slang"),
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
		let mut linear = std::collections::BTreeSet::new();
		for op in &self.instructions {
			if op.opcode != 4 {
				for source in &op.sources {
					if let Source::Input(slot) = source {
						linear.insert(*slot);
					}
				}
			}
		}
		for slot in linear {
			code.push_str(&format!(
				"\tfloat input{slot} = asfloat(storage_buffers[push.buffer{slot}].Load(index * 4u));\n"
			));
		}
		for (index, op) in self.instructions.iter().enumerate() {
			if op.opcode == 4 {
				let [Source::Input(table), Source::Input(indices)] = op.sources[..] else {
					unreachable!()
				};
				let [rows, width, dtype] = op.parameters[..] else {
					unreachable!()
				};
				let read = if dtype == 0 {
					format!(
						"(storage_buffers[push.buffer{indices}].Load((position{index} & ~3u)) >> ((position{index} & 3u) * 8u)) & 255u"
					)
				} else {
					format!("storage_buffers[push.buffer{indices}].Load(position{index} * 4u)")
				};
				code.push_str(&format!("\tuint position{index} = thread.x / {width}u;\n\tuint row{index} = {read};\n\tfloat v{index} = row{index} < {rows}u ? asfloat(storage_buffers[push.buffer{table}].Load((row{index} * {width}u + thread.x % {width}u) * 4u)) : asfloat(0x7fc00000u);\n"));
			} else {
				let mut args = Vec::new();
				for (operand, source) in op.sources.iter().enumerate() {
					args.push(match source {
						Source::Input(slot) => format!("input{slot}"),
						Source::Value(value) => format!("v{value}"),
						Source::Constant(bits) => format!("asfloat(0x{bits:08x}u)"),
						Source::Broadcast {
							input,
							rank,
							dimensions,
							strides,
						} => {
							let name = format!("mapped{index}_{operand}");
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
					op.opcode,
					args[0],
					args.get(1).cloned().unwrap_or_else(|| "0.0f".into()),
					op.parameters.first().copied().unwrap_or(0)
				);
				code.push_str(&format!("\tfloat v{index} = {expression};\n"));
			}
			for (slot, &value) in self.outputs.iter().enumerate() {
				if value == index {
					code.push_str(&format!(
						"\tstorage_buffers[push.buffer{}].Store(thread.x * 4u, asuint(v{index}));\n",
						self.input_bytes.len() + slot
					));
				}
			}
		}
		code.push_str("}\n");
		Ok(code)
	}
}
