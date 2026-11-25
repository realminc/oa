//! Deterministic Slang lowering of the tile IR (OA-CC-11).
//!
//! The elementwise pointwise template emits the incumbent text verbatim so
//! the IR swap is bit-identical by construction; later templates extend
//! this emitter with shared-memory, subgroup and `dot` ops.

use crate::runtime::dna::ir::tile::{
	BiasAct, BlockProgram, Extent, LaneExpr, LaneReduce, PlaneScale, Scalar, TileDecl, TileOp,
	TileProgram,
};
use crate::{Error, Result};

/// Implicit scalar type of a `LaneExpr` operand.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LaneType {
	U32,
	F32,
	Bool,
}

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
				TileOp::Loop { .. }
				| TileOp::ClearTile { .. }
				| TileOp::LoadShared { .. }
				| TileOp::Barrier
				| TileOp::DotShared { .. }
				| TileOp::DotSharedParam { .. }
				| TileOp::StoreTile { .. }
				| TileOp::StoreTileBiasParam { .. }
				| TileOp::StoreTileBias { .. }
				| TileOp::StoreTileAct { .. }
				| TileOp::StoreTileResidual { .. }
				| TileOp::StoreTileResidualAct { .. }
				| TileOp::StoreTileGated { .. }
				| TileOp::StoreTileGateUp { .. }
				| TileOp::LoadSharedConcat { .. }
				| TileOp::StoreTileResidualNorm { .. }
				| TileOp::LaneLoopRange { .. }
				| TileOp::Guard { .. }
				| TileOp::LaneLoop { .. }
				| TileOp::LaneFold { .. }
				| TileOp::LaneFoldDual { .. }
				| TileOp::AssignScalar { .. }
				| TileOp::StoreShared { .. }
				| TileOp::TreeReduce { .. }
				| TileOp::StoreScalar { .. }
				| TileOp::LaneGuard { .. }
				| TileOp::LaneIf { .. }
				| TileOp::GuardIf { .. }
				| TileOp::Return
				| TileOp::ExpertScan { .. }
				| TileOp::BoundaryGateUp { .. }
				| TileOp::BoundaryLinear { .. }
				| TileOp::StoreSharedU32 { .. }
				| TileOp::AssignScalarU32 { .. }
				| TileOp::StoreScalarU32 { .. }
				| TileOp::TreeReduceArgmax { .. }
				| TileOp::LoadSharedPrologue { .. }
				| TileOp::LoadSharedNorm { .. }
				| TileOp::DotSharedWide { .. }
				| TileOp::StoreTileSegment { .. } => {
					return Err(Error::invalid_argument(
						"block op inside an elementwise tile program",
					));
				}
			}
		}
		code.push_str("}\n");
		Ok(code)
	}
}

impl BlockProgram {
	/// Deterministic lowering of a block tile program to compute Slang.
	/// All bounds masking, staging layout and barrier placement are
	/// spelled by the IR; this emitter renders text only.
	pub(in crate::runtime) fn slang_source(&self, bounded: bool) -> Result<String> {
		if self.bindings.is_empty() || self.bindings.len() > 31 {
			return Err(Error::invalid_argument(
				"block program exceeds its validated descriptor ABI",
			));
		}
		let mut code = String::from("struct PushConstants {\n");
		for slot in 0..self.bindings.len() {
			code.push_str(&format!("\tuint buffer{slot};\n"));
		}
		for scalar in &self.push_scalars {
			code.push_str(&format!("\tuint {scalar};\n"));
		}
		for scalar in &self.push_f32_scalars {
			code.push_str(&format!("\tfloat {scalar};\n"));
		}
		code.push_str("};\n[[vk::push_constant]] PushConstants push;\n");
		code.push_str(&format!(
			"[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{}];\n",
			if bounded {
				self.bindings.len().to_string()
			} else {
				String::new()
			}
		));
		for (index, tile) in self.tiles.iter().enumerate() {
			match tile {
				TileDecl::Shared { rows, stride, .. } => code.push_str(&format!(
					"groupshared float shared{index}[{rows}u * {stride}u];\n"
				)),
				TileDecl::SharedU32 { elements } => {
					code.push_str(&format!("groupshared uint shared{index}[{elements}u];\n"))
				}
				_ => {}
			}
		}
		if let Some((kind, alpha)) = self.act_epilogue(&self.ops)? {
			code.push_str(&format!(
				"float tile_sigmoid(float value) {{\n\treturn 1.0f / (1.0f + exp(-value));\n}}\n				 float apply_act(float value) {{\n\tswitch (push.{kind}) {{\n				 \tcase 0u:\n\t\treturn max(value, 0.0f);\n				 \tcase 1u:\n\t\treturn 0.5f * value * (1.0f + tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));\n				 \tcase 2u:\n\t\treturn value * tile_sigmoid(value);\n				 \tcase 3u:\n\t\treturn tanh(value);\n				 \tcase 4u:\n\t\treturn tile_sigmoid(value);\n				 \tcase 5u:\n\t\treturn value * tanh(log(1.0f + exp(value)));\n				 \tcase 6u:\n\t\treturn value > 0.0f ? value : push.{alpha} * (exp(value) - 1.0f);\n				 \tcase 7u:\n\t\treturn max(push.{alpha} * value, value);\n				 \tdefault:\n\t\treturn max(value, 0.0f) + log(1.0f + exp(-abs(value)));\n				 \t}}\n}}\n"
			));
		}
		if self.gated_epilogue(&self.ops)? {
			if self.act_epilogue(&self.ops)?.is_none() {
				code.push_str(
					"float tile_sigmoid(float value) {\n\treturn 1.0f / (1.0f + exp(-value));\n}\n",
				);
			}
			code.push_str(
				"float tile_swiglu(float gate, float up) {\n\treturn (gate * tile_sigmoid(gate)) * up;\n}\n",
			);
		}
		if let Some((upstream, saved)) = self.prologue_epilogue(&self.ops)? {
			code.push_str(&format!(
				"float tile_sigmoid(float value) {{\n\treturn 1.0f / (1.0f + exp(-value));\n}}\n\
				 float tile_silu(float value) {{\n\treturn value * tile_sigmoid(value);\n}}\n\
				 float tile_silu_grad(float value) {{\n\tfloat s = tile_sigmoid(value);\n\treturn s * (1.0f + value * (1.0f - s));\n}}\n\
				 float tile_apply_prologue(float saved, float upstream) {{\n\tswitch (push.prologue_kind) {{\n\
				 \tcase 0u:\n\t\treturn saved > 0.0f ? upstream : 0.0f;\n\
				 \tcase 1u: {{\n\t\tfloat x2 = saved * saved;\n\t\tfloat inner = 0.7978845608028654f * (saved + 0.044715f * x2 * saved);\n\t\tfloat t = tanh(inner);\n\
				 \t\treturn upstream * (0.5f * (1.0f + t)\n\t\t\t+ 0.5f * saved * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 0.134145f * x2));\n\t\t}}\n\
				 \tcase 2u:\n\t\treturn upstream * tile_silu_grad(saved);\n\
				 \tcase 3u:\n\t\treturn upstream * (1.0f - saved * saved);\n\
				 \tcase 4u:\n\t\treturn upstream * saved * (1.0f - saved);\n\
				 \tcase 5u: {{\n\t\tfloat softplus = saved > 20.0f ? saved : log(1.0f + exp(saved));\n\t\tfloat tanh_softplus = tanh(softplus);\n\
				 \t\treturn upstream * (tanh_softplus\n\t\t\t+ saved * tile_sigmoid(saved) * (1.0f - tanh_softplus * tanh_softplus));\n\t\t}}\n\
				 \tcase 6u:\n\t\treturn upstream * (saved > 0.0f ? 1.0f : saved + push.alpha);\n\
				 \tcase 7u:\n\t\treturn saved > 0.0f ? upstream : upstream * push.alpha;\n\
				 \tcase 8u:\n\t\treturn upstream * (1.0f - exp(-saved));\n\
				 \tcase 10u:\n\t\treturn upstream + saved;\n\
				 \tcase 11u:\n\t\treturn upstream - saved;\n\
				 \tdefault:\n\t\treturn upstream * saved;\n\t}}\n}}\n\
				 float tile_prologue_value(uint row, uint feature) {{\n\
				 \tuint element = row * push.output_features + feature;\n\
				 \tif (push.prologue_kind == 9u) {{\n\
				 \t\tuint hidden = push.output_features / 2u;\n\
				 \t\tuint column = feature % hidden;\n\
				 \t\tuint reduced = row * hidden + column;\n\
				 \t\tfloat gate = asfloat(storage_buffers[push.buffer{saved}].Load((row * push.output_features + column) * 4u));\n\
				 \t\tfloat upstream = asfloat(storage_buffers[push.buffer{upstream}].Load(reduced * 4u));\n\
				 \t\tif (feature < hidden) {{\n\
				 \t\t\tfloat up = asfloat(storage_buffers[push.buffer{saved}].Load((row * push.output_features + hidden + column) * 4u));\n\
				 \t\t\treturn upstream * up * tile_silu_grad(gate);\n\t\t}}\n\
				 \t\treturn upstream * tile_silu(gate);\n\t}}\n\
				 \tfloat upstream = asfloat(storage_buffers[push.buffer{upstream}].Load(element * 4u));\n\
				 \tfloat saved = push.prologue_kind == 13u\n\
				 \t\t? asfloat(storage_buffers[push.buffer{saved}].Load(0u))\n\
				 \t\t: asfloat(storage_buffers[push.buffer{saved}].Load(element * 4u));\n\
				 \treturn tile_apply_prologue(saved, upstream);\n}}\n"
			));
		}
		if self.needs_video_helpers() {
			for module in [
				include_str!("../../../../../slang/core/math/storage.slang"),
				include_str!("../../../../../slang/video/to_matrix_common.slang"),
				include_str!("../../../../../slang/video/yuv_to_rgb.slang"),
			] {
				for line in module.lines() {
					if line.starts_with("module ") || line.starts_with("import ") {
						continue;
					}
					code.push_str(line.strip_prefix("export ").unwrap_or(line));
					code.push('\n');
				}
			}
		}
		if self.needs_image_helpers() {
			for module in [
				include_str!("../../../../../slang/core/math/storage.slang"),
				include_str!("../../../../../slang/image/image_gather.slang"),
			] {
				for line in module.lines() {
					if line.starts_with("module ") || line.starts_with("import ") {
						continue;
					}
					code.push_str(line.strip_prefix("export ").unwrap_or(line));
					code.push('\n');
				}
			}
		}
		code.push_str(&format!(
			"[shader(\"compute\")]\n[numthreads({}, 1, 1)]\nvoid main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {{\n",
			self.shape.lanes
		));
		code.push_str(&format!(
			"\tuint lane = group_thread_id.x;\n\tuint thread_row = lane / {}u;\n\tuint thread_column = lane % {}u;\n",
			self.shape.thread_columns, self.shape.thread_columns
		));
		let mut declared = std::collections::BTreeSet::new();
		let mut body = String::new();
		for op in &self.ops {
			self.emit_op(&mut body, op, 1, None, &mut declared)?;
		}
		let mut hoisted = String::with_capacity(body.len());
		let mut rest = body.as_str();
		let mut buffers = std::collections::BTreeSet::new();
		while let Some(at) = rest.find("push.buffer") {
			hoisted.push_str(&rest[..at]);
			let tail = &rest[at + "push.buffer".len()..];
			let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
			if digits.is_empty() {
				hoisted.push_str("push.buffer");
				rest = tail;
			} else {
				if buffers.insert(digits.clone()) {
					code.push_str(&format!("\tuint buf{digits} = push.buffer{digits};\n"));
				}
				hoisted.push_str("buf");
				hoisted.push_str(&digits);
				rest = &tail[digits.len()..];
			}
		}
		hoisted.push_str(rest);
		code.push_str(&hoisted);
		code.push_str("}\n");
		Ok(code)
	}

	fn extent(&self, extent: Extent, induction: Option<usize>) -> Result<String> {
		Ok(match extent {
			Extent::Push(index) => format!(
				"push.{}",
				self.push_scalars.get(index).ok_or_else(|| {
					Error::invalid_argument("block extent references a missing push scalar")
				})?
			),
			Extent::GroupX(scale) => format!("group_id.x * {scale}u"),
			Extent::GroupY(scale) => format!("group_id.y * {scale}u"),
			Extent::Induction => format!(
				"loop{}",
				induction.ok_or_else(|| {
					Error::invalid_argument("induction extent used outside a tile loop")
				})?
			),
			Extent::HalfPush(index) => format!(
				"(push.{} / 2u)",
				self.push_scalars.get(index).ok_or_else(|| {
					Error::invalid_argument("block extent references a missing push scalar")
				})?
			),
			Extent::Expr(index) => self.lane_u32(index, induction)?,
		})
	}

	fn emit_op(
		&self,
		code: &mut String,
		op: &TileOp,
		depth: usize,
		induction: Option<usize>,
		declared: &mut std::collections::BTreeSet<usize>,
	) -> Result<()> {
		let pad = "\t".repeat(depth);
		match op {
			TileOp::Loop {
				extent,
				step,
				start,
				body,
			} => {
				let bound = self.extent(*extent, induction)?;
				let start = match start {
					Some(start) => self.extent(*start, induction)?,
					None => String::from("0u"),
				};
				let loop_name = format!("loop{}", depth + 1);
				let bound_name = format!("bound{}", depth + 1);
				code.push_str(&format!(
					"{pad}for (uint {loop_name} = {start}, {bound_name} = {bound}; {loop_name} < {bound_name}; {loop_name} += {step}u) {{\n"
				));
				for nested in body {
					self.emit_op(code, nested, depth + 1, Some(depth + 1), declared)?;
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::ClearTile { tile } => {
				let (rows, columns) = self.register_tile(*tile)?;
				if rows * columns == 1 {
					if declared.insert(*tile) {
						code.push_str(&format!("{pad}float tile{tile};\n"));
					}
					code.push_str(&format!("{pad}tile{tile} = 0.0f;\n"));
				} else {
					if declared.insert(*tile) {
						code.push_str(&format!("{pad}float tile{tile}[{}];\n", rows * columns));
					}
					code.push_str(&format!(
						"{pad}[[unroll]] for (uint i = 0u; i < {}u; ++i) tile{tile}[i] = 0.0f;\n",
						rows * columns
					));
				}
			}
			TileOp::Guard { bound } => {
				let bound = self.lane_u32(*bound, induction)?;
				code.push_str(&format!("{pad}if (group_id.x >= {bound}) return;\n"));
			}
			TileOp::LaneLoop { extent, body } => {
				let bound = self.extent(*extent, induction)?;
				let loop_name = format!("loop{}", depth + 1);
				let bound_name = format!("bound{}", depth + 1);
				let lanes = self.shape.lanes;
				code.push_str(&format!(
					"{pad}for (uint {loop_name} = lane, {bound_name} = {bound}; {loop_name} < {bound_name}; {loop_name} += {lanes}u) {{\n"
				));
				for nested in body {
					self.emit_op(code, nested, depth + 1, Some(depth + 1), declared)?;
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::LaneLoopRange {
				start,
				step,
				extent,
				body,
			} => {
				let start = self.lane_u32(*start, induction)?;
				let bound = self.extent(*extent, induction)?;
				let loop_name = format!("loop{}", depth + 1);
				let bound_name = format!("bound{}", depth + 1);
				code.push_str(&format!(
					"{pad}for (uint {loop_name} = {start}, {bound_name} = {bound}; {loop_name} < {bound_name}; {loop_name} += {step}u) {{\n"
				));
				for nested in body {
					self.emit_op(code, nested, depth + 1, Some(depth + 1), declared)?;
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::LaneFold {
				acc,
				init,
				extent,
				op,
				element,
				serial,
			} => {
				let (rows, columns) = self.register_tile(*acc)?;
				if rows * columns != 1 {
					return Err(Error::invalid_argument(
						"lane fold accumulator must be a 1x1 register tile",
					));
				}
				if declared.insert(*acc) {
					code.push_str(&format!("{pad}float tile{acc};\n"));
				}
				let init = self.lane_f32(*init, induction)?;
				let bound = self.extent(*extent, induction)?;
				let (start, step) = if *serial {
					("0u".to_string(), "1u".to_string())
				} else {
					("lane".to_string(), format!("{}u", self.shape.lanes))
				};
				let loop_name = format!("loop{}", depth + 1);
				let bound_name = format!("bound{}", depth + 1);
				code.push_str(&format!(
					"{pad}tile{acc} = {init};\n{pad}for (uint {loop_name} = {start}, {bound_name} = {bound}; {loop_name} < {bound_name}; {loop_name} += {step}) {{\n"
				));
				let element = self.lane_f32(*element, Some(depth + 1))?;
				let update = match op {
					LaneReduce::Sum => format!("tile{acc} + ({element})"),
					LaneReduce::Max => format!("max(tile{acc}, {element})"),
					LaneReduce::Min => format!("min(tile{acc}, {element})"),
				};
				code.push_str(&format!("{pad}\ttile{acc} = {update};\n{pad}}}\n"));
			}
			TileOp::LaneFoldDual {
				acc,
				acc2,
				init,
				extent,
				element,
				element2,
			} => {
				for tile in [*acc, *acc2] {
					let (rows, columns) = self.register_tile(tile)?;
					if rows * columns != 1 {
						return Err(Error::invalid_argument(
							"lane fold accumulator must be a 1x1 register tile",
						));
					}
					if declared.insert(tile) {
						code.push_str(&format!("{pad}float tile{tile};\n"));
					}
				}
				let init = self.lane_f32(*init, induction)?;
				let bound = self.extent(*extent, induction)?;
				let loop_name = format!("loop{}", depth + 1);
				let bound_name = format!("bound{}", depth + 1);
				let lanes = self.shape.lanes;
				code.push_str(&format!(
					"{pad}tile{acc} = {init};\n{pad}tile{acc2} = {init};\n{pad}for (uint {loop_name} = lane, {bound_name} = {bound}; {loop_name} < {bound_name}; {loop_name} += {lanes}u) {{\n"
				));
				let element = self.lane_f32(*element, Some(depth + 1))?;
				let element2 = self.lane_f32(*element2, Some(depth + 1))?;
				code.push_str(&format!(
					"{pad}\ttile{acc} = tile{acc} + ({element});\n{pad}\ttile{acc2} = tile{acc2} + ({element2});\n{pad}}}\n"
				));
			}
			TileOp::AssignScalar { tile, index, value } => {
				let (rows, columns) = self.register_tile(*tile)?;
				if *index >= rows * columns {
					return Err(Error::invalid_argument(
						"scalar assignment exceeds its register tile",
					));
				}
				let scalar = rows * columns == 1;
				if declared.insert(*tile) {
					if scalar {
						code.push_str(&format!("{pad}float tile{tile};\n"));
					} else {
						code.push_str(&format!("{pad}float tile{tile}[{}];\n", rows * columns));
					}
				}
				let value = self.lane_f32(*value, induction)?;
				if scalar {
					code.push_str(&format!("{pad}tile{tile} = {value};\n"));
				} else {
					code.push_str(&format!("{pad}tile{tile}[{index}u] = {value};\n"));
				}
			}
			TileOp::StoreShared {
				tile,
				offset,
				value,
			} => {
				self.shared_tile(*tile)?;
				let offset = self.lane_u32(*offset, induction)?;
				let value = self.lane_f32(*value, induction)?;
				code.push_str(&format!("{pad}shared{tile}[{offset}] = {value};\n"));
			}
			TileOp::StoreSharedU32 {
				tile,
				offset,
				value,
			} => {
				self.shared_u32_tile(*tile)?;
				let offset = self.lane_u32(*offset, induction)?;
				let value = self.lane_u32(*value, induction)?;
				code.push_str(&format!("{pad}shared{tile}[{offset}] = {value};\n"));
			}
			TileOp::AssignScalarU32 { tile, index, value } => {
				let elements = self.register_u32_tile(*tile)?;
				if *index >= elements {
					return Err(Error::invalid_argument(
						"u32 scalar assignment exceeds its register tile",
					));
				}
				let scalar = elements == 1;
				if declared.insert(*tile) {
					if scalar {
						code.push_str(&format!("{pad}uint tile{tile};\n"));
					} else {
						code.push_str(&format!("{pad}uint tile{tile}[{elements}u];\n"));
					}
				}
				let value = self.lane_u32(*value, induction)?;
				if scalar {
					code.push_str(&format!("{pad}tile{tile} = {value};\n"));
				} else {
					code.push_str(&format!("{pad}tile{tile}[{index}u] = {value};\n"));
				}
			}
			TileOp::StoreScalarU32 {
				binding,
				address,
				value,
			} => {
				let address = self.lane_u32(*address, induction)?;
				let value = self.lane_u32(*value, induction)?;
				code.push_str(&format!(
					"{pad}storage_buffers[push.buffer{binding}].Store({address} * 4u, {value});\n"
				));
			}
			TileOp::TreeReduceArgmax {
				value_tile,
				index_tile,
			} => {
				let (rows, _, stride) = self.shared_tile(*value_tile)?;
				let elements = self.shared_u32_tile(*index_tile)?;
				let lanes = self.shape.lanes;
				if !lanes.is_power_of_two()
					|| u64::from(rows) * u64::from(stride) < u64::from(lanes)
					|| u64::from(elements) < u64::from(lanes)
				{
					return Err(Error::invalid_argument(
						"argmax reduction requires power-of-two lanes within the shared tiles",
					));
				}
				code.push_str(&format!(
					"{pad}for (uint stride{depth} = {}u; stride{depth} > 0u; stride{depth} >>= 1u) {{\n\
					 {pad}\tif (lane < stride{depth} &&\n\
					 {pad}\t\t(shared{value_tile}[lane + stride{depth}] > shared{value_tile}[lane] ||\n\
					 {pad}\t\t (shared{value_tile}[lane + stride{depth}] == shared{value_tile}[lane] &&\n\
					 {pad}\t\t  shared{index_tile}[lane + stride{depth}] < shared{index_tile}[lane]))) {{\n\
					 {pad}\t\tshared{value_tile}[lane] = shared{value_tile}[lane + stride{depth}];\n\
					 {pad}\t\tshared{index_tile}[lane] = shared{index_tile}[lane + stride{depth}];\n\
					 {pad}\t}}\n\
					 {pad}\tGroupMemoryBarrierWithGroupSync();\n{pad}}}\n",
					lanes / 2
				));
			}
			TileOp::TreeReduce { tile, op } => {
				let (rows, _, stride) = self.shared_tile(*tile)?;
				let lanes = self.shape.lanes;
				if !lanes.is_power_of_two() || u64::from(rows) * u64::from(stride) < u64::from(lanes) {
					return Err(Error::invalid_argument(
						"tree reduction requires power-of-two lanes within the shared tile",
					));
				}
				code.push_str(&format!(
					"{pad}for (uint stride{depth} = {}u; stride{depth} > 0u; stride{depth} >>= 1u) {{\n\
					 {pad}\tif (lane < stride{depth}) {{\n",
					lanes / 2
				));
				let combine = match op {
					LaneReduce::Sum => format!("shared{tile}[lane] += shared{tile}[lane + stride{depth}];"),
					LaneReduce::Max => format!(
						"shared{tile}[lane] = max(shared{tile}[lane], shared{tile}[lane + stride{depth}]);"
					),
					LaneReduce::Min => format!(
						"shared{tile}[lane] = min(shared{tile}[lane], shared{tile}[lane + stride{depth}]);"
					),
				};
				code.push_str(&format!(
					"{pad}\t\t{combine}\n{pad}\t}}\n{pad}\tGroupMemoryBarrierWithGroupSync();\n{pad}}}\n"
				));
			}
			TileOp::StoreScalar {
				binding,
				address,
				value,
			} => {
				let address = self.lane_u32(*address, induction)?;
				let value = self.lane_f32(*value, induction)?;
				code.push_str(&format!(
					"{pad}storage_buffers[push.buffer{binding}].Store({address} * 4u, asuint({value}));\n"
				));
			}
			TileOp::LaneIf { cond, body } => {
				let cond = self.lane_bool(*cond, induction)?;
				code.push_str(&format!("{pad}if ({cond}) {{\n"));
				for nested in body {
					self.emit_op(code, nested, depth + 1, induction, declared)?;
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::GuardIf { cond } => {
				let cond = self.lane_bool(*cond, induction)?;
				code.push_str(&format!("{pad}if ({cond}) {{\n{pad}\treturn;\n{pad}}}\n"));
			}
			TileOp::Return => {
				code.push_str(&format!("{pad}return;\n"));
			}
			TileOp::ExpertScan {
				offsets,
				experts,
				row,
				acc,
			} => {
				let (acc_rows, acc_columns) = self.register_tile(*acc)?;
				if declared.insert(*acc) {
					if acc_rows * acc_columns == 1 {
						code.push_str(&format!("{pad}float tile{acc};\n"));
					} else {
						code.push_str(&format!(
							"{pad}float tile{acc}[{}];\n",
							acc_rows * acc_columns
						));
					}
				}
				let experts = self.extent(*experts, induction)?;
				let row = self.lane_u32(*row, induction)?;
				let acc_ref = if acc_rows * acc_columns == 1 {
					format!("tile{acc}")
				} else {
					format!("tile{acc}[0u]")
				};
				let probe =
					format!("storage_buffers[push.buffer{offsets}].Load((uint({acc_ref}) + 1u) * 4u)");
				code.push_str(&format!(
					"{pad}{acc_ref} = 0.0f;\n\
					 {pad}while (uint({acc_ref}) + 1u < {experts} && {row} >= {probe}) {{\n\
					 {pad}\t{acc_ref} += 1.0f;\n{pad}}}\n"
				));
			}
			TileOp::BoundaryGateUp {
				acc,
				offsets,
				input,
				weight,
				bias,
				output,
				act,
				experts,
				rows,
				outputs,
				inner,
				gate_width,
				row_origin,
				column_origin,
			} => {
				let (fr, fc) = self.register_tile(*acc)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let rows = self.extent(*rows, induction)?;
				let outputs = self.extent(*outputs, induction)?;
				let inner = self.extent(*inner, induction)?;
				let gate_width = self.extent(*gate_width, induction)?;
				let experts = self.extent(*experts, induction)?;
				code.push_str(&format!(
					"{pad}[[unroll]] for (uint i = 0u; i < {fr}u; ++i) {{\n\
					 {pad}\tuint boundary_row = {row_origin} + thread_row * {fr}u + i;\n\
					 {pad}\tif (boundary_row >= {rows}) {{\n{pad}\t\tcontinue;\n{pad}\t}}\n\
					 {pad}\tuint expert = 0u;\n\
					 {pad}\twhile (expert + 1u < {experts} && boundary_row\n\
					 {pad}\t\t>= storage_buffers[push.buffer{offsets}].Load((expert + 1u) * 4u)) {{\n\
					 {pad}\t\t++expert;\n{pad}\t}}\n\
					 {pad}\tuint expert_row = expert * {outputs};\n\
					 {pad}\t[[unroll]] for (uint j = 0u; j < {fc}u; ++j) {{\n\
					 {pad}\t\tuint boundary_column = {column_origin} + thread_column * {fc}u + j;\n\
					 {pad}\t\tif (boundary_column >= {gate_width}) {{\n{pad}\t\t\tcontinue;\n{pad}\t\t}}\n\
					 {pad}\t\tfloat sum = asfloat(storage_buffers[push.buffer{bias}]\n\
					 {pad}\t\t\t.Load((expert_row + boundary_column) * 4u));\n\
					 {pad}\t\tfloat up = asfloat(storage_buffers[push.buffer{bias}]\n\
					 {pad}\t\t\t.Load((expert_row + boundary_column + {gate_width}) * 4u));\n\
					 {pad}\t\tfor (uint k = 0u; k < {inner}; ++k) {{\n\
					 {pad}\t\t\tfloat activation = asfloat(storage_buffers[push.buffer{input}]\n\
					 {pad}\t\t\t\t.Load((boundary_row * {inner} + k) * 4u));\n\
					 {pad}\t\t\tsum += activation * asfloat(storage_buffers[push.buffer{weight}]\n\
					 {pad}\t\t\t\t.Load(((expert_row + boundary_column) * {inner} + k) * 4u));\n\
					 {pad}\t\t\tup += activation * asfloat(storage_buffers[push.buffer{weight}]\n\
					 {pad}\t\t\t\t.Load(((expert_row + boundary_column + {gate_width}) * {inner} + k) * 4u));\n\
					 {pad}\t\t}}\n\
					 {pad}\t\tstorage_buffers[push.buffer{output}]\n\
					 {pad}\t\t\t.Store((boundary_row * {outputs} + boundary_column) * 4u, asuint(sum));\n\
					 {pad}\t\tstorage_buffers[push.buffer{output}].Store(\n\
					 {pad}\t\t\t(boundary_row * {outputs} + boundary_column + {gate_width}) * 4u,\n\
					 {pad}\t\t\tasuint(up));\n\
					 {pad}\t\tstorage_buffers[push.buffer{act}].Store(\n\
					 {pad}\t\t\t(boundary_row * {gate_width} + boundary_column) * 4u,\n\
					 {pad}\t\t\tasuint(tile_swiglu(sum, up)));\n\
					 {pad}\t}}\n{pad}}}\n"
				));
			}
			TileOp::BoundaryLinear {
				acc,
				offsets,
				input,
				weight,
				bias,
				output,
				experts,
				rows,
				outputs,
				inner,
				column_extent,
				row_origin,
				column_origin,
				transposed,
			} => {
				let (fr, fc) = self.register_tile(*acc)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let rows = self.extent(*rows, induction)?;
				let outputs = self.extent(*outputs, induction)?;
				let inner = self.extent(*inner, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				let experts = self.extent(*experts, induction)?;
				let sum_init = match bias {
					Some(bias) => format!(
						"asfloat(storage_buffers[push.buffer{bias}]\n\
						 {pad}\t\t\t.Load((expert_row + boundary_column) * 4u))"
					),
					None => String::from("0.0f"),
				};
				// The transposed (data-backward) contraction reduces over
				// `outputs`: `dX[row][k] = Σ_n dY[row][n] * W[(expert*outputs
				// + n)][k]` — the expert term strides the weight's
				// reduction axis and `inner` is the output column extent.
				let body = if *transposed {
					format!(
						"{pad}\t\tfloat sum = {sum_init};\n\
						 {pad}\t\tfor (uint k = 0u; k < {outputs}; ++k) {{\n\
						 {pad}\t\t\tsum += asfloat(storage_buffers[push.buffer{input}]\n\
						 {pad}\t\t\t\t.Load((boundary_row * {outputs} + k) * 4u))\n\
						 {pad}\t\t\t\t* asfloat(storage_buffers[push.buffer{weight}]\n\
						 {pad}\t\t\t\t.Load(((expert_row + k) * {inner} + boundary_column) * 4u));\n\
						 {pad}\t\t}}\n\
						 {pad}\t\tstorage_buffers[push.buffer{output}]\n\
						 {pad}\t\t\t.Store((boundary_row * {inner} + boundary_column) * 4u, asuint(sum));\n"
					)
				} else {
					format!(
						"{pad}\t\tfloat sum = {sum_init};\n\
						 {pad}\t\tfor (uint k = 0u; k < {inner}; ++k) {{\n\
						 {pad}\t\t\tsum += asfloat(storage_buffers[push.buffer{input}]\n\
						 {pad}\t\t\t\t.Load((boundary_row * {inner} + k) * 4u))\n\
						 {pad}\t\t\t\t* asfloat(storage_buffers[push.buffer{weight}]\n\
						 {pad}\t\t\t\t.Load(((expert_row + boundary_column) * {inner} + k) * 4u));\n\
						 {pad}\t\t}}\n\
						 {pad}\t\tstorage_buffers[push.buffer{output}]\n\
						 {pad}\t\t\t.Store((boundary_row * {outputs} + boundary_column) * 4u, asuint(sum));\n"
					)
				};
				code.push_str(&format!(
					"{pad}[[unroll]] for (uint i = 0u; i < {fr}u; ++i) {{\n\
					 {pad}\tuint boundary_row = {row_origin} + thread_row * {fr}u + i;\n\
					 {pad}\tif (boundary_row >= {rows}) {{\n{pad}\t\tcontinue;\n{pad}\t}}\n\
					 {pad}\tuint expert = 0u;\n\
					 {pad}\twhile (expert + 1u < {experts} && boundary_row\n\
					 {pad}\t\t>= storage_buffers[push.buffer{offsets}].Load((expert + 1u) * 4u)) {{\n\
					 {pad}\t\t++expert;\n{pad}\t}}\n\
					 {pad}\tuint expert_row = expert * {outputs};\n\
					 {pad}\t[[unroll]] for (uint j = 0u; j < {fc}u; ++j) {{\n\
					 {pad}\t\tuint boundary_column = {column_origin} + thread_column * {fc}u + j;\n\
					 {pad}\t\tif (boundary_column >= {column_extent}) {{\n{pad}\t\t\tcontinue;\n{pad}\t\t}}\n\
					 {body}\
					 {pad}\t}}\n{pad}}}\n"
				));
			}
			TileOp::LaneGuard { lane, body } => {
				let lane = self.lane_u32(*lane, induction)?;
				code.push_str(&format!("{pad}if (lane == {lane}) {{\n"));
				for nested in body {
					self.emit_op(code, nested, depth + 1, induction, declared)?;
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::LoadShared {
				tile,
				z,
				binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				row_offset,
				mask_origin,
				column_offset,
				transposed,
				vec,
			} => {
				let (rows, columns, stride) = self.shared_tile(*tile)?;
				if *vec == 0 || !columns.is_multiple_of(*vec) {
					return Err(Error::invalid_argument(
						"staging width must divide the shared tile columns",
					));
				}
				let mut row_origin = self.extent(*row_origin, induction)?;
				if let Some(offset) = row_offset {
					row_origin = format!("{row_origin} + {}", self.extent(*offset, induction)?);
				}
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				let mask_row = match mask_origin {
					Some(origin) => format!("{} + local_row", self.extent(*origin, induction)?),
					None => String::from("global_row"),
				};
				let load_column = match column_offset {
					Some(offset) => {
						format!("(global_column + {})", self.extent(*offset, induction)?)
					}
					None => String::from("global_column"),
				};
				code.push_str(&format!(
					"{pad}for (uint element = lane * {vec}u; element < {rows}u * {columns}u; element += {}u) {{\n\
					 {pad}\tuint local_row = element / {columns}u;\n\
					 {pad}\tuint local_column = element % {columns}u;\n\
					 {pad}\tuint global_row = {row_origin} + local_row;\n\
					 {pad}\tuint global_column = {column_origin} + local_column;\n\
					 {pad}\tif ({mask_row} < {row_extent} && global_column + {vec}u <= {column_extent}) {{\n",
					self.shape.lanes * vec
				));
				let address = if *transposed {
					format!("({load_column} * {row_extent} + global_row) * 4u")
				} else {
					format!("(global_row * {column_extent} + {load_column}) * 4u")
				};
				let binding = if *z {
					self.z_binding(*binding)?
				} else {
					format!("push.buffer{binding}")
				};
				if *vec == 1 {
					code.push_str(&format!(
						"{pad}\t\tshared{tile}[local_row * {stride}u + local_column] = asfloat(storage_buffers[{binding}].Load({address}));\n"
					));
				} else {
					code.push_str(&format!(
						"{pad}\t\tfloat4 packed{tile} = storage_buffers[{binding}].Load<float4>({address});\n"
					));
					for (element, member) in ["x", "y", "z", "w"].iter().enumerate().take(*vec as usize) {
						code.push_str(&format!(
							"{pad}\t\tshared{tile}[local_row * {stride}u + local_column + {element}u] = packed{tile}.{member};\n"
						));
					}
				}
				code.push_str(&format!("{pad}\t}} else {{\n"));
				let tail_address = if *transposed {
					format!("(({load_column} + j) * {row_extent} + global_row) * 4u")
				} else {
					format!("(global_row * {column_extent} + {load_column} + j) * 4u")
				};
				code.push_str(&format!(
					"{pad}\t\t[[unroll]] for (uint j = 0u; j < {vec}u; ++j) {{\n\
					 {pad}\t\t\tshared{tile}[local_row * {stride}u + local_column + j] =\n\
					 {pad}\t\t\t\t({mask_row} < {row_extent} && global_column + j < {column_extent})\n\
					 {pad}\t\t\t\t? asfloat(storage_buffers[{binding}].Load({tail_address}))\n\
					 {pad}\t\t\t\t: 0.0f;\n\
					 {pad}\t\t}}\n{pad}\t}}\n{pad}}}\n"
				));
			}
			TileOp::LoadSharedPrologue {
				tile,
				dy,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				gate,
				..
			} => {
				let (rows, columns, stride) = self.shared_tile(*tile)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				let gate = self.lane_bool(*gate, induction)?;
				code.push_str(&format!(
					"{pad}for (uint element = lane; element < {rows}u * {columns}u; element += {}u) {{\n\
					 {pad}\tuint local_row = element / {columns}u;\n\
					 {pad}\tuint local_column = element % {columns}u;\n\
					 {pad}\tuint global_row = {row_origin} + local_row;\n\
					 {pad}\tuint global_column = {column_origin} + local_column;\n\
					 {pad}\tfloat dy{tile} = (global_row < {row_extent} && global_column < {column_extent})\n\
					 {pad}\t\t? tile_prologue_value(global_row, global_column) : 0.0f;\n\
					 {pad}\tshared{tile}[local_row * {stride}u + local_column] = dy{tile};\n\
					 {pad}\tif ({gate} && global_row < {row_extent} && global_column < {column_extent}) {{\n\
					 {pad}\t\tstorage_buffers[push.buffer{dy}].Store((global_row * {column_extent} + global_column) * 4u, asuint(dy{tile}));\n\
					 {pad}\t}}\n{pad}}}\n",
					self.shape.lanes
				));
			}
			TileOp::LoadSharedNorm {
				tile,
				normalized,
				inv_stddev,
				norm_weight,
				upstream,
				addend,
				norm_input_gradient,
				weight_contribution,
				stage1,
				stats,
				inverse_columns,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				gate,
			} => {
				let (rows, columns, stride) = self.shared_tile(*tile)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				let inverse_columns = self.lane_f32(*inverse_columns, induction)?;
				let gate = self.lane_bool(*gate, induction)?;
				let dy_expr = if let Some(addend) = addend {
					format!(
						"adjoint{tile} + asfloat(storage_buffers[push.buffer{addend}]\n\
						 {pad}\t\t.Load((global_row * {column_extent} + global_column) * 4u))"
					)
				} else {
					format!("adjoint{tile}")
				};
				let stage1_store = if let Some(stage1) = stage1 {
					format!(
						"{pad}\t\t\tstorage_buffers[push.buffer{stage1}].Store((global_row * {column_extent} + global_column) * 4u, asuint(dyv{tile}));\n"
					)
				} else {
					String::new()
				};
				code.push_str(&format!(
					"{pad}for (uint element = lane; element < {rows}u * {columns}u; element += {}u) {{\n\
					 {pad}\tuint local_row = element / {columns}u;\n\
					 {pad}\tuint local_column = element % {columns}u;\n\
					 {pad}\tuint global_row = {row_origin} + local_row;\n\
					 {pad}\tuint global_column = {column_origin} + local_column;\n\
					 {pad}\tfloat dy{tile} = 0.0f;\n\
					 {pad}\tif (global_row < {row_extent} && global_column < {column_extent}) {{\n\
					 {pad}\t\tfloat upstream{tile} = asfloat(storage_buffers[push.buffer{upstream}]\n\
					 {pad}\t\t\t.Load((global_row * {column_extent} + global_column) * 4u));\n\
					 {pad}\t\tfloat normalized{tile} = asfloat(storage_buffers[push.buffer{normalized}]\n\
					 {pad}\t\t\t.Load((global_row * {column_extent} + global_column) * 4u));\n\
					 {pad}\t\tfloat gamma{tile} = asfloat(storage_buffers[push.buffer{norm_weight}]\n\
					 {pad}\t\t\t.Load(global_column * 4u));\n\
					 {pad}\t\tfloat adjoint{tile} = asfloat(storage_buffers[push.buffer{inv_stddev}]\n\
					 {pad}\t\t\t.Load(global_row * 4u)) * (upstream{tile} * gamma{tile}\n\
					 {pad}\t\t\t- shared{stats}[local_row * 2u] * {inverse_columns}\n\
					 {pad}\t\t\t- normalized{tile} * shared{stats}[local_row * 2u + 1u] * {inverse_columns});\n\
					 {pad}\t\tfloat dyv{tile} = {dy_expr};\n\
					 {pad}\t\tif ({gate}) {{\n\
					 {pad}\t\t\tstorage_buffers[push.buffer{norm_input_gradient}].Store((global_row * {column_extent} + global_column) * 4u, asuint(adjoint{tile}));\n\
					 {pad}\t\t\tstorage_buffers[push.buffer{weight_contribution}].Store((global_row * {column_extent} + global_column) * 4u, asuint(upstream{tile} * normalized{tile}));\n\
					 {stage1_store}\
					 {pad}\t\t}}\n\
					 {pad}\t\tdy{tile} = dyv{tile};\n\
					 {pad}\t}}\n\
					 {pad}\tshared{tile}[local_row * {stride}u + local_column] = dy{tile};\n\
					 {pad}}}\n",
					self.shape.lanes
				));
			}
			TileOp::DotSharedWide {
				acc,
				lhs,
				rhs,
				column_tiles,
			} => {
				let (acc_rows, acc_columns) = self.register_tile(*acc)?;
				let (left_rows, left_columns, left_stride) = self.shared_tile(*lhs)?;
				let (right_rows, right_columns, right_stride) = self.shared_tile(*rhs)?;
				if left_columns != right_columns
					|| acc_rows != self.shape.fragment_rows * column_tiles
					|| acc_columns != self.shape.fragment_columns
					|| left_rows != self.shape.tile_rows
					|| right_rows != self.shape.tile_columns * column_tiles
				{
					return Err(Error::invalid_argument(
						"wide dot tiles do not match the program's execution shape",
					));
				}
				let (tm, tn) = (self.shape.fragment_rows, self.shape.fragment_columns);
				code.push_str(&format!(
					"{pad}[[unroll]] for (uint inner = 0u; inner < {left_columns}u; ++inner) {{\n\
					 {pad}\tfloat left_fragment[{tm}];\n"
				));
				for row in 0..tm {
					code.push_str(&format!(
						"{pad}\tleft_fragment[{row}] = shared{lhs}[(thread_row * {tm}u + {row}u) * {left_stride}u + inner];\n"
					));
				}
				code.push_str(&format!(
					"{pad}\t[[unroll]] for (uint segment = 0u; segment < {column_tiles}u; ++segment) {{\n\
					 {pad}\t\tfloat right_fragment[{tn}];\n"
				));
				for column in 0..tn {
					code.push_str(&format!(
						"{pad}\t\tright_fragment[{column}] = shared{rhs}[segment * {}u * {right_stride}u + (thread_column * {tn}u + {column}u) * {right_stride}u + inner];\n",
						self.shape.tile_columns
					));
				}
				for row in 0..tm {
					for column in 0..tn {
						let acc_index = format!("segment * {}u + {row}u * {tn}u + {column}u", tm * tn);
						let acc_ref = self.register_ref(*acc, acc_index)?;
						code.push_str(&format!(
							"{pad}\t\t{acc_ref} += left_fragment[{row}] * right_fragment[{column}];\n"
						));
					}
				}
				code.push_str(&format!("{pad}\t}}\n{pad}}}\n"));
			}
			TileOp::StoreTileSegment {
				acc,
				binding,
				row_origin,
				row_lo,
				row_hi,
				column_origin,
				column_extent,
				column_tiles,
			} => {
				let (rows, columns) = self.register_tile(*acc)?;
				if rows != self.shape.fragment_rows * column_tiles || columns != self.shape.fragment_columns
				{
					return Err(Error::invalid_argument(
						"segment store accumulator does not match the fragment shape",
					));
				}
				let (tm, tn) = (self.shape.fragment_rows, self.shape.fragment_columns);
				let row_origin = self.extent(*row_origin, induction)?;
				let row_lo = self.extent(*row_lo, induction)?;
				let row_hi = self.extent(*row_hi, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				code.push_str(&format!(
					"{pad}uint output_row = {row_origin} + thread_row * {tm}u;\n\
					 {pad}[[unroll]] for (uint segment = 0u; segment < {column_tiles}u; ++segment) {{\n\
					 {pad}\tuint output_column = {column_origin} + segment * {}u + thread_column * {tn}u;\n",
					self.shape.tile_columns
				));
				for row in 0..tm {
					for column in 0..tn {
						let acc_index = format!("segment * {}u + {row}u * {tn}u + {column}u", tm * tn);
						let acc_ref = self.register_ref(*acc, acc_index)?;
						code.push_str(&format!(
							"{pad}\t{{\n\
							 {pad}\t\tuint store_row = output_row + {row}u;\n\
							 {pad}\t\tuint store_column = output_column + {column}u;\n\
							 {pad}\t\tif (store_row >= {row_lo} && store_row < {row_hi} && store_column < {column_extent}) {{\n\
							 {pad}\t\t\tstorage_buffers[push.buffer{binding}].Store((store_row * {column_extent} + store_column) * 4u, asuint({acc_ref}));\n\
							 {pad}\t\t}}\n{pad}\t}}\n"
						));
					}
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::LoadSharedConcat {
				tile,
				bindings,
				widths,
				materialize,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
			} => {
				let (rows, columns, stride) = self.shared_tile(*tile)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				let width = |index: usize| -> Result<String> {
					Ok(format!(
						"push.{}",
						self.push_scalars.get(index).ok_or_else(|| {
							Error::invalid_argument("concat staging references a missing push scalar")
						})?
					))
				};
				let (w0, w1, w2) = (width(widths[0])?, width(widths[1])?, width(widths[2])?);
				code.push_str(&format!(
					"{pad}for (uint element = lane; element < {rows}u * {columns}u; element += {}u) {{\n\
					 {pad}\tuint local_row = element / {columns}u;\n\
					 {pad}\tuint local_column = element % {columns}u;\n\
					 {pad}\tuint global_row = {row_origin} + local_row;\n\
					 {pad}\tuint global_k = {column_origin} + local_column;\n\
					 {pad}\tfloat concat_value = 0.0f;\n\
					 {pad}\tif (global_row < {row_extent} && global_k < {column_extent}) {{\n\
					 {pad}\t\tif (global_k < {w0}) concat_value = asfloat(storage_buffers[push.buffer{}].Load((global_row * {w0} + global_k) * 4u));\n\
					 {pad}\t\telse if (global_k < {w0} + {w1}) concat_value = asfloat(storage_buffers[push.buffer{}].Load((global_row * {w1} + global_k - {w0}) * 4u));\n\
					 {pad}\t\telse concat_value = asfloat(storage_buffers[push.buffer{}].Load((global_row * {w2} + global_k - {w0} - {w1}) * 4u));\n\
					 {pad}\t\tif (group_id.y == 0u) storage_buffers[push.buffer{materialize}].Store((global_row * {column_extent} + global_k) * 4u, asuint(concat_value));\n\
					 {pad}\t}}\n\
					 {pad}\tshared{tile}[local_row * {stride}u + local_column] = concat_value;\n\
					 {pad}}}\n",
					self.shape.lanes,
					bindings[0],
					bindings[1],
					bindings[2]
				));
			}
			TileOp::Barrier => {
				code.push_str(&format!("{pad}GroupMemoryBarrierWithGroupSync();\n"));
			}
			TileOp::DotShared { acc, lhs, rhs } => {
				let (acc_rows, acc_columns) = self.register_tile(*acc)?;
				let (left_rows, left_columns, left_stride) = self.shared_tile(*lhs)?;
				let (right_rows, right_columns, right_stride) = self.shared_tile(*rhs)?;
				if left_columns != right_columns
					|| acc_rows != self.shape.fragment_rows
					|| acc_columns != self.shape.fragment_columns
					|| left_rows != self.shape.tile_rows
					|| right_rows != self.shape.tile_columns
				{
					return Err(Error::invalid_argument(
						"dot tiles do not match the program's execution shape",
					));
				}
				let (tm, tn) = (self.shape.fragment_rows, self.shape.fragment_columns);
				code.push_str(&format!(
					"{pad}[[unroll]] for (uint inner = 0u; inner < {left_columns}u; ++inner) {{\n\
					 {pad}\tfloat left_fragment[{tm}];\n\
					 {pad}\tfloat right_fragment[{tn}];\n"
				));
				for row in 0..tm {
					code.push_str(&format!(
						"{pad}\tleft_fragment[{row}] = shared{lhs}[(thread_row * {tm}u + {row}u) * {left_stride}u + inner];\n"
					));
				}
				for column in 0..tn {
					code.push_str(&format!(
						"{pad}\tright_fragment[{column}] = shared{rhs}[(thread_column * {tn}u + {column}u) * {right_stride}u + inner];\n"
					));
				}
				for row in 0..tm {
					for column in 0..tn {
						let acc_index = format!("{row}u * {tn}u + {column}u");
						let acc_ref = self.register_ref(*acc, acc_index)?;
						code.push_str(&format!(
							"{pad}\t{acc_ref} += left_fragment[{row}] * right_fragment[{column}];\n"
						));
					}
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::DotSharedParam {
				acc,
				lhs,
				rhs,
				bias_acc,
				bias_gate,
			} => {
				let (acc_rows, acc_columns) = self.register_tile(*acc)?;
				let (bias_rows, bias_columns) = self.register_tile(*bias_acc)?;
				let (left_rows, left_columns, left_stride) = self.shared_tile(*lhs)?;
				let (right_rows, right_columns, right_stride) = self.shared_tile(*rhs)?;
				if left_columns != right_columns
					|| acc_rows != self.shape.fragment_rows
					|| acc_columns != self.shape.fragment_columns
					|| bias_rows != self.shape.fragment_rows
					|| bias_columns != 1
					|| left_rows != self.shape.tile_rows
					|| right_rows != self.shape.tile_columns
				{
					return Err(Error::invalid_argument(
						"dot tiles do not match the program's execution shape",
					));
				}
				let gate = self.lane_bool(*bias_gate, induction)?;
				let (tm, tn) = (self.shape.fragment_rows, self.shape.fragment_columns);
				code.push_str(&format!(
					"{pad}[[unroll]] for (uint inner = 0u; inner < {left_columns}u; ++inner) {{\n\
					 {pad}\tfloat left_fragment[{tm}];\n\
					 {pad}\tfloat right_fragment[{tn}];\n"
				));
				for row in 0..tm {
					code.push_str(&format!(
						"{pad}\tleft_fragment[{row}] = shared{lhs}[(thread_row * {tm}u + {row}u) * {left_stride}u + inner];\n"
					));
				}
				for column in 0..tn {
					code.push_str(&format!(
						"{pad}\tright_fragment[{column}] = shared{rhs}[(thread_column * {tn}u + {column}u) * {right_stride}u + inner];\n"
					));
				}
				for row in 0..tm {
					let bias_ref = self.register_ref(*bias_acc, format!("{row}u"))?;
					code.push_str(&format!(
						"{pad}\tif ({gate}) {{\n\
						 {pad}\t\t{bias_ref} += left_fragment[{row}];\n\
						 {pad}\t}}\n"
					));
					for column in 0..tn {
						let acc_index = format!("{row}u * {tn}u + {column}u");
						let acc_ref = self.register_ref(*acc, acc_index)?;
						code.push_str(&format!(
							"{pad}\t{acc_ref} += left_fragment[{row}] * right_fragment[{column}];\n"
						));
					}
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::StoreTileBias {
				acc,
				bias_binding,
				binding,
				z,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				bias_origin,
				act,
			} => {
				self.emit_epilogue(
					code,
					*acc,
					*bias_binding,
					None,
					*binding,
					None,
					None,
					None,
					*z,
					(*row_origin, *column_origin, *row_extent, *column_extent),
					*bias_origin,
					*act,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTileAct {
				acc,
				bias_binding,
				linear_binding,
				act_binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				..
			} => {
				self.emit_epilogue(
					code,
					*acc,
					*bias_binding,
					None,
					*linear_binding,
					None,
					Some(*act_binding),
					None,
					false,
					(*row_origin, *column_origin, *row_extent, *column_extent),
					None,
					None,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTileResidual {
				acc,
				bias_binding,
				residual_binding,
				linear_binding,
				sum_binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
			} => {
				self.emit_epilogue(
					code,
					*acc,
					*bias_binding,
					Some(*residual_binding),
					*linear_binding,
					Some(*sum_binding),
					None,
					None,
					false,
					(*row_origin, *column_origin, *row_extent, *column_extent),
					None,
					None,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTileResidualAct {
				acc,
				bias_binding,
				residual_binding,
				linear_binding,
				sum_binding,
				act_binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				..
			} => {
				self.emit_epilogue(
					code,
					*acc,
					*bias_binding,
					Some(*residual_binding),
					*linear_binding,
					Some(*sum_binding),
					Some(*act_binding),
					None,
					false,
					(*row_origin, *column_origin, *row_extent, *column_extent),
					None,
					None,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTileResidualNorm {
				acc,
				bias_binding,
				residual_binding,
				linear_binding,
				sum_binding,
				tile,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
			} => {
				let (_, _, stride) = self.shared_tile(*tile)?;
				self.emit_epilogue(
					code,
					*acc,
					*bias_binding,
					Some(*residual_binding),
					*linear_binding,
					Some(*sum_binding),
					None,
					Some((*tile, stride)),
					false,
					(*row_origin, *column_origin, *row_extent, *column_extent),
					None,
					None,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTileGateUp {
				gate_acc,
				up_acc,
				gate_bias_binding,
				up_bias_binding,
				binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
			} => {
				let (rows, columns) = self.register_tile(*gate_acc)?;
				if self.register_tile(*up_acc)? != (rows, columns) {
					return Err(Error::invalid_argument(
						"gate/up accumulator shapes must agree",
					));
				}
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				for row in 0..rows {
					for column in 0..columns {
						let index = row * columns + column;
						let gate_ref = self.register_ref(*gate_acc, format!("{index}u"))?;
						let up_ref = self.register_ref(*up_acc, format!("{index}u"))?;
						code.push_str(&format!(
							"{pad}{{\n\
							 {pad}\tuint output_row = {row_origin} + thread_row * {rows}u + {row}u;\n\
							 {pad}\tuint output_column = {column_origin} + thread_column * {columns}u + {column}u;\n\
							 {pad}\tif (output_row < {row_extent} && output_column < {column_extent}) {{\n\
							 {pad}\t\tuint element = output_row * {column_extent} + output_column;\n\
							 {pad}\t\tfloat gate_bias = {gate_ref}\n\
							 {pad}\t\t\t+ asfloat(storage_buffers[push.buffer{gate_bias_binding}].Load(output_column * 4u));\n\
							 {pad}\t\tfloat up_bias = {up_ref}\n\
							 {pad}\t\t\t+ asfloat(storage_buffers[push.buffer{up_bias_binding}].Load(output_column * 4u));\n\
							 {pad}\t\tfloat silu = gate_bias / (1.0f + exp(-gate_bias));\n\
							 {pad}\t\tstorage_buffers[push.buffer{binding}].Store(element * 4u, asuint(silu * up_bias));\n\
							 {pad}\t}}\n{pad}}}\n"
						));
					}
				}
			}
			TileOp::StoreTileGated {
				gate_acc,
				up_acc,
				bias_binding,
				linear_binding,
				act_binding,
				linear_stride,
				row_origin,
				column_origin,
				row_extent,
				gate_width,
				bias_origin,
			} => {
				self.emit_gated_epilogue(
					code,
					*gate_acc,
					*up_acc,
					*bias_binding,
					*linear_binding,
					*act_binding,
					*linear_stride,
					(*row_origin, *column_origin, *row_extent, *gate_width),
					*bias_origin,
					induction,
					&pad,
				)?;
			}
			TileOp::StoreTile {
				acc,
				binding,
				row_origin,
				column_origin,
				row_extent,
				column_extent,
				mask_origin,
			} => {
				let (rows, columns) = self.register_tile(*acc)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let column_origin = self.extent(*column_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let column_extent = self.extent(*column_extent, induction)?;
				for row in 0..rows {
					let mask_row = match mask_origin {
						Some(origin) => format!(
							"{} + thread_row * {rows}u + {row}u",
							self.extent(*origin, induction)?
						),
						None => String::from("output_row"),
					};
					for column in 0..columns {
						let acc_ref = self.register_ref(*acc, format!("{row}u * {columns}u + {column}u"))?;
						code.push_str(&format!(
							"{pad}{{\n\
							 {pad}\tuint output_row = {row_origin} + thread_row * {rows}u + {row}u;\n\
							 {pad}\tuint output_column = {column_origin} + thread_column * {columns}u + {column}u;\n\
							 {pad}\tif ({mask_row} < {row_extent} && output_column < {column_extent}) {{\n\
							 {pad}\t\tstorage_buffers[push.buffer{binding}].Store(\n\
							 {pad}\t\t\t(output_row * {column_extent} + output_column) * 4u,\n\
							 {pad}\t\t\tasuint({acc_ref})\n\
							 {pad}\t\t);\n\
							 {pad}\t}}\n{pad}}}\n"
						));
					}
				}
			}
			TileOp::StoreTileBiasParam {
				acc,
				binding,
				origin,
				row_origin,
				row_extent,
				gate,
			} => {
				let (rows, columns) = self.register_tile(*acc)?;
				if columns != 1 {
					return Err(Error::invalid_argument(
						"parameter bias store needs a single-column register tile",
					));
				}
				let origin = self.extent(*origin, induction)?;
				let row_origin = self.extent(*row_origin, induction)?;
				let row_extent = self.extent(*row_extent, induction)?;
				let gate = self.lane_bool(*gate, induction)?;
				code.push_str(&format!("{pad}if ({gate}) {{\n"));
				for row in 0..rows {
					let acc_ref = self.register_ref(*acc, format!("{row}u"))?;
					code.push_str(&format!(
						"{pad}\t{{\n\
						 {pad}\t\tuint output_feature = {row_origin} + thread_row * {rows}u + {row}u;\n\
						 {pad}\t\tif (output_feature < {row_extent}) {{\n\
						 {pad}\t\t\tstorage_buffers[push.buffer{binding}].Store(\n\
						 {pad}\t\t\t\t(({origin} + output_feature) * 4u),\n\
						 {pad}\t\t\t\tasuint({acc_ref})\n\
						 {pad}\t\t\t);\n\
						 {pad}\t\t}}\n\
						 {pad}\t}}\n"
					));
				}
				code.push_str(&format!("{pad}}}\n"));
			}
			TileOp::LoadLinear { .. }
			| TileOp::GatherRow { .. }
			| TileOp::Pointwise { .. }
			| TileOp::Store { .. } => {
				return Err(Error::invalid_argument(
					"elementwise op inside a block tile program",
				));
			}
		}
		Ok(())
	}

	/// The shared fragment-store skeleton of every GEMM epilogue op:
	/// `linear = acc[r,c] + bias[column]` always writes to
	/// `linear_binding`; `residual` adds a flat-element input and
	/// `sum_binding` materializes the sum; `act_binding` stores
	/// `apply_act` of the pre-act value (sum when a residual is bound,
	/// the bare linear value otherwise) — matching each donor's
	/// epilogue order.
	#[allow(clippy::too_many_arguments)]
	fn emit_epilogue(
		&self,
		code: &mut String,
		acc: usize,
		bias_binding: usize,
		residual_binding: Option<usize>,
		linear_binding: usize,
		sum_binding: Option<usize>,
		act_binding: Option<usize>,
		shared_sum: Option<(usize, u32)>,
		z: bool,
		extents: (Extent, Extent, Extent, Extent),
		bias_origin: Option<usize>,
		bias_act: Option<BiasAct>,
		induction: Option<usize>,
		pad: &str,
	) -> Result<()> {
		let (rows, columns) = self.register_tile(acc)?;
		let row_origin = self.extent(extents.0, induction)?;
		let column_origin = self.extent(extents.1, induction)?;
		let row_extent = self.extent(extents.2, induction)?;
		let column_extent = self.extent(extents.3, induction)?;
		for row in 0..rows {
			for column in 0..columns {
				let zed = |slot: usize| -> Result<String> {
					if z {
						self.z_binding(slot)
					} else {
						Ok(format!("push.buffer{slot}"))
					}
				};
				let bias_text = zed(bias_binding)?;
				let linear_text = zed(linear_binding)?;
				let bias_index = match bias_origin {
					Some(origin) => format!(
						"({} + output_column) * 4u",
						self.lane_u32(origin, induction)?
					),
					None => String::from("output_column * 4u"),
				};
				let sum_decl = if shared_sum.is_some() {
					format!("{pad}\tfloat sum = 0.0f;\n")
				} else {
					String::new()
				};
				// The `matmul_nt_bias*` donors apply a fixed activation
				// to `acc + bias` at the single store — spelled with each
				// donor's literal constants so the artifact is
				// bit-identical.
				let stored = match bias_act {
					None => "linear_value".to_string(),
					Some(BiasAct::Relu) => "max(0.0f, linear_value)".to_string(),
					Some(BiasAct::Gelu) => "0.5f * linear_value * (1.0f + tanh(0.7978845608f * (linear_value + 0.044715f * linear_value * linear_value * linear_value)))".to_string(),
					Some(BiasAct::Silu) => {
						"linear_value * (1.0f / (1.0f + exp(-linear_value)))".to_string()
					}
				};
				let acc_ref = self.register_ref(acc, format!("{row}u * {columns}u + {column}u"))?;
				let mut body = format!(
					"{pad}{{\n\
					 {pad}\tuint output_row = {row_origin} + thread_row * {rows}u + {row}u;\n\
					 {pad}\tuint output_column = {column_origin} + thread_column * {columns}u + {column}u;\n{sum_decl}\
					 {pad}\tif (output_row < {row_extent} && output_column < {column_extent}) {{\n\
					 {pad}\t\tuint element = output_row * {column_extent} + output_column;\n\
					 {pad}\t\tfloat linear_value = {acc_ref}\n\
					 {pad}\t\t\t+ asfloat(storage_buffers[{bias_text}].Load({bias_index}));\n\
					 {pad}\t\tstorage_buffers[{linear_text}].Store(element * 4u, asuint({stored}));\n"
				);
				if let Some(residual) = residual_binding {
					let decl = if shared_sum.is_some() {
						"sum"
					} else {
						"float sum"
					};
					body.push_str(&format!(
						"{pad}\t\t{decl} = linear_value\n\
						 {pad}\t\t\t+ asfloat(storage_buffers[push.buffer{residual}].Load(element * 4u));\n"
					));
				}
				if let Some(sum) = sum_binding {
					body.push_str(&format!(
						"{pad}\t\tstorage_buffers[push.buffer{sum}].Store(element * 4u, asuint(sum));\n"
					));
				}
				if let Some(act) = act_binding {
					let operand = if residual_binding.is_some() {
						"sum"
					} else {
						"linear_value"
					};
					body.push_str(&format!(
						"{pad}\t\tstorage_buffers[push.buffer{act}].Store(element * 4u, asuint(apply_act({operand})));\n"
					));
				}
				body.push_str(&format!("{pad}\t}}\n"));
				if let Some((tile, stride)) = shared_sum {
					body.push_str(&format!(
						"{pad}\tshared{tile}[(thread_row * {rows}u + {row}u) * {stride}u \
						 + thread_column * {columns}u + {column}u] = sum;\n"
					));
				}
				body.push_str(&format!("{pad}}}\n"));
				code.push_str(&body);
			}
		}
		Ok(())
	}

	/// `(act_kind_scalar, alpha_scalar)` names when the op tree stores a
	/// `linear_act` epilogue — the emitter then spells the donor's
	/// `apply_act` helper once ahead of `main`.
	fn act_epilogue(&self, ops: &[TileOp]) -> Result<Option<(String, String)>> {
		for op in ops {
			let found = match op {
				TileOp::StoreTileAct { kind, alpha, .. }
				| TileOp::StoreTileResidualAct { kind, alpha, .. } => {
					let kind = self.push_scalars.get(*kind).cloned().ok_or_else(|| {
						Error::invalid_argument("act epilogue references a missing push scalar")
					})?;
					let alpha = self.push_f32_scalars.get(*alpha).cloned().ok_or_else(|| {
						Error::invalid_argument("act epilogue references a missing f32 scalar")
					})?;
					Some((kind, alpha))
				}
				TileOp::Loop { body, .. }
				| TileOp::LaneLoop { body, .. }
				| TileOp::LaneGuard { body, .. } => self.act_epilogue(body)?,
				_ => None,
			};
			if found.is_some() {
				return Ok(found);
			}
		}
		Ok(None)
	}

	/// `linear_gated_act` epilogue: `linear = gate_acc[r,c] + bias[col]`
	/// stores at `element`, `up = up_acc[r,c] + bias[col + gate_width]`
	/// at `element + gate_width`, and `swiglu(linear, up)` into the
	/// half-width `act_binding` — the donor's silu_mul epilogue.
	#[allow(clippy::too_many_arguments)]
	fn emit_gated_epilogue(
		&self,
		code: &mut String,
		gate_acc: usize,
		up_acc: usize,
		bias_binding: usize,
		linear_binding: usize,
		act_binding: usize,
		linear_stride: Extent,
		extents: (Extent, Extent, Extent, Extent),
		bias_origin: Option<usize>,
		induction: Option<usize>,
		pad: &str,
	) -> Result<()> {
		let (rows, columns) = self.register_tile(gate_acc)?;
		self.register_tile(up_acc)?;
		let row_origin = self.extent(extents.0, induction)?;
		let column_origin = self.extent(extents.1, induction)?;
		let row_extent = self.extent(extents.2, induction)?;
		let gate_width = self.extent(extents.3, induction)?;
		let linear_stride = self.extent(linear_stride, induction)?;
		let bias_index = |suffix: &str| -> Result<String> {
			Ok(match bias_origin {
				Some(origin) => format!(
					"{} + output_column{suffix}",
					self.lane_u32(origin, induction)?
				),
				None => format!("output_column{suffix}"),
			})
		};
		let gate_bias_index = bias_index("")?;
		let up_bias_index = bias_index(&format!(" + {gate_width}"))?;
		for row in 0..rows {
			for column in 0..columns {
				let gate_ref = self.register_ref(gate_acc, format!("{row}u * {columns}u + {column}u"))?;
				let up_ref = self.register_ref(up_acc, format!("{row}u * {columns}u + {column}u"))?;
				code.push_str(&format!(
					"{pad}{{\n\
					 {pad}\tuint output_row = {row_origin} + thread_row * {rows}u + {row}u;\n\
					 {pad}\tuint output_column = {column_origin} + thread_column * {columns}u + {column}u;\n\
					 {pad}\tif (output_row < {row_extent} && output_column < {gate_width}) {{\n\
					 {pad}\t\tuint element = output_row * {linear_stride} + output_column;\n\
					 {pad}\t\tfloat linear_value = {gate_ref}\n\
					 {pad}\t\t\t+ asfloat(storage_buffers[push.buffer{bias_binding}].Load(({gate_bias_index}) * 4u));\n\
					 {pad}\t\tstorage_buffers[push.buffer{linear_binding}].Store(element * 4u, asuint(linear_value));\n\
					 {pad}\t\tfloat up_value = {up_ref}\n\
					 {pad}\t\t\t+ asfloat(storage_buffers[push.buffer{bias_binding}].Load(({up_bias_index}) * 4u));\n\
					 {pad}\t\tstorage_buffers[push.buffer{linear_binding}].Store((element + {gate_width}) * 4u, asuint(up_value));\n\
					 {pad}\t\tstorage_buffers[push.buffer{act_binding}].Store((output_row * {gate_width} + output_column) * 4u, asuint(tile_swiglu(linear_value, up_value)));\n\
					 {pad}\t}}\n\
					 {pad}}}\n"
				));
			}
		}
		Ok(())
	}

	/// `true` when the op tree stores a `linear_gated_act` epilogue —
	/// the emitter then spells `tile_sigmoid`/`tile_swiglu` helpers
	/// ahead of `main`.
	fn gated_epilogue(&self, ops: &[TileOp]) -> Result<bool> {
		for op in ops {
			let found = match op {
				TileOp::StoreTileGated { .. } => true,
				TileOp::Loop { body, .. }
				| TileOp::LaneLoop { body, .. }
				| TileOp::LaneGuard { body, .. } => self.gated_epilogue(body)?,
				_ => false,
			};
			if found {
				return Ok(true);
			}
		}
		Ok(false)
	}

	/// `(upstream_binding, saved_binding)` when the op tree stages a
	/// fused-prologue tile — the emitter then spells the donor's
	/// `apply_prologue`/`prologue_value` helpers once ahead of `main`.
	fn prologue_epilogue(&self, ops: &[TileOp]) -> Result<Option<(usize, usize)>> {
		for op in ops {
			let found = match op {
				TileOp::LoadSharedPrologue {
					upstream, saved, ..
				} => Some((*upstream, *saved)),
				TileOp::Loop { body, .. }
				| TileOp::LaneLoop { body, .. }
				| TileOp::LaneGuard { body, .. } => self.prologue_epilogue(body)?,
				_ => None,
			};
			if found.is_some() {
				return Ok(found);
			}
		}
		Ok(None)
	}

	fn register_tile(&self, index: usize) -> Result<(u32, u32)> {
		match self.tiles.get(index) {
			Some(TileDecl::Register { rows, columns }) => Ok((*rows, *columns)),
			_ => Err(Error::invalid_argument(
				"tile op references a missing or non-register tile",
			)),
		}
	}

	fn register_u32_tile(&self, index: usize) -> Result<u32> {
		match self.tiles.get(index) {
			Some(TileDecl::RegisterU32 { elements }) => Ok(*elements),
			_ => Err(Error::invalid_argument(
				"lane op references a missing u32 register tile",
			)),
		}
	}

	/// Element spelling for a register tile: one-element tiles emit a
	/// bare scalar (`tile{N}`) so the generated accumulator stays an
	/// SSA-promotable local instead of a `float[1]` forced into
	/// function storage; larger tiles keep array element syntax.
	fn register_ref(&self, tile: usize, index: impl std::fmt::Display) -> Result<String> {
		let (rows, columns) = self.register_tile(tile)?;
		if rows * columns == 1 {
			Ok(format!("tile{tile}"))
		} else {
			Ok(format!("tile{tile}[{index}]"))
		}
	}

	fn shared_u32_tile(&self, index: usize) -> Result<u32> {
		match self.tiles.get(index) {
			Some(TileDecl::SharedU32 { elements }) => Ok(*elements),
			_ => Err(Error::invalid_argument(
				"lane op references a missing u32 shared tile",
			)),
		}
	}

	fn shared_tile(&self, index: usize) -> Result<(u32, u32, u32)> {
		match self.tiles.get(index) {
			Some(TileDecl::Shared {
				rows,
				columns,
				stride,
			}) => Ok((*rows, *columns, *stride)),
			_ => Err(Error::invalid_argument(
				"tile op references a missing or non-shared tile",
			)),
		}
	}

	/// The expression node's own implicit type; operand agreement is
	/// checked by each parent's render.
	fn lane_type(&self, expr: usize) -> Result<LaneType> {
		match self
			.lane_exprs
			.get(expr)
			.ok_or_else(|| Error::invalid_argument("lane expression index out of range"))?
		{
			LaneExpr::GroupX
			| LaneExpr::GroupY
			| LaneExpr::GroupZ
			| LaneExpr::Lane
			| LaneExpr::Push(_)
			| LaneExpr::Induction
			| LaneExpr::ConstU32(_)
			| LaneExpr::AddU32(..)
			| LaneExpr::SubU32(..)
			| LaneExpr::MulU32(..)
			| LaneExpr::MinU32(..)
			| LaneExpr::MaxU32(..)
			| LaneExpr::LoopVar(..)
			| LaneExpr::DivU32(..)
			| LaneExpr::ModU32(..)
			| LaneExpr::BufferU32 { .. }
			| LaneExpr::BufferU8 { .. }
			| LaneExpr::SharedU32 { .. }
			| LaneExpr::RegisterU32 { .. }
			| LaneExpr::SelectU32 { .. }
			| LaneExpr::WaveMinU32(..)
			| LaneExpr::WaveLaneCount
			| LaneExpr::ShrU32 { .. }
			| LaneExpr::BorderCoordU32 { .. }
			| LaneExpr::LoadU32 { .. }
			| LaneExpr::PushBitsU32(..)
			| LaneExpr::U32FromF32(..) => Ok(LaneType::U32),
			LaneExpr::Alias { source } => self.lane_type(*source),
			LaneExpr::LtU32(..)
			| LaneExpr::GeU32(..)
			| LaneExpr::NeqU32(..)
			| LaneExpr::EqU32(..)
			| LaneExpr::GtF32(..)
			| LaneExpr::EqF32(..)
			| LaneExpr::LtF32(..)
			| LaneExpr::GeF32(..)
			| LaneExpr::OrBool(..)
			| LaneExpr::AndBool(..) => Ok(LaneType::Bool),
			_ => Ok(LaneType::F32),
		}
	}

	fn lane_u32(&self, expr: usize, induction: Option<usize>) -> Result<String> {
		if self.lane_type(expr)? != LaneType::U32 {
			return Err(Error::invalid_argument(
				"u32 lane operand given an f32 expression",
			));
		}
		self.lane_expr(expr, induction)
	}

	fn lane_f32(&self, expr: usize, induction: Option<usize>) -> Result<String> {
		if self.lane_type(expr)? != LaneType::F32 {
			return Err(Error::invalid_argument(
				"f32 lane operand given a u32 expression",
			));
		}
		self.lane_expr(expr, induction)
	}

	fn lane_bool(&self, expr: usize, induction: Option<usize>) -> Result<String> {
		if self.lane_type(expr)? != LaneType::Bool {
			return Err(Error::invalid_argument(
				"bool lane operand given a non-comparison expression",
			));
		}
		self.lane_expr(expr, induction)
	}

	/// Render one lane expression inline at its use site. `induction` is
	/// the innermost enclosing lane/tile loop's variable number.
	fn lane_expr(&self, expr: usize, induction: Option<usize>) -> Result<String> {
		let node = self
			.lane_exprs
			.get(expr)
			.ok_or_else(|| Error::invalid_argument("lane expression index out of range"))?;
		Ok(match node {
			LaneExpr::GroupX => "group_id.x".into(),
			LaneExpr::GroupY => "group_id.y".into(),
			LaneExpr::GroupZ => "group_id.z".into(),
			LaneExpr::Lane => "lane".into(),
			LaneExpr::Push(index) => format!(
				"push.{}",
				self.push_scalars.get(*index).ok_or_else(|| {
					Error::invalid_argument("lane expression references a missing push scalar")
				})?
			),
			LaneExpr::Induction => format!(
				"loop{}",
				induction.ok_or_else(|| {
					Error::invalid_argument("induction lane expression used outside a lane loop")
				})?
			),
			LaneExpr::ConstU32(value) => format!("{value}u"),
			LaneExpr::ConstF32(bits) => format!("asfloat(0x{bits:08x}u)"),
			LaneExpr::AddU32(a, b) => {
				format!(
					"({} + {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::SubU32(a, b) => {
				format!(
					"({} - {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::MinU32(a, b) => {
				format!(
					"min({}, {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::MaxU32(a, b) => {
				format!(
					"max({}, {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::LoopVar(depth) => format!("loop{depth}"),
			LaneExpr::GtF32(a, b) => {
				format!(
					"({} > {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::EqF32(a, b) => {
				format!(
					"({} == {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::WaveMax(operand) => {
				format!("WaveActiveMax({})", self.lane_f32(*operand, induction)?)
			}
			LaneExpr::WaveSum(operand) => {
				format!("WaveActiveSum({})", self.lane_f32(*operand, induction)?)
			}
			LaneExpr::WaveMinU32(operand) => {
				format!("WaveActiveMin({})", self.lane_u32(*operand, induction)?)
			}
			LaneExpr::WaveLaneCount => "WaveGetLaneCount()".to_string(),
			LaneExpr::SelectU32 {
				cond,
				if_true,
				if_false,
			} => {
				format!(
					"({} ? {} : {})",
					self.lane_bool(*cond, induction)?,
					self.lane_u32(*if_true, induction)?,
					self.lane_u32(*if_false, induction)?
				)
			}
			LaneExpr::SelectF32 {
				cond,
				if_true,
				if_false,
			} => {
				format!(
					"({} ? {} : {})",
					self.lane_bool(*cond, induction)?,
					self.lane_f32(*if_true, induction)?,
					self.lane_f32(*if_false, induction)?
				)
			}
			LaneExpr::SharedU32 { tile, offset } => {
				self.shared_u32_tile(*tile)?;
				format!("shared{tile}[{}]", self.lane_u32(*offset, induction)?)
			}
			LaneExpr::RegisterU32 { tile, index } => {
				let elements = self.register_u32_tile(*tile)?;
				if *index >= elements {
					return Err(Error::invalid_argument(
						"lane expression exceeds its u32 register tile",
					));
				}
				if elements == 1 {
					format!("tile{tile}")
				} else {
					format!("tile{tile}[{index}u]")
				}
			}
			LaneExpr::LtU32(a, b) => {
				format!(
					"({} < {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::MulU32(a, b) => {
				format!(
					"({} * {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::DivU32(a, b) => {
				format!(
					"({} / {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::ModU32(a, b) => {
				format!(
					"({} % {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::SubF32(a, b) => {
				format!(
					"({} - {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::MulF32(a, b) => {
				format!(
					"({} * {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::DivF32(a, b) => {
				format!(
					"({} / {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::MaxF32(a, b) => {
				format!(
					"max({}, {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::MinF32(a, b) => {
				format!(
					"min({}, {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::AddF32(a, b) => {
				format!(
					"({} + {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::ExpF32(arg) => format!("exp({})", self.lane_f32(*arg, induction)?),
			LaneExpr::LogF32(arg) => format!("log({})", self.lane_f32(*arg, induction)?),
			LaneExpr::SqrtF32(arg) => format!("sqrt({})", self.lane_f32(*arg, induction)?),
			LaneExpr::RsqrtF32(arg) => format!("rsqrt({})", self.lane_f32(*arg, induction)?),
			LaneExpr::PowF32(base, exponent) => format!(
				"pow({}, {})",
				self.lane_f32(*base, induction)?,
				self.lane_f32(*exponent, induction)?
			),
			LaneExpr::LoadU32 { binding, address } => format!(
				"storage_buffers[push.buffer{binding}].Load({} * 4u)",
				self.lane_u32(*address, induction)?
			),
			LaneExpr::PushBitsU32(index) => format!(
				"asuint(push.{})",
				self.push_f32_scalars.get(*index).ok_or_else(|| {
					Error::invalid_argument("lane expression references a missing f32 push scalar")
				})?
			),
			LaneExpr::FloorF32(arg) => {
				format!("floor({})", self.lane_f32(*arg, induction)?)
			}
			LaneExpr::PushF32(index) => format!(
				"push.{}",
				self.push_f32_scalars.get(*index).ok_or_else(|| {
					Error::invalid_argument("lane expression references a missing f32 push scalar")
				})?
			),
			LaneExpr::F32FromU32(arg) => {
				format!("float({})", self.lane_u32(*arg, induction)?)
			}
			LaneExpr::U32FromF32(arg) => {
				format!("uint({})", self.lane_f32(*arg, induction)?)
			}
			LaneExpr::GeU32(a, b) => {
				format!(
					"({} >= {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::NeqU32(a, b) => {
				format!(
					"({} != {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::EqU32(a, b) => {
				format!(
					"({} == {})",
					self.lane_u32(*a, induction)?,
					self.lane_u32(*b, induction)?
				)
			}
			LaneExpr::BufferU32 { binding, index } => {
				format!(
					"storage_buffers[push.buffer{binding}].Load(({}) * 4u)",
					self.lane_u32(*index, induction)?
				)
			}
			LaneExpr::BufferU8 { binding, index } => {
				let index = self.lane_u32(*index, induction)?;
				format!(
					"((storage_buffers[push.buffer{binding}].Load(({index}) & ~3u) >> ((({index}) & 3u) * 8u)) & 255u)"
				)
			}
			LaneExpr::OrBool(a, b) => {
				format!(
					"({} || {})",
					self.lane_bool(*a, induction)?,
					self.lane_bool(*b, induction)?
				)
			}
			LaneExpr::AndBool(a, b) => {
				format!(
					"({} && {})",
					self.lane_bool(*a, induction)?,
					self.lane_bool(*b, induction)?
				)
			}
			LaneExpr::LoadF32 { binding, address } => format!(
				"asfloat(storage_buffers[push.buffer{binding}].Load({} * 4u))",
				self.lane_u32(*address, induction)?
			),
			LaneExpr::Alias { source } => self.lane_expr(*source, induction)?,
			LaneExpr::SharedF32 { tile, offset } => {
				self.shared_tile(*tile)?;
				format!("shared{tile}[{}]", self.lane_u32(*offset, induction)?)
			}
			LaneExpr::RegisterF32 { tile, index } => {
				let (rows, columns) = self.register_tile(*tile)?;
				if *index >= rows * columns {
					return Err(Error::invalid_argument(
						"lane expression exceeds its register tile",
					));
				}
				if rows * columns == 1 {
					format!("tile{tile}")
				} else {
					format!("tile{tile}[{index}u]")
				}
			}
			LaneExpr::ShrU32 { value, bits } => {
				format!("({} >> {bits}u)", self.lane_u32(*value, induction)?)
			}
			LaneExpr::LtF32(a, b) => {
				format!(
					"({} < {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::GeF32(a, b) => {
				format!(
					"({} >= {})",
					self.lane_f32(*a, induction)?,
					self.lane_f32(*b, induction)?
				)
			}
			LaneExpr::PlaneSample {
				binding,
				x,
				y,
				width,
				height,
				components,
				component,
				interp,
				sample_bits,
			} => format!(
				"to_matrix_sample_component(storage_buffers[push.buffer{binding}], {}, {}, {}, {}, {components}u, {component}u, {}, {sample_bits}u)",
				self.lane_f32(*x, induction)?,
				self.lane_f32(*y, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?,
				self.lane_u32(*interp, induction)?
			),
			LaneExpr::ScalePlane {
				kind,
				code,
				color_mode,
				sample_bits,
			} => {
				let name = match (kind, *sample_bits) {
					(PlaneScale::Luma, 8) => "to_matrix_scale_luma",
					(PlaneScale::Chroma, 8) => "to_matrix_scale_chroma",
					(PlaneScale::Luma, 10) => "to_matrix_scale_p010_luma",
					(PlaneScale::Chroma, 10) => "to_matrix_scale_p010_chroma",
					_ => {
						return Err(Error::invalid_argument(
							"plane scale admits 8- or 10-bit samples",
						));
					}
				};
				format!(
					"{name}({}, {})",
					self.lane_f32(*code, induction)?,
					self.lane_u32(*color_mode, induction)?
				)
			}
			LaneExpr::YuvToRgb {
				luma,
				cb,
				cr,
				space,
				component,
			} => format!(
				"yuv_to_rgb(float3({}, {}, {}), {})[{}]",
				self.lane_f32(*luma, induction)?,
				self.lane_f32(*cb, induction)?,
				self.lane_f32(*cr, induction)?,
				self.lane_u32(*space, induction)?,
				self.lane_u32(*component, induction)?
			),
			LaneExpr::BilinearF32 {
				binding,
				base,
				xf,
				yf,
				width,
				height,
			} => format!(
				"image_bilinear_f32(storage_buffers[push.buffer{binding}], {}, {}, {}, {}, {})",
				self.lane_u32(*base, induction)?,
				self.lane_f32(*xf, induction)?,
				self.lane_f32(*yf, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?
			),
			LaneExpr::PadFetchF32 {
				binding,
				base,
				x,
				y,
				pad_left,
				pad_top,
				width,
				height,
				mode,
				value,
			} => format!(
				"image_pad_fetch(storage_buffers[push.buffer{binding}], {}, {}, {}, {}, {}, {}, {}, {}, {})",
				self.lane_u32(*base, induction)?,
				self.lane_u32(*x, induction)?,
				self.lane_u32(*y, induction)?,
				self.lane_u32(*pad_left, induction)?,
				self.lane_u32(*pad_top, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?,
				self.lane_u32(*mode, induction)?,
				self.lane_f32(*value, induction)?
			),
			LaneExpr::BorderCoordU32 {
				coordinate,
				extent,
				mode,
			} => format!(
				"(uint)(image_border_coordinate(int({}), {}, {}))",
				self.lane_u32(*coordinate, induction)?,
				self.lane_u32(*extent, induction)?,
				self.lane_u32(*mode, induction)?
			),
			LaneExpr::BorderedFetchF32 {
				binding,
				base,
				x,
				y,
				width,
				height,
				mode,
				value,
			} => format!(
				"image_sample_bordered(storage_buffers[push.buffer{binding}], {}, int({}), int({}), {}, {}, {}, {})",
				self.lane_u32(*base, induction)?,
				self.lane_u32(*y, induction)?,
				self.lane_u32(*x, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?,
				self.lane_u32(*mode, induction)?,
				self.lane_f32(*value, induction)?
			),
			LaneExpr::WarpF32 {
				input,
				transform,
				base,
				x,
				y,
				width,
				height,
				interp,
				mode,
				value,
				perspective,
			} => format!(
				"image_warp_f32(storage_buffers[push.buffer{input}], storage_buffers[push.buffer{transform}], {}, {}, {}, {}, {}, {}, {}, {}, {}u)",
				self.lane_u32(*base, induction)?,
				self.lane_u32(*x, induction)?,
				self.lane_u32(*y, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?,
				self.lane_u32(*interp, induction)?,
				self.lane_u32(*mode, induction)?,
				self.lane_f32(*value, induction)?,
				u32::from(*perspective)
			),
			LaneExpr::RemapF32 {
				input,
				map,
				base,
				x,
				y,
				batch,
				width,
				height,
				out_width,
				out_height,
				interp,
				mode,
				map_batch,
				value,
			} => format!(
				"image_remap_f32(storage_buffers[push.buffer{input}], storage_buffers[push.buffer{map}], {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
				self.lane_u32(*base, induction)?,
				self.lane_u32(*x, induction)?,
				self.lane_u32(*y, induction)?,
				self.lane_u32(*batch, induction)?,
				self.lane_u32(*width, induction)?,
				self.lane_u32(*height, induction)?,
				self.lane_u32(*out_width, induction)?,
				self.lane_u32(*out_height, induction)?,
				self.lane_u32(*interp, induction)?,
				self.lane_u32(*mode, induction)?,
				self.lane_u32(*map_batch, induction)?,
				self.lane_f32(*value, induction)?
			),
		})
	}

	/// Whether any lane expression needs the image gather helpers —
	/// the typed bilinear plane fetch shared by geometric lowerings.
	fn needs_image_helpers(&self) -> bool {
		self.lane_exprs.iter().any(|expr| {
			matches!(
				expr,
				LaneExpr::BilinearF32 { .. }
					| LaneExpr::PadFetchF32 { .. }
					| LaneExpr::BorderedFetchF32 { .. }
					| LaneExpr::WarpF32 { .. }
					| LaneExpr::RemapF32 { .. }
			)
		})
	}

	/// Whether any lane expression needs the `to_matrix` video helpers —
	/// typed plane samples, range expansion, or YUV→RGB conversion.
	fn needs_video_helpers(&self) -> bool {
		self.lane_exprs.iter().any(|expr| {
			matches!(
				expr,
				LaneExpr::PlaneSample { .. } | LaneExpr::ScalePlane { .. } | LaneExpr::YuvToRgb { .. }
			)
		})
	}

	/// Projection-`group_id.z` binding select. Core-profile push words hold
	/// arbitrary heap descriptor indices, so the donor ternary over three
	/// consecutive fields is required — `push.buffer{base} + group_id.z`
	/// would index an unrelated descriptor.
	fn z_binding(&self, base: usize) -> Result<String> {
		if base + 3 > self.bindings.len() {
			return Err(Error::invalid_argument(
				"z-selected binding requires three consecutive buffer fields",
			));
		}
		Ok(format!(
			"(group_id.z == 0u ? push.buffer{base} : group_id.z == 1u ? push.buffer{} : push.buffer{})",
			base + 1,
			base + 2
		))
	}
}
