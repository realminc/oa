//! Typed tile-level IR (OA-CC-11), lowered deterministically to Slang.
//!
//! The IR separates a block program's structure — index space, descriptor
//! bindings, and ordered tile ops — from the emitted shader text. This
//! first slice carries the vocabulary the elementwise pointwise template
//! needs; shared-memory staging, barriers, subgroup ops, `dot`, row/column
//! reduce and scan arrive with the templates that consume them.

use crate::runtime::dna::ir::elementwise::{Program, Source};
use crate::{Error, Result};

/// One descriptor slot's role in the template ABI. Slot order is the
/// elementwise convention: inputs then outputs, dense push-constant
/// indices — the artifact ABI is unchanged by the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum Binding {
	Input(usize),
	Output(usize),
	/// In-place read-modify-write site (e.g. the `conv_1d_bias_add`
	/// values buffer). Occupies one positional descriptor slot and
	/// counts as neither a leading input nor a tail output.
	ReadWrite(usize),
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

/// Plane-code range expansion applied by `LaneExpr::ScalePlane` — the
/// `to_matrix_scale_{,p010_}{luma,chroma}` donor helpers' two modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum PlaneScale {
	/// Range-expand a normalized code to `[0,1]` luma.
	Luma,
	/// Range-expand a normalized code to zero-centered chroma.
	Chroma,
}

/// Cooperative fold operation applied by `LaneFold` and `TreeReduce`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum LaneReduce {
	Sum,
	Max,
	Min,
}

/// The `mat_mul_nt_bias*` donors' fixed epilogue activation — spelled
/// with each donor's literal constants so the emitted store is
/// bit-identical, unlike the `apply_act` push-dispatch helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum BiasAct {
	/// `max(0.0f, value)`
	Relu,
	/// `0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value³)))`
	Gelu,
	/// `value * (1.0f / (1.0f + exp(-value)))`
	Silu,
}

/// Conv1d bias-transpose tail epilogue selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvTailStore {
	/// Store the biased tile value.
	Plain,
	/// Store `max(0.0f, value)`.
	Relu,
	/// Store the activated value plus `activated + residual` on the
	/// extra `sum` binding.
	ResidualRelu,
}

/// A per-lane scalar expression (block programs only). Expressions form
/// a flat SSA DAG inside `BlockProgram::lane_exprs` — indices are
/// operands; the emitter renders each expression inline at its use
/// site, so a value used inside a lane loop is re-evaluated per
/// iteration exactly like the donor's local temporaries.
///
/// Every variant has an implicit type: `Lane`, `GroupX`, `Push`,
/// `Induction`, `ConstU32`, and the `*U32` arithmetic are `u32`;
/// `ConstF32`, the `*F32` arithmetic, `LoadF32`, `SharedF32`, and
/// `RegisterF32` are `f32`. Operand typing is validated at emission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum LaneExpr {
	/// `group_id.x` (u32).
	GroupX,
	/// `group_id.y` (u32).
	GroupY,
	/// `group_id.z` (u32) — the expert axis of the grouped
	/// parameter-adjoint grid.
	GroupZ,
	/// `group_thread_id.x` (u32).
	Lane,
	/// `push.{push_scalars[index]}` (u32).
	Push(usize),
	/// The innermost enclosing lane loop's element index (u32).
	Induction,
	/// `uint(expr)` — truncate an integral `f32` register value back to
	/// `u32` (expert indices staged through float register tiles).
	U32FromF32(usize),
	/// A `u32` literal.
	ConstU32(u32),
	/// An `f32` literal bit pattern.
	ConstF32(u32),
	AddU32(usize, usize),
	SubU32(usize, usize),
	MulU32(usize, usize),
	MinU32(usize, usize),
	MaxU32(usize, usize),
	/// `loop{depth}` — an enclosing `TileOp::Loop` induction variable at
	/// an absolute emit depth (1 = outermost). Unlike `Induction`, which
	/// binds the innermost loop, this reaches the outer expert-segment
	/// and column-group counters of the fused backward prologue.
	LoopVar(u32),
	/// `expr_a < expr_b` — a u32 comparison rendered as `bool`.
	LtU32(usize, usize),
	/// `expr_a >= expr_b` — a u32 comparison rendered as `bool`.
	GeU32(usize, usize),
	/// `expr_a != expr_b` — a u32 comparison rendered as `bool`.
	NeqU32(usize, usize),
	/// `lhs == rhs` — u32 comparison producing a bool lane value.
	EqU32(usize, usize),
	/// `storage_buffers[push.buffer{binding}].Load(index * 4u)` — a
	/// uniform u32 read of a scalar table element; the parameter
	/// adjoint's `begin`/`end` expert range from the offsets buffer.
	BufferU32 {
		binding: usize,
		index: usize,
	},
	/// `load_u8` byte extraction —
	/// `(Load(index & ~3u) >> ((index & 3u) * 8u)) & 0xffu` — the packed
	/// index-table read the u8 embedding adjoint performs per position.
	BufferU8 {
		binding: usize,
		index: usize,
	},
	/// `expr_a || expr_b` — a boolean disjunction.
	OrBool(usize, usize),
	/// `expr_a && expr_b` — a boolean conjunction.
	AndBool(usize, usize),
	/// `expr_a > expr_b` — an f32 comparison rendered as `bool`.
	GtF32(usize, usize),
	/// `expr_a == expr_b` — an f32 comparison rendered as `bool`.
	EqF32(usize, usize),
	/// `WaveActiveMax(operand)` — an f32 subgroup arithmetic reduce;
	/// order-insensitive, so the result is identical to any serial
	/// maximum. Requires the queried subgroup arithmetic capability.
	WaveMax(usize),
	/// `WaveActiveSum(operand)` — an f32 subgroup arithmetic reduce.
	/// Unlike `WaveMax` this changes reduction order versus the
	/// shared tree, so admission is an explicit opt-in candidate and
	/// qualification uses a tolerance oracle, not incumbent bits.
	/// Requires the queried subgroup arithmetic capability.
	WaveSum(usize),
	/// `WaveActiveMin(operand)` — a `u32` subgroup arithmetic reduce;
	/// order-insensitive. Paired with `WaveMax` it reproduces the
	/// argmax tie-break: the minimum index among lanes holding the
	/// subgroup maximum. Requires the queried subgroup arithmetic
	/// capability.
	WaveMinU32(usize),
	/// `WaveGetLaneCount()` — the device's runtime subgroup width, a
	/// `u32`. Baking it into a constant would make the artifact
	/// device-specific; the intrinsic keeps one program portable.
	WaveLaneCount,
	/// `cond ? a : b` — a `u32` select over a bool `LaneExpr`.
	SelectU32 {
		cond: usize,
		if_true: usize,
		if_false: usize,
	},
	/// `cond ? a : b` — an `f32` select over a bool `LaneExpr`.
	SelectF32 {
		cond: usize,
		if_true: usize,
		if_false: usize,
	},
	/// `shared{tile}[offset]` — a u32 shared-tile element read.
	SharedU32 {
		tile: usize,
		offset: usize,
	},
	/// `tile{tile}[index]` — a per-lane u32 register element read.
	RegisterU32 {
		tile: usize,
		index: u32,
	},
	DivU32(usize, usize),
	ModU32(usize, usize),
	AddF32(usize, usize),
	SubF32(usize, usize),
	MulF32(usize, usize),
	DivF32(usize, usize),
	MaxF32(usize, usize),
	MinF32(usize, usize),
	ExpF32(usize),
	LogF32(usize),
	SqrtF32(usize),
	RsqrtF32(usize),
	/// `pow(base, exponent)` — the `pointwise` donor's gamma branch.
	PowF32(usize, usize),
	/// `floor(value)` — the `pointwise` donor's posterize branch.
	FloorF32(usize),
	/// `push.{push_f32_scalars[index]}` — an f32 push scalar.
	PushF32(usize),
	/// `asuint(push.{push_f32_scalars[index]})` — a u32 push word
	/// stored in an f32 slot (the `resize_normalize` donor's trailing
	/// `swap_rb` word shares the f32 push tail).
	PushBitsU32(usize),
	/// `expr >> bits` — a u32 right shift (the `color_mode >> 1` color-space
	/// select in the video conversion donors).
	ShrU32 {
		value: usize,
		bits: u32,
	},
	/// `expr_a < expr_b` — an f32 comparison rendered as `bool`.
	LtF32(usize, usize),
	/// `expr_a >= expr_b` — an f32 comparison rendered as `bool`.
	GeF32(usize, usize),
	/// `to_matrix_sample_component(plane, x, y, width, height, components,
	/// component, interp, sample_bits)` — the typed plane load: bilinear
	/// (`interp == 0`) or nearest sampling of one component plane with
	/// pixel-center clamped coordinates. `components`/`component` spell
	/// 4:2:0 addressing — interleaved UV planes take `components = 2` with
	/// component 0/1; planar chroma takes `components = 1`. `sample_bits`
	/// selects u8 codes (8) or 10-in-16 lanes (10).
	PlaneSample {
		/// Plane descriptor slot.
		binding: usize,
		/// Continuous source position (f32 lane expressions).
		x: usize,
		y: usize,
		/// Plane element extent (u32 lane expressions).
		width: usize,
		height: usize,
		components: u32,
		component: u32,
		/// u32 lane expression — the interpolation mode.
		interp: usize,
		sample_bits: u32,
	},
	/// `to_matrix_scale_{,p010_}{luma,chroma}(code, color_mode)` — range
	/// expansion of one normalized plane code under the frame's color range.
	ScalePlane {
		kind: PlaneScale,
		/// f32 lane expression — the normalized code value.
		code: usize,
		/// u32 lane expression — the packed color mode.
		color_mode: usize,
		sample_bits: u32,
	},
	/// `yuv_to_rgb(float3(luma, cb, cr), space)[component]` — one channel
	/// of the donor's YUV→RGB conversion.
	YuvToRgb {
		luma: usize,
		cb: usize,
		cr: usize,
		/// u32 lane expression — the color-space index.
		space: usize,
		/// u32 lane expression — the component to extract.
		component: usize,
	},
	/// `image_bilinear_f32(buffer, base, xf, yf, width, height)` — the
	/// gather anchor (OA-CC-33): a clamped bilinear fetch over a packed
	/// f32 image plane, shared by resize, flip, pad, remap, and warp
	/// lowerings. `base` is the plane offset; `xf`/`yf` continuous source
	/// coordinates; `width`/`height` the source plane extent.
	BilinearF32 {
		binding: usize,
		base: usize,
		xf: usize,
		yf: usize,
		width: usize,
		height: usize,
	},
	/// `image_pad_fetch(buffer, base, x, y, pad_left, pad_top, width,
	/// height, mode, border_value)` — the pad donor's signed-offset
	/// bordered fetch (OA-CC-33).
	PadFetchF32 {
		binding: usize,
		base: usize,
		x: usize,
		y: usize,
		pad_left: usize,
		pad_top: usize,
		width: usize,
		height: usize,
		mode: usize,
		value: usize,
	},
	/// `image_warp_f32(input, transform, base, x, y, width, height,
	/// interp, mode, border_value, perspective)` — the warp donors'
	/// matrix coordinate + bordered gather.
	WarpF32 {
		input: usize,
		transform: usize,
		base: usize,
		x: usize,
		y: usize,
		width: usize,
		height: usize,
		interp: usize,
		mode: usize,
		value: usize,
		perspective: bool,
	},
	/// `image_remap_f32(input, map, base, x, y, batch, width, height,
	/// out_width, out_height, interp, mode, map_batch, border_value)` —
	/// the remap donor's map-fed bordered gather.
	RemapF32 {
		input: usize,
		map: usize,
		base: usize,
		x: usize,
		y: usize,
		batch: usize,
		width: usize,
		height: usize,
		out_width: usize,
		out_height: usize,
		interp: usize,
		mode: usize,
		map_batch: usize,
		value: usize,
	},
	/// `image_sample_bordered(buffer, base, y, x, width, height, mode,
	/// border_value)` — the filter donors' per-tap bordered fetch: the
	/// coordinate folds through the border mode and out-of-range taps
	/// take the fill value (OA-CC-33 filter anchor). `x`/`y` are u32
	/// lane expressions cast to `int` — two's-complement wraparound
	/// preserves the signed offsets the border fold expects.
	BorderedFetchF32 {
		binding: usize,
		base: usize,
		x: usize,
		y: usize,
		width: usize,
		height: usize,
		mode: usize,
		value: usize,
	},
	/// `image_border_coordinate(int(coordinate), extent, mode)` — the
	/// border fold as a u32 lane value; a -1 (constant-mode miss)
	/// arrives as two's-complement `0xFFFFFFFF`.
	BorderCoordU32 {
		coordinate: usize,
		extent: usize,
		mode: usize,
	},
	/// `float(expr)` — u32 to f32 conversion.
	F32FromU32(usize),
	/// `load_f32(storage_buffers[binding], address)` — an f32 element
	/// load at a u32 element address.
	LoadF32 {
		binding: usize,
		address: usize,
	},
	/// `storage_buffers[push.buffer{binding}].Load(address * 4u)` — the
	/// raw u32 lane read used by the segmentation mask's i32 labels.
	LoadU32 {
		binding: usize,
		address: usize,
	},
	/// Another lane expression's value under this index — emitted
	/// inline as the source's expression text; lets a rewritten node
	/// keep its index identity while resolving to a folded chain.
	Alias {
		source: usize,
	},
	/// `shared{tile}[offset]` — an f32 shared-tile element read at a
	/// u32 offset expression.
	SharedF32 {
		tile: usize,
		offset: usize,
	},
	/// `tile_swiglu(gate, up)` — the donor's `silu(gate) * up` helper;
	/// requires the gated-epilogue helper preamble (emitted whenever
	/// `tile{tile}[index]` — a per-lane register element read.
	RegisterF32 {
		tile: usize,
		index: u32,
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
	/// Runtime-extent counted loop over `0..extent` stepping `step`; the
	/// induction variable is referenced by `Extent::Induction`.
	Loop {
		extent: Extent,
		step: u32,
		/// `None` counts `0..extent`; `Some` ranges `start..extent` —
		/// the parameter adjoint's `begin..end` expert row range.
		start: Option<Extent>,
		body: Vec<TileOp>,
	},
	/// Zero a per-lane register tile.
	ClearTile { tile: usize },
	/// Masked cooperative load of a `Shared` tile from a dense binding:
	/// every lane stages `vec` consecutive elements per iteration, with a
	/// vectorized fast path when the whole `vec` is in bounds and a
	/// scalar masked tail otherwise. `columns % vec == 0` is validated at
	/// build time so a `vec` group never crosses a row.
	LoadShared {
		tile: usize,
		binding: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
		vec: u32,
		/// Extra term added to the emitted `global_row` —
		/// `linear_gated_act`'s up-half rows sit at `row_origin +
		/// gate_width`.
		row_offset: Option<Extent>,
		/// Alternate row coordinate for the bounds mask — the grouped
		/// route masks on the tile-local column (`column < gate_width`)
		/// while the load address carries the expert-row offset.
		mask_origin: Option<Extent>,
		/// `true` selects `push.buffer{binding + group_id.z}` by ternary —
		/// the `linear_triple` z-dispatch buffer selector. Descriptor push
		/// words hold arbitrary heap indices, so this is a field select,
		/// never index arithmetic.
		z: bool,
		/// Extra term added to the emitted `global_column` in the load
		/// address only — the grouped backward weight's expert rows sit
		/// at `reduction + first_expert * outputs` while the mask keeps
		/// the raw reduction coordinate.
		column_offset: Option<Extent>,
		/// `true` emits `column * stride + row` addressing — the
		/// transpose-in-load of the grouped data-backward weight tile.
		transposed: bool,
	},
	/// Per-element concat-gather staging for `linear_concat`: each staged
	/// element selects one of three part bindings by k range and offsets
	/// into it by that part's own row stride. The gathered value is also
	/// stored into the `materialize` binding — restricted by the donor
	/// to the `group_id.y == 0` column tile.
	LoadSharedConcat {
		tile: usize,
		/// Part bindings ordered by k range.
		bindings: [usize; 3],
		/// Push-scalar indices of the three part widths.
		widths: [usize; 3],
		/// Concat output binding; stored only on `group_id.y == 0`.
		materialize: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// `GroupMemoryBarrierWithGroupSync` between a shared write phase and
	/// its readers; placement is template-owned and group-uniform.
	Barrier,
	/// `acc[r,c] += lhs[r,k] * rhs[c,k]` over the shared tiles' common
	/// column extent for the lane's `fragment_rows × fragment_columns`
	/// register block. Tails are pre-masked to zero at load time.
	DotShared { acc: usize, lhs: usize, rhs: usize },
	/// `DotShared` plus the parameter adjoint's gated bias fold:
	/// `if (bias_gate) bias_acc[i] += left_fragment[i]` inside the
	/// fragment-row loop, matching the donor's dual-accumulator dot.
	DotSharedParam {
		acc: usize,
		lhs: usize,
		rhs: usize,
		bias_acc: usize,
		/// Bool lane expression: `thread_column == 0 && group_id.y == 0`.
		bias_gate: usize,
	},
	/// Masked store of a register tile to a dense binding at
	/// `(row_origin + r, column_origin + c)` for every fragment element.
	StoreTile {
		acc: usize,
		binding: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
		/// `Some` masks on `mask_origin + local` instead of the store
		/// coordinate — the parameter adjoint's weight tile carries the
		/// expert offset on `row_origin` while masking the bare feature.
		mask_origin: Option<Extent>,
	},
	/// Gated scalar-column store for the parameter adjoint's bias
	/// gradient: under `gate`, every fragment row writes
	/// `binding[origin + output_feature]` where `output_feature =
	/// row_origin + fragment` masks against `row_extent`.
	/// `grouped_backward_prologue` cooperative staging of the fused
	/// producer output: each lane strides the `rows x columns` tile,
	/// evaluates `tile_prologue_value(row, feature)` (the donor's
	/// `prologue_value` — activation/pointwise adjoint over the upstream
	/// and saved buffers) under the row/column mask, stages the result,
	/// and materializes `dy` once per element behind `gate` — the
	/// `expert == first_expert && column_base == 0` term that gives every
	/// `dy` element exactly one writer across segments and column groups.
	LoadSharedPrologue {
		tile: usize,
		/// Upstream-gradient binding (helper input).
		upstream: usize,
		/// Saved-value binding (helper input).
		saved: usize,
		/// `dy` output binding materialized behind `gate`.
		dy: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
		/// `LaneExpr` gate for the `dy` store (`loop1`/`loop2` terms).
		gate: usize,
	},
	/// Cooperative staging of the fused layer-norm data-adjoint —
	/// `linear_backward_norm(_add)`'s `shared_output_gradient` load.
	/// Each lane strides the `rows x columns` tile and computes
	/// `adjoint = inv_stddev[row] * (upstream*gamma - stats.x*inv_cols
	/// - normalized*stats.y*inv_cols)` with `stats` read as the
	///   row-statistics shared pair at `local_row`. The staged value is
	///   `adjoint + addend` when `addend` is bound; `gate` materializes
	///   `norm_input_gradient` (the plain adjoint), `weight_contribution`
	///   (`upstream * normalized`) and `stage1` (the combined dy) so every
	///   republished element keeps exactly one writer.
	LoadSharedNorm {
		tile: usize,
		/// `normalized` input binding.
		normalized: usize,
		/// `inv_stddev` input binding.
		inv_stddev: usize,
		/// `norm_weight` (gamma) input binding.
		norm_weight: usize,
		/// `upstream_gradient` input binding.
		upstream: usize,
		/// Optional `addend` input binding (norm_add variant).
		addend: Option<usize>,
		/// `norm_input_gradient` output binding (adjoint store).
		norm_input_gradient: usize,
		/// `weight_contribution` output binding.
		weight_contribution: usize,
		/// Optional `stage1` output binding (combined dy store).
		stage1: Option<usize>,
		/// Shared f32 tile holding the `stats.x/stats.y` row pairs.
		stats: usize,
		/// f32 `LaneExpr` for `1.0f / output_features`.
		inverse_columns: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
		/// `LaneExpr` gate for the republished stores (`group_id.y == 0`).
		gate: usize,
	},
	/// Cooperative dot over `column_tiles` adjacent output tiles — the
	/// fused prologue's 64-row shared weight tile folded into one
	/// `[tm * tn * column_tiles]` accumulator. `lhs` is the staged
	/// gradient tile, `rhs` the wide weight tile; the accumulator layout
	/// is `acc[segment * tm * tn + i * tn + j]`.
	DotSharedWide {
		acc: usize,
		lhs: usize,
		rhs: usize,
		column_tiles: u32,
	},
	/// Segment-masked fragment store — the fused prologue's per-expert
	/// `input_gradient` write. Each element checks
	/// `row in [row_lo, row_hi)` (the current expert segment inside the
	/// row tile) plus the column bound before storing.
	StoreTileSegment {
		acc: usize,
		binding: usize,
		row_origin: Extent,
		row_lo: Extent,
		row_hi: Extent,
		column_origin: Extent,
		column_extent: Extent,
		column_tiles: u32,
	},
	StoreTileBiasParam {
		acc: usize,
		binding: usize,
		/// Expert-strided address base, e.g. `group_id.z * outputs`.
		origin: Extent,
		row_origin: Extent,
		row_extent: Extent,
		/// Bool lane expression: `thread_column == 0 && group_id.y == 0`.
		gate: usize,
	},
	/// Masked epilogue store for the `linear` donor: for every fragment
	/// element, `acc[r,c] + bias[column]` writes to `binding`.
	StoreTileBias {
		acc: usize,
		bias_binding: usize,
		binding: usize,
		/// `true` ternary-selects `push.buffer{binding + group_id.z}` on
		/// both the bias and output descriptor fields — `linear_triple`'s
		/// z-selected stores.
		z: bool,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
		/// Lane-expr `u32` added to the bias element index — the grouped
		/// donors' `first_expert * output_features` bias base.
		bias_origin: Option<usize>,
		/// Fixed activation applied to `acc + bias` at the single
		/// store — the `mat_mul_nt_bias*` donors' inline variants, as
		/// opposed to `StoreTileAct`'s separate `apply_act` output.
		act: Option<BiasAct>,
	},
	/// Masked epilogue store for the `linear_act` donor: for every
	/// fragment element, `linear = acc[r,c] + bias[column]` writes to
	/// `linear_binding`, and `act(linear)` (an `act_kind`/`alpha` push
	/// scalar dispatch) writes to `act_binding`. The emitter owns the
	/// `apply_act` helper text.
	StoreTileAct {
		acc: usize,
		bias_binding: usize,
		linear_binding: usize,
		act_binding: usize,
		kind: usize,
		alpha: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// Masked epilogue store for the `linear_residual` donor:
	/// `linear = acc + bias[column]` writes to `linear_binding`;
	/// `linear + residual[element]` writes to `sum_binding`. The
	/// residual is addressed by the flat output element, not the column.
	StoreTileResidual {
		acc: usize,
		bias_binding: usize,
		residual_binding: usize,
		linear_binding: usize,
		sum_binding: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// Masked epilogue store for the `linear_residual_act` donor — the
	/// residual epilogue plus `apply_act(sum)` into `act_binding` and a
	/// materialized `sum` store for autograd.
	StoreTileResidualAct {
		acc: usize,
		bias_binding: usize,
		residual_binding: usize,
		linear_binding: usize,
		sum_binding: usize,
		act_binding: usize,
		kind: usize,
		alpha: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// Masked epilogue store for the `linear_residual_norm` donor — the
	/// residual epilogue plus a `shared{tile}` write of `sum` at the
	/// local fragment position, staging every tile row for the
	/// single-lane-per-row statistics pass that follows.
	StoreTileResidualNorm {
		acc: usize,
		bias_binding: usize,
		residual_binding: usize,
		linear_binding: usize,
		sum_binding: usize,
		/// Shared tile (`BM × BN`, stride `BN`) receiving each `sum`.
		tile: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// Masked epilogue store for the `linear_gated_act` donor (silu_mul,
	/// `act_kind` 9): for every fragment element, `linear =
	/// acc_gate + bias[column]` stores at `element` and `up = acc_up +
	/// bias[column + gate_width]` at `element + gate_width` of the
	/// packed `linear_binding`; `act_binding` receives
	/// `swiglu(linear, up)` at `row * gate_width + column`.
	StoreTileGated {
		gate_acc: usize,
		up_acc: usize,
		bias_binding: usize,
		linear_binding: usize,
		act_binding: usize,
		/// `push.output_features` — the packed `linear` element stride.
		linear_stride: Extent,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		/// Gate-width bound — also the up-half bias/element offset and
		/// the act-output row stride.
		gate_width: Extent,
		/// Extra `u32` term added to both bias indices — the grouped
		/// route's `first_expert * output_features` row base.
		bias_origin: Option<usize>,
	},
	/// `gate_up_swiglu_bias` epilogue: two independent `[n, k]`
	/// projections with separate bias vectors, storing
	/// `silu(gate + gate_bias) * (up + up_bias)` per element — the
	/// donor's `gate_bias / (1 + exp(-gate_bias)) * up_bias` spelling.
	StoreTileGateUp {
		gate_acc: usize,
		up_acc: usize,
		gate_bias_binding: usize,
		up_bias_binding: usize,
		binding: usize,
		row_origin: Extent,
		column_origin: Extent,
		row_extent: Extent,
		column_extent: Extent,
	},
	/// `if (group_id.x >= bound) return;` — a workgroup-uniform dispatch
	/// guard. `bound` is a u32 `LaneExpr`; placement is template-owned
	/// and must precede any barrier in the op stream.
	Guard { bound: usize },
	/// `if (cond) return;` — a workgroup-uniform early exit over a bool
	/// lane expression. Grouped routes place it after the boundary
	/// fallback so the tiled path only runs for single-expert tiles.
	GuardIf { cond: usize },
	/// Unconditional `return;` — terminates every lane uniformly inside
	/// a `LaneIf` body (the grouped boundary fallback's tail).
	Return,
	/// `expert_for_row` scan — a data-dependent `while` over the offsets
	/// table writing the integral expert into an `f32` register element
	/// (values below 2^24 round-trip exactly through `uint(tile)`).
	ExpertScan {
		/// Offsets table buffer binding.
		offsets: usize,
		/// Expert-count extent (`push.experts`).
		experts: Extent,
		/// The `u32` row being resolved.
		row: usize,
		/// Destination 1x1 register tile.
		acc: usize,
	},
	/// `grouped_linear_m_act` boundary-tile fallback — per-fragment
	/// serial expert scan and K fold for tiles whose rows span expert
	/// boundaries. Emits the donor's nested `i`/`j` loops with `sum`/`up`
	/// scalar folds and the three packed stores.
	BoundaryGateUp {
		/// Register tile supplying the `fr x fc` fragment shape.
		acc: usize,
		offsets: usize,
		input: usize,
		weight: usize,
		bias: usize,
		output: usize,
		act: usize,
		experts: Extent,
		rows: Extent,
		outputs: Extent,
		inner: Extent,
		gate_width: Extent,
		row_origin: Extent,
		column_origin: Extent,
	},
	/// `grouped_linear_m`/`grouped_gemm_m` boundary-tile fallback —
	/// the single-accumulator form of `BoundaryGateUp`: per-fragment
	/// serial expert scan and K fold writing `acc + bias?` to the wide
	/// output. `bias: None` emits the `grouped_gemm_m` zero-init fold.
	BoundaryLinear {
		/// Register tile supplying the `fr x fc` fragment shape.
		acc: usize,
		offsets: usize,
		input: usize,
		weight: usize,
		bias: Option<usize>,
		output: usize,
		experts: Extent,
		rows: Extent,
		outputs: Extent,
		inner: Extent,
		column_extent: Extent,
		row_origin: Extent,
		column_origin: Extent,
		/// `true` emits the data-backward contraction: reduction over the
		/// `outputs` extent through `dY[row][n] * W[(expert*outputs+n)][k]`.
		transposed: bool,
	},
	/// Per-lane strided loop `for (i = lane; i < extent; i += lanes)`:
	/// `LaneExpr::Induction` inside the body resolves to the element
	/// index. Distinct from `Loop`, whose induction is a workgroup-
	/// uniform `0..extent` tile coordinate.
	LaneLoop { extent: Extent, body: Vec<TileOp> },
	/// `acc[0] = init` then a lane-strided fold
	/// `acc[0] = op(acc[0], element)` over `0..extent` — the per-lane
	/// half of a cooperative reduction. `acc` must be a `1×1` register
	/// tile; `element` may reference `LaneExpr::Induction`.
	LaneFold {
		acc: usize,
		init: usize,
		extent: Extent,
		op: LaneReduce,
		element: usize,
		/// `true` spells the donor's serial fold (`0..extent`, step 1);
		/// `false` is the lane-strided cooperative half.
		serial: bool,
	},
	/// Donor dual-accumulator fold: `acc` and `acc2` are both `1×1`
	/// register tiles, `element`/`element2` the two per-iteration terms
	/// folded in one strided loop — the layer-norm backward donor's
	/// `local_scaled_gradient` + `local_scaled_gradient_normalized`
	/// pair. Keeping both accumulates in one loop lets the shared
	/// product materialize like the donor's named local, preserving
	/// the donor's FMA contraction behavior bitwise.
	LaneFoldDual {
		acc: usize,
		acc2: usize,
		init: usize,
		extent: Extent,
		element: usize,
		element2: usize,
	},
	/// Per-lane strided loop `for (i = start; i < extent; i += step)`:
	/// like `LaneLoop` but with an arbitrary per-lane start and literal
	/// step — the `linear_residual_norm` normalize pass runs 4 lanes
	/// per row (`start = lane % 4`, `step = 4`).
	LaneLoopRange {
		start: usize,
		step: u32,
		extent: Extent,
		body: Vec<TileOp>,
	},
	/// `tile{acc}[index] = value` — a scalar write to a per-lane
	/// register element, e.g. hoisting a cooperative result out of
	/// shared memory.
	AssignScalar {
		tile: usize,
		index: u32,
		value: usize,
	},
	/// `shared{tile}[offset] = value` — one store per lane; `offset` is
	/// a u32 `LaneExpr`, `value` an f32 `LaneExpr`.
	StoreShared {
		tile: usize,
		offset: usize,
		value: usize,
	},
	/// `shared{tile}[offset] = value` — a u32 shared-tile store.
	StoreSharedU32 {
		tile: usize,
		offset: usize,
		value: usize,
	},
	/// `tile{tile}[index] = value` — a u32 register element write.
	AssignScalarU32 {
		tile: usize,
		index: u32,
		value: usize,
	},
	/// `storage_buffers[push.buffer{binding}].Store(address * 4u, value)` —
	/// a u32 element store (the router's `indices` output).
	StoreScalarU32 {
		binding: usize,
		address: usize,
		value: usize,
	},
	/// Cooperative paired argmax tree reduction across `shape.lanes`
	/// lanes: `value_tile` holds the f32 candidates, `index_tile` the
	/// u32 candidate indices; ties keep the smaller index.
	TreeReduceArgmax {
		value_tile: usize,
		index_tile: usize,
	},
	/// Cooperative tree reduction across `shape.lanes` shared-tile
	/// elements: `for (stride = lanes/2; stride; stride >>= 1)` with a
	/// barrier per step. Element 0 holds the result afterwards; the
	/// producer's `StoreShared` and its `Barrier` are separate ops.
	/// `lanes` must be a power of two.
	TreeReduce { tile: usize, op: LaneReduce },
	/// `store_f32(binding, address, value)` — a scalar f32 store per
	/// lane at a u32 element address.
	StoreScalar {
		binding: usize,
		address: usize,
		value: usize,
	},
	/// `if (lane == lane) { body }` — a single-lane guard for exclusive
	/// writes such as a per-row reduction result. `lane` is a u32
	/// `LaneExpr`.
	LaneGuard { lane: usize, body: Vec<TileOp> },
	/// `if (cond) { body }` — a per-lane condition guard (e.g. the
	/// `flat_index < count` bounds check of a flat thread program).
	/// `cond` is a `bool` `LaneExpr` (`LtU32`); bodies containing
	/// barriers require workgroup-uniform conditions.
	LaneIf { cond: usize, body: Vec<TileOp> },
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

/// A runtime extent operand in a block program: a push-constant scalar,
/// a scaled workgroup coordinate, or the enclosing `Loop` induction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum Extent {
	/// `push.{push_scalars[index]}` — a runtime `u32`.
	Push(usize),
	/// `group_id.x * scale` — the output row origin of the workgroup.
	GroupX(u32),
	/// `group_id.y * scale` — the output column origin of the workgroup.
	GroupY(u32),
	/// The innermost enclosing `TileOp::Loop` induction variable.
	Induction,
	/// `(push.{push_scalars[index]} / 2u)` — the `linear_gated_act`
	/// gate width (`output_features` packs gate and up halves).
	HalfPush(usize),
	/// A composite `u32` lane expression (expert-offset tile origins).
	Expr(usize),
}

/// A declared tile in a block program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum TileDecl {
	/// `groupshared float` staging: logical `rows × columns`, physical
	/// row `stride` (column padding is a template-owned bank policy).
	Shared {
		rows: u32,
		columns: u32,
		stride: u32,
	},
	/// Per-lane `float[rows * columns]` register fragment.
	Register { rows: u32, columns: u32 },
	/// `groupshared uint` staging: a flat `elements`-entry u32 array
	/// (router `taken`/`shared_indices` scratch).
	SharedU32 { elements: u32 },
	/// Per-lane `uint[elements]` register fragment.
	RegisterU32 { elements: u32 },
}

/// Per-workgroup execution shape for a block program. `lanes` is the
/// 1-D `numthreads` width; lanes are laid out as a
/// `lanes/thread_columns × thread_columns` fragment grid, each owning a
/// `fragment_rows × fragment_columns` register tile. The dispatch grid
/// derives from `tile_rows × tile_columns` output coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct BlockShape {
	pub lanes: u32,
	pub thread_columns: u32,
	pub fragment_rows: u32,
	pub fragment_columns: u32,
	pub tile_rows: u32,
	pub tile_columns: u32,
}

/// A scalar pointwise op folded into a tile program's store tail
/// (OA-CC-33): the operand bakes into the emitted shader as a
/// `ConstF32`, so a fused epilogue adds no bindings or pushes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::runtime) enum ScalarTail {
	Add(f32),
	Sub(f32),
	Mul(f32),
	Div(f32),
	Max(f32),
	Min(f32),
	Neg,
	Abs,
	Relu,
	Log,
	Exp,
	Sqrt,
	Recip,
}

/// A lowered block-level program: dense descriptor bindings, named
/// push-constant scalars, declared tiles, and ordered block ops over a
/// 2-D workgroup index space. Unlike the elementwise `TileProgram`,
/// loop structure and staging are first-class IR — the emitter owns
/// only text rendering, not schedule discovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct BlockProgram {
	pub bindings: Vec<Binding>,
	/// `uint` push fields, emitted before `push_f32_scalars`.
	pub push_scalars: Vec<String>,
	/// `float` push fields, emitted after the u32 scalars.
	pub push_f32_scalars: Vec<String>,
	pub shape: BlockShape,
	pub tiles: Vec<TileDecl>,
	/// Flat per-lane scalar expression DAG referenced by lane ops.
	pub lane_exprs: Vec<LaneExpr>,
	pub ops: Vec<TileOp>,
}

/// Push `node` onto `lane_exprs`, returning its index — the builder's
/// expression accumulator (closures cannot nest without double borrows).
/// Coordinate frame shared by the 16x16-tile image gather programs —
/// output pixel, bounds exit condition, plane base, and output index
/// over the donors' `[batch, channels, in_h, in_w, out_h, out_w]`
/// push prefix.
/// Expression indices emitted by `filter_prelude` — the shared
/// coordinate and border vocabulary of the `filter.slang` donors.
struct FilterPrelude {
	out_of_bounds: usize,
	zero_u32: usize,
	plane: usize,
	plane_base: usize,
	out_x: usize,
	out_y: usize,
	out_index: usize,
	height: usize,
	width: usize,
	mode: usize,
	kernel_width: usize,
	kernel_extent: usize,
	size: usize,
	size_radius: usize,
	size_extent: usize,
	radius_x: usize,
	radius_y: usize,
	border_value: usize,
	zero: usize,
	flt_max: usize,
	neg_flt_max: usize,
}

struct ImageGatherFrame {
	out_of_bounds: usize,
	plane: usize,
	plane_base: usize,
	out_x: usize,
	out_y: usize,
	out_index: usize,
	channels: usize,
	in_height: usize,
	in_width: usize,
	out_height: usize,
	out_width: usize,
}

impl ImageGatherFrame {
	fn new(lane_exprs: &mut Vec<LaneExpr>) -> Self {
		Self::new_out_at(lane_exprs, 4, 5)
	}

	/// `out_h`/`out_w` push indices — `crop`/`center_crop` interleave
	/// their offsets before the output extent.
	fn new_out_at(lane_exprs: &mut Vec<LaneExpr>, out_h_push: usize, out_w_push: usize) -> Self {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let plane = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(0));
		let channels = expr!(LaneExpr::Push(1));
		let in_height = expr!(LaneExpr::Push(2));
		let in_width = expr!(LaneExpr::Push(3));
		let out_height = expr!(LaneExpr::Push(out_h_push));
		let out_width = expr!(LaneExpr::Push(out_w_push));
		let batch_channels = expr!(LaneExpr::MulU32(batch, channels));
		let x_over = expr!(LaneExpr::GeU32(out_x, out_width));
		let y_over = expr!(LaneExpr::GeU32(out_y, out_height));
		let z_over = expr!(LaneExpr::GeU32(plane, batch_channels));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let in_plane = expr!(LaneExpr::MulU32(in_height, in_width));
		let plane_base = expr!(LaneExpr::MulU32(plane, in_plane));
		let out_rows = expr!(LaneExpr::MulU32(plane, out_height));
		let out_row = expr!(LaneExpr::AddU32(out_rows, out_y));
		let out_line = expr!(LaneExpr::MulU32(out_row, out_width));
		let out_index = expr!(LaneExpr::AddU32(out_line, out_x));
		Self {
			out_of_bounds,
			plane,
			plane_base,
			out_x,
			out_y,
			out_index,
			channels,
			in_height,
			in_width,
			out_height,
			out_width,
		}
	}

	/// Assemble the program shell: the guard plus one store of the
	/// gather value at the output index. `store_binding` is the output
	/// position in `bindings`.
	fn program(
		self,
		bindings: Vec<Binding>,
		push_scalars: Vec<String>,
		value: usize,
		lane_exprs: Vec<LaneExpr>,
	) -> BlockProgram {
		let store_binding = bindings.len() - 1;
		BlockProgram {
			bindings,
			push_scalars,
			push_f32_scalars: vec!["border_value".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: self.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: store_binding,
					address: self.out_index,
					value,
				},
			],
		}
	}
}

fn lane(lane_exprs: &mut Vec<LaneExpr>, node: LaneExpr) -> usize {
	lane_exprs.push(node);
	lane_exprs.len() - 1
}

/// `tail` applied in order to `value` — the f32 lane expression that
/// `with_scalar_tail` substitutes for a `StoreScalar` operand.
fn apply_scalar_tail(
	lane_exprs: &mut Vec<LaneExpr>,
	mut value: usize,
	tail: &[ScalarTail],
) -> usize {
	for op in tail {
		value = match *op {
			ScalarTail::Add(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::AddF32(value, c))
			}
			ScalarTail::Sub(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::SubF32(value, c))
			}
			ScalarTail::Mul(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::MulF32(value, c))
			}
			ScalarTail::Div(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::DivF32(value, c))
			}
			ScalarTail::Max(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::MaxF32(value, c))
			}
			ScalarTail::Min(scalar) => {
				let c = lane(lane_exprs, LaneExpr::ConstF32(scalar.to_bits()));
				lane(lane_exprs, LaneExpr::MinF32(value, c))
			}
			ScalarTail::Neg => {
				let zero = lane(lane_exprs, LaneExpr::ConstF32(0.0f32.to_bits()));
				lane(lane_exprs, LaneExpr::SubF32(zero, value))
			}
			ScalarTail::Abs => {
				let zero = lane(lane_exprs, LaneExpr::ConstF32(0.0f32.to_bits()));
				let negated = lane(lane_exprs, LaneExpr::SubF32(zero, value));
				lane(lane_exprs, LaneExpr::MaxF32(value, negated))
			}
			ScalarTail::Relu => {
				let zero = lane(lane_exprs, LaneExpr::ConstF32(0.0f32.to_bits()));
				lane(lane_exprs, LaneExpr::MaxF32(value, zero))
			}
			ScalarTail::Log => lane(lane_exprs, LaneExpr::LogF32(value)),
			ScalarTail::Exp => lane(lane_exprs, LaneExpr::ExpF32(value)),
			ScalarTail::Sqrt => lane(lane_exprs, LaneExpr::SqrtF32(value)),
			ScalarTail::Recip => {
				let one = lane(lane_exprs, LaneExpr::ConstF32(1.0f32.to_bits()));
				lane(lane_exprs, LaneExpr::DivF32(one, value))
			}
		};
	}
	value
}

/// `channel == 0 ? f0 : channel == 1 ? f1 : f2` — the video donors'
/// per-channel scalar select over three consecutive `PushF32` fields
/// (`fill`, `mean`, `std` triples). `zero_c`/`one_c` are the shared
/// `EqU32(channel, 0/1)` lane expressions.
fn channel_f32_sel(
	lane_exprs: &mut Vec<LaneExpr>,
	zero_c: usize,
	one_c: usize,
	base: usize,
) -> usize {
	let f0 = lane(lane_exprs, LaneExpr::PushF32(base));
	let f1 = lane(lane_exprs, LaneExpr::PushF32(base + 1));
	let f2 = lane(lane_exprs, LaneExpr::PushF32(base + 2));
	let tail = lane(
		lane_exprs,
		LaneExpr::SelectF32 {
			cond: one_c,
			if_true: f1,
			if_false: f2,
		},
	);
	lane(
		lane_exprs,
		LaneExpr::SelectF32 {
			cond: zero_c,
			if_true: f0,
			if_false: tail,
		},
	)
}

impl BlockProgram {
	/// Fold `tail` into every `StoreScalar` of this program: the
	/// stored value passes through each scalar op in order before the
	/// write, re-expressing a producer → scalar-epilogue node pair as
	/// one region (OA-CC-33). Operands bake as `ConstF32`, so the
	/// fused program keeps the producer's binding and push contract.
	pub(in crate::runtime) fn with_scalar_tail(mut self, tail: &[ScalarTail]) -> Self {
		if tail.is_empty() {
			return self;
		}
		for op in &mut self.ops {
			if let TileOp::StoreScalar { value, .. } = op {
				*value = apply_scalar_tail(&mut self.lane_exprs, *value, tail);
			}
		}
		self
	}

	/// Fold `prologue` into every `LoadF32` on `binding`: the loaded
	/// value passes through each scalar op before any consumer sees it,
	/// re-expressing a scalar-producer → tile-consumer node pair as one
	/// region (OA-CC-33). The load's expression index keeps its
	/// identity — the original load moves to a fresh slot and the slot
	/// all dependents resolve becomes the folded tail result.
	pub(in crate::runtime) fn with_scalar_prologue(
		mut self,
		binding: usize,
		prologue: &[ScalarTail],
	) -> Self {
		if prologue.is_empty() {
			return self;
		}
		let loads = self
			.lane_exprs
			.iter()
			.enumerate()
			.filter_map(|(index, expr)| {
				matches!(expr, LaneExpr::LoadF32 { binding: b, .. } if *b == binding).then_some(index)
			})
			.collect::<Vec<_>>();
		for index in loads {
			let cloned = self.lane_exprs[index].clone();
			let load = lane(&mut self.lane_exprs, cloned);
			let folded = apply_scalar_tail(&mut self.lane_exprs, load, prologue);
			self.lane_exprs[index] = LaneExpr::Alias { source: folded };
		}
		self
	}

	/// `[ceil(rows/tile_rows), ceil(columns/tile_columns), 1]` dispatch
	/// for this program's output tile coverage.
	pub(in crate::runtime) fn workgroups(&self, rows: u32, columns: u32) -> [u32; 3] {
		[
			rows.div_ceil(self.shape.tile_rows),
			columns.div_ceil(self.shape.tile_columns),
			1,
		]
	}

	/// `to_matrix` video conversion — the tile-IR re-expression of the
	/// `to_matrix_{i420,nv12,p010}` donors: one 16×16 workgroup per output
	/// tile, `group_id.z` the channel axis, typed plane loads over the
	/// `DevicePlanar` bindings, and the crop/resize/color/normalize
	/// prologue composed from lane expressions. `planar` selects separate
	/// U/V chroma bindings (I420) versus one interleaved UV plane
	/// (NV12/P010); `sample_bits` selects u8 or 10-in-16 codes.
	///
	/// The push ABI mirrors the donors: buffer indices first, then the
	/// eleven u32 scalars and seventeen f32 scalars in donor order.
	pub(in crate::runtime) fn video_to_matrix(planar: bool, sample_bits: u32) -> Result<Self> {
		if sample_bits != 8 && sample_bits != 10 {
			return Err(Error::invalid_argument(
				"video plane loads admit 8- or 10-bit samples",
			));
		}
		if planar && sample_bits != 8 {
			return Err(Error::invalid_argument(
				"planar video conversion admits 8-bit samples",
			));
		}
		let mut lane_exprs: Vec<LaneExpr> = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		// Push-scalar indices in donor order.
		const P_OUT_WIDTH: usize = 0;
		const P_OUT_HEIGHT: usize = 1;
		const P_OUTPUT_BASE: usize = 2;
		const P_CHANNELS: usize = 3;
		const P_LAYOUT: usize = 4;
		const P_INTERP: usize = 5;
		const P_LUMA_W: usize = 6;
		const P_LUMA_H: usize = 7;
		const P_CHROMA_W: usize = 8;
		const P_CHROMA_H: usize = 9;
		const P_COLOR: usize = 10;
		// Push-f32 indices in donor order.
		const F_OFF_X: usize = 0;
		const F_OFF_Y: usize = 1;
		const F_SCALE_X: usize = 2;
		const F_SCALE_Y: usize = 3;
		const F_VALID_MIN_X: usize = 4;
		const F_VALID_MIN_Y: usize = 5;
		const F_VALID_MAX_X: usize = 6;
		const F_VALID_MAX_Y: usize = 7;
		const F_FILL: usize = 8;
		const F_MEAN: usize = 11;
		const F_STD: usize = 14;

		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let channel = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let half = expr!(LaneExpr::ConstF32(0.5f32.to_bits()));
		let zero = expr!(LaneExpr::ConstU32(0));
		let one = expr!(LaneExpr::ConstU32(1));
		let two = expr!(LaneExpr::ConstU32(2));
		let three = expr!(LaneExpr::ConstU32(3));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let out_width = expr!(LaneExpr::Push(P_OUT_WIDTH));
		let out_height = expr!(LaneExpr::Push(P_OUT_HEIGHT));
		let channels_mode = expr!(LaneExpr::Push(P_CHANNELS));
		let luma_only = expr!(LaneExpr::EqU32(channels_mode, zero));
		let channel_count = expr!(LaneExpr::SelectU32 {
			cond: luma_only,
			if_true: one,
			if_false: three,
		});
		let fit_x = expr!(LaneExpr::LtU32(out_x, out_width));
		let fit_y = expr!(LaneExpr::LtU32(out_y, out_height));
		let fit_c = expr!(LaneExpr::LtU32(channel, channel_count));
		let fit_xy = expr!(LaneExpr::AndBool(fit_x, fit_y));
		let in_bounds = expr!(LaneExpr::AndBool(fit_xy, fit_c));
		let out_x_f = expr!(LaneExpr::F32FromU32(out_x));
		let out_y_f = expr!(LaneExpr::F32FromU32(out_y));
		let px = expr!(LaneExpr::AddF32(out_x_f, half));
		let py = expr!(LaneExpr::AddF32(out_y_f, half));
		let valid_min_x = expr!(LaneExpr::PushF32(F_VALID_MIN_X));
		let valid_max_x = expr!(LaneExpr::PushF32(F_VALID_MAX_X));
		let valid_min_y = expr!(LaneExpr::PushF32(F_VALID_MIN_Y));
		let valid_max_y = expr!(LaneExpr::PushF32(F_VALID_MAX_Y));
		let inside_x_lo = expr!(LaneExpr::GeF32(px, valid_min_x));
		let inside_x_hi = expr!(LaneExpr::LtF32(px, valid_max_x));
		let inside_y_lo = expr!(LaneExpr::GeF32(py, valid_min_y));
		let inside_y_hi = expr!(LaneExpr::LtF32(py, valid_max_y));
		let inside_x = expr!(LaneExpr::AndBool(inside_x_lo, inside_x_hi));
		let inside_y = expr!(LaneExpr::AndBool(inside_y_lo, inside_y_hi));
		let inside = expr!(LaneExpr::AndBool(inside_x, inside_y));
		let scale_x = expr!(LaneExpr::PushF32(F_SCALE_X));
		let scale_y = expr!(LaneExpr::PushF32(F_SCALE_Y));
		let off_x = expr!(LaneExpr::PushF32(F_OFF_X));
		let off_y = expr!(LaneExpr::PushF32(F_OFF_Y));
		let px_scaled = expr!(LaneExpr::MulF32(px, scale_x));
		let py_scaled = expr!(LaneExpr::MulF32(py, scale_y));
		let sx = expr!(LaneExpr::AddF32(px_scaled, off_x));
		let sy = expr!(LaneExpr::AddF32(py_scaled, off_y));
		let interp = expr!(LaneExpr::Push(P_INTERP));
		let color_mode = expr!(LaneExpr::Push(P_COLOR));
		let luma_width = expr!(LaneExpr::Push(P_LUMA_W));
		let luma_height = expr!(LaneExpr::Push(P_LUMA_H));
		let chroma_width = expr!(LaneExpr::Push(P_CHROMA_W));
		let chroma_height = expr!(LaneExpr::Push(P_CHROMA_H));
		let luma_code = expr!(LaneExpr::PlaneSample {
			binding: 0,
			x: sx,
			y: sy,
			width: luma_width,
			height: luma_height,
			components: 1,
			component: 0,
			interp,
			sample_bits,
		});
		let luma = expr!(LaneExpr::ScalePlane {
			kind: PlaneScale::Luma,
			code: luma_code,
			color_mode,
			sample_bits,
		});
		let sx_half = expr!(LaneExpr::SubF32(sx, half));
		let sy_half = expr!(LaneExpr::SubF32(sy, half));
		let chroma_x = expr!(LaneExpr::MulF32(sx_half, half));
		let chroma_y = expr!(LaneExpr::MulF32(sy_half, half));
		let (cb_binding, cr_binding, components) = if planar { (1, 2, 1) } else { (1, 1, 2) };
		let cb_code = expr!(LaneExpr::PlaneSample {
			binding: cb_binding,
			x: chroma_x,
			y: chroma_y,
			width: chroma_width,
			height: chroma_height,
			components,
			component: 0,
			interp,
			sample_bits,
		});
		let cr_code = expr!(LaneExpr::PlaneSample {
			binding: cr_binding,
			x: chroma_x,
			y: chroma_y,
			width: chroma_width,
			height: chroma_height,
			components,
			component: u32::from(!planar),
			interp,
			sample_bits,
		});
		let cb = expr!(LaneExpr::ScalePlane {
			kind: PlaneScale::Chroma,
			code: cb_code,
			color_mode,
			sample_bits,
		});
		let cr = expr!(LaneExpr::ScalePlane {
			kind: PlaneScale::Chroma,
			code: cr_code,
			color_mode,
			sample_bits,
		});
		let zero_c = expr!(LaneExpr::EqU32(channel, zero));
		let one_c = expr!(LaneExpr::EqU32(channel, one));
		let cb_half = expr!(LaneExpr::AddF32(cb, half));
		let cr_half = expr!(LaneExpr::AddF32(cr, half));
		let yuv_tail = expr!(LaneExpr::SelectF32 {
			cond: one_c,
			if_true: cb_half,
			if_false: cr_half,
		});
		let yuv_value = expr!(LaneExpr::SelectF32 {
			cond: zero_c,
			if_true: luma,
			if_false: yuv_tail,
		});
		let bgr_mode = expr!(LaneExpr::EqU32(channels_mode, two));
		let bgr_component = expr!(LaneExpr::SubU32(two, channel));
		let component = expr!(LaneExpr::SelectU32 {
			cond: bgr_mode,
			if_true: bgr_component,
			if_false: channel,
		});
		let space = expr!(LaneExpr::ShrU32 {
			value: color_mode,
			bits: 1
		});
		let rgb_value = expr!(LaneExpr::YuvToRgb {
			luma,
			cb,
			cr,
			space,
			component,
		});
		let yuv_mode = expr!(LaneExpr::EqU32(channels_mode, three));
		let tri_value = expr!(LaneExpr::SelectF32 {
			cond: yuv_mode,
			if_true: yuv_value,
			if_false: rgb_value,
		});
		let converted = expr!(LaneExpr::SelectF32 {
			cond: luma_only,
			if_true: luma,
			if_false: tri_value,
		});
		let fill = channel_f32_sel(&mut lane_exprs, zero_c, one_c, F_FILL);
		let mean = channel_f32_sel(&mut lane_exprs, zero_c, one_c, F_MEAN);
		let std_sel = channel_f32_sel(&mut lane_exprs, zero_c, one_c, F_STD);
		let value = expr!(LaneExpr::SelectF32 {
			cond: inside,
			if_true: converted,
			if_false: fill,
		});
		let centered = expr!(LaneExpr::SubF32(value, mean));
		let result = expr!(LaneExpr::DivF32(centered, std_sel));
		// layout_mode 0: (channel * out_height + out_y) * out_width + out_x
		// layout_mode 1: (out_y * out_width + out_x) * channel_count + channel
		let output_base = expr!(LaneExpr::Push(P_OUTPUT_BASE));
		let layout_mode = expr!(LaneExpr::Push(P_LAYOUT));
		let ch_span = expr!(LaneExpr::MulU32(channel, out_height));
		let ch_row = expr!(LaneExpr::AddU32(ch_span, out_y));
		let chw_row = expr!(LaneExpr::MulU32(ch_row, out_width));
		let chw_flat = expr!(LaneExpr::AddU32(chw_row, out_x));
		let chw_index = expr!(LaneExpr::AddU32(output_base, chw_flat));
		let hw_row = expr!(LaneExpr::MulU32(out_y, out_width));
		let hw_px = expr!(LaneExpr::AddU32(hw_row, out_x));
		let hwc_px = expr!(LaneExpr::MulU32(hw_px, channel_count));
		let hwc_flat = expr!(LaneExpr::AddU32(hwc_px, channel));
		let hwc_index = expr!(LaneExpr::AddU32(output_base, hwc_flat));
		let planar_layout = expr!(LaneExpr::EqU32(layout_mode, zero));
		let index = expr!(LaneExpr::SelectU32 {
			cond: planar_layout,
			if_true: chw_index,
			if_false: hwc_index,
		});
		let output_binding = if planar { 3 } else { 2 };
		let mut bindings: Vec<Binding> = Vec::new();
		if planar {
			bindings.extend([Binding::Input(0), Binding::Input(1), Binding::Input(2)]);
		} else {
			bindings.extend([Binding::Input(0), Binding::Input(1)]);
		}
		bindings.push(Binding::Output(0));
		Ok(Self {
			bindings,
			push_scalars: [
				"out_width",
				"out_height",
				"output_base",
				"channels_mode",
				"layout_mode",
				"interp_mode",
				"luma_width",
				"luma_height",
				"chroma_width",
				"chroma_height",
				"color_mode",
			]
			.iter()
			.map(|name| name.to_string())
			.collect(),
			push_f32_scalars: [
				"src_off_x",
				"src_off_y",
				"src_scale_x",
				"src_scale_y",
				"valid_min_x",
				"valid_min_y",
				"valid_max_x",
				"valid_max_y",
				"fill_0",
				"fill_1",
				"fill_2",
				"mean_0",
				"mean_1",
				"mean_2",
				"std_0",
				"std_1",
				"std_2",
			]
			.iter()
			.map(|name| name.to_string())
			.collect(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 16,
				tile_columns: 16,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![TileOp::LaneIf {
				cond: in_bounds,
				body: vec![TileOp::StoreScalar {
					binding: output_binding,
					address: index,
					value: result,
				}],
			}],
		})
	}

	/// Tiled `C[m,n] = A[m,k] · B[n,k]ᵀ` template — the tile-IR
	/// re-expression of `mat_mul_nt_tiled`'s schedule: one workgroup per
	/// output tile, cooperative `vec`-wide staging into padded shared
	/// tiles, per-lane fragment FMA, masked tails on every boundary.
	///
	/// `tile_rows × tile_columns` is the workgroup output coverage,
	/// `tile_k` the reduction tile, `fragment_rows × fragment_columns`
	/// the per-lane register tile and `vec` the staging load width.
	/// `Err` rejects geometry the emitted schedule cannot honor: the
	/// fragment grid must exactly cover the output tile, and a `vec`
	/// group may never straddle a staged row.
	pub(in crate::runtime) fn mat_mul_nt(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		if vec == 0 || vec > 4 || !tile_k.is_multiple_of(vec) {
			return Err(Error::invalid_argument(
				"tile_k must be a positive multiple of the staging width (≤4)",
			));
		}
		if !tile_rows.is_multiple_of(fragment_rows) || !tile_columns.is_multiple_of(fragment_columns) {
			return Err(Error::invalid_argument(
				"output tile must be an exact multiple of the fragment tile",
			));
		}
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(2, 0, 1),
		)?;
		ops.push(TileOp::StoreTile {
			acc: 2,
			binding: 2,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(1),
			mask_origin: None,
		});
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec!["m".into(), "n".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `mat_mul_nt_bias*` template — the `mat_mul_nt` schedule
	/// with the BLAS epilogue donors' `[left, right, bias, output]`
	/// binding order and `[m, n, k]` push ABI. `act` selects the
	/// donor's fixed inline activation (`bias_relu`/`bias_gelu`/
	/// `bias_silu`); `None` is the plain bias store.
	pub(in crate::runtime) fn mat_mul_nt_bias(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
		act: Option<BiasAct>,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(2, 0, 1),
		)?;
		ops.push(TileOp::StoreTileBias {
			acc: 2,
			bias_binding: 2,
			z: false,
			binding: 3,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(1),
			bias_origin: None,
			act,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec!["m".into(), "n".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear` template — the same schedule as `mat_mul_nt`
	/// with the `linear.slang` ABI: `[batch, input_features,
	/// output_features]` scalars, a third bias input and a
	/// `bias + acc` epilogue store.
	pub(in crate::runtime) fn linear(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		ops.push(TileOp::StoreTileBias {
			acc: 2,
			bias_binding: 2,
			z: false,
			binding: 3,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
			bias_origin: None,
			act: None,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_act` template — the `linear` schedule plus the
	/// donor's `switch (act_kind)` epilogue writing both the
	/// pre-activation and activated outputs. The `kind` push scalar is
	/// u32 index 3 and `alpha` is f32 index 0.
	pub(in crate::runtime) fn linear_act(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		ops.push(TileOp::StoreTileAct {
			acc: 2,
			bias_binding: 2,
			linear_binding: 3,
			act_binding: 4,
			kind: 3,
			alpha: 0,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
				"act_kind".into(),
			],
			push_f32_scalars: vec!["alpha".into()],
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_residual` template — `linear` plus a residual
	/// input and the dual linear/sum output stores of the donor.
	pub(in crate::runtime) fn linear_residual(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		ops.push(TileOp::StoreTileResidual {
			acc: 2,
			bias_binding: 2,
			residual_binding: 3,
			linear_binding: 4,
			sum_binding: 5,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_residual_act` template — the residual epilogue
	/// plus the `switch (act_kind)` store, materializing linear, sum
	/// and activated outputs like the donor.
	pub(in crate::runtime) fn linear_residual_act(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		ops.push(TileOp::StoreTileResidualAct {
			acc: 2,
			bias_binding: 2,
			residual_binding: 3,
			linear_binding: 4,
			sum_binding: 5,
			act_binding: 6,
			kind: 3,
			alpha: 0,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
				"act_kind".into(),
			],
			push_f32_scalars: vec!["alpha".into()],
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Whole-tensor `sum` template — the tile-IR re-expression of
	/// `sum.slang`: one 256-lane workgroup, a lane-strided serial fold,
	/// shared-memory tree reduce and a lane-0 scalar store. The emitted
	/// program preserves the donor ABI (`[element_count]` behind two
	/// buffer indices) and its `[1,1,1]` dispatch.
	pub(in crate::runtime) fn sum(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"sum lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			},
			/* 3 */ E::ConstF32(0), // 0.0f
			/* 4 */ E::RegisterF32 { tile: 1, index: 0 },
			/* 5 */ E::Lane,
			/* 6 */ E::ConstU32(0),
			/* 7 */ E::SharedF32 { tile: 0, offset: 6 },
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::LaneFold {
					acc: 1,
					init: 3,
					extent: Extent::Push(0),
					op: LaneReduce::Sum,
					element: 2,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 5,
					value: 4,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::LaneGuard {
					lane: 6,
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 6,
						value: 7,
					}],
				},
			],
		})
	}

	/// `sum_axis` keep-dim template — the donor's flat-thread schedule
	/// spelled per lane: each lane owns output element
	/// `group_id.x * lanes + lane`, folds `0..dim_size` serially and
	/// stores. No shared memory; the `[outer, dim, inner]` push ABI and
	/// `ceil(outer*inner/lanes)` grid are preserved, and the emitted
	/// order is bit-exact with the incumbent.
	pub(in crate::runtime) fn sum_axis(lanes: u32) -> Result<Self> {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // outer_size
			/* 1 */ E::Push(1), // dim_size
			/* 2 */ E::Push(2), // inner_size
			/* 3 */ E::GroupX,
			/* 4 */ E::ConstU32(lanes),
			/* 5 */ E::MulU32(3, 4), // group*lanes
			/* 6 */ E::Lane,
			/* 7 */ E::AddU32(5, 6), // flat thread/output index
			/* 8 */ E::MulU32(0, 2), // outer*inner bound
			/* 9 */ E::LtU32(7, 8), // index < bound
			/* 10 */ E::ModU32(7, 2), // inner index
			/* 11 */ E::DivU32(7, 2), // outer index
			/* 12 */ E::MulU32(11, 1), // outer*dim
			/* 13 */ E::MulU32(12, 2), // outer*dim*inner
			/* 14 */ E::AddU32(13, 10), // row base
			/* 15 */ E::Induction, // axis index
			/* 16 */ E::MulU32(15, 2), // index*inner
			/* 17 */ E::AddU32(14, 16), // element address
			/* 18 */ E::LoadF32 {
				binding: 0,
				address: 17,
			},
			/* 19 */ E::ConstF32(0),
			/* 20 */ E::RegisterF32 { tile: 0, index: 0 },
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			ops: vec![TileOp::LaneIf {
				cond: 9,
				body: vec![
					TileOp::LaneFold {
						acc: 0,
						init: 19,
						extent: Extent::Push(1),
						op: LaneReduce::Sum,
						element: 18,
						serial: true,
					},
					TileOp::StoreScalar {
						binding: 1,
						address: 7,
						value: 20,
					},
				],
			}],
		})
	}

	/// `sum_axis` cooperative variant — one workgroup per
	/// `(outer, inner)` slice for long reduction axes, where the flat
	/// donor serializes the whole `dim` scan into one thread. Each lane
	/// folds a strided share of the axis into shared memory, the tree
	/// reduce finishes, and lane 0 stores at the slice index
	/// `group_id.x`. The `[outer, dim, inner]` push ABI is preserved;
	/// specialization rewrites `workgroups` to `[outer*inner, 1, 1]`.
	/// The tree order differs from the donor's serial scan, so callers
	/// qualify this variant against a tolerance oracle.
	pub(in crate::runtime) fn sum_axis_cooperative(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"sum_axis_cooperative lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // outer_size
			/* 1 */ E::Push(1), // dim_size
			/* 2 */ E::Push(2), // inner_size
			/* 3 */ E::GroupX, // slice index
			/* 4 */ E::ModU32(3, 2), // inner index
			/* 5 */ E::DivU32(3, 2), // outer index
			/* 6 */ E::MulU32(5, 1), // outer*dim
			/* 7 */ E::MulU32(6, 2), // outer*dim*inner
			/* 8 */ E::AddU32(7, 4), // row base
			/* 9 */ E::Induction, // strided axis index
			/* 10 */ E::MulU32(9, 2), // index*inner
			/* 11 */ E::AddU32(8, 10), // element address
			/* 12 */ E::LoadF32 {
				binding: 0,
				address: 11,
			},
			/* 13 */ E::ConstF32(0),
			/* 14 */ E::RegisterF32 { tile: 1, index: 0 },
			/* 15 */ E::Lane,
			/* 16 */ E::ConstU32(0),
			/* 17 */ E::SharedF32 {
				tile: 0,
				offset: 16,
			},
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::LaneFold {
					acc: 1,
					init: 13,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 12,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 14,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::LaneGuard {
					lane: 16,
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 3,
						value: 17,
					}],
				},
			],
		})
	}

	/// `conv_1d` implicit-GEMM prologue — the donor's flat im2col
	/// gather: `columns[index] = x[batch, channel, position]` or zero
	/// when the dilated kernel tap falls outside `[0, input_length)`.
	/// The donor's signed `input_position` is spelled as the u32 pair
	/// `base >= padding && base - padding < input_length`, identical
	/// under the in-range case the donor's `int` comparison admits.
	/// `[input, columns]` + eight u32 scalars ABI.
	pub(in crate::runtime) fn conv_1d_im2_col() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // flat index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // input_length
			/* 8 */ E::Push(3), // kernel_size
			/* 9 */ E::Push(4), // stride
			/* 10 */ E::Push(5), // padding
			/* 11 */ E::Push(6), // dilation
			/* 12 */ E::Push(7), // output_length
			/* 13 */ E::MulU32(6, 8), // inner_size = channels*kernel
			/* 14 */ E::MulU32(5, 12), // batch*output_length
			/* 15 */ E::MulU32(14, 13), // element_count
			/* 16 */ E::GeU32(4, 15), // out-of-range guard
			/* 17 */ E::ModU32(4, 13), // column
			/* 18 */ E::DivU32(4, 13), // row
			/* 19 */ E::ModU32(17, 8), // kernel tap
			/* 20 */ E::DivU32(17, 8), // input_channel
			/* 21 */ E::ModU32(18, 12), // output_position
			/* 22 */ E::DivU32(18, 12), // batch
			/* 23 */ E::MulU32(21, 9), // position*stride
			/* 24 */ E::MulU32(19, 11), // kernel*dilation
			/* 25 */ E::AddU32(23, 24), // pre-shift base
			/* 26 */ E::GeU32(25, 10), // base >= padding
			/* 27 */ E::SubU32(25, 10), // input_position
			/* 28 */ E::LtU32(27, 7), // position < input_length
			/* 29 */ E::AndBool(26, 28), // valid tap
			/* 30 */ E::MulU32(22, 6), // batch*channels
			/* 31 */ E::AddU32(30, 20), // + channel
			/* 32 */ E::MulU32(31, 7), // * input_length
			/* 33 */ E::AddU32(32, 27), // input_offset
			/* 34 */ E::LoadF32 {
				binding: 0,
				address: 33,
			},
			/* 35 */ E::ConstF32(0), // 0.0f
			/* 36 */ E::RegisterF32 { tile: 0, index: 0 },
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"input_length".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"dilation".into(),
				"output_length".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 16 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 35,
				},
				TileOp::LaneIf {
					cond: 29,
					body: vec![TileOp::AssignScalar {
						tile: 0,
						index: 0,
						value: 34,
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 4,
					value: 36,
				},
			],
		)
	}

	/// `conv_1d` bias epilogue — the donor's in-place last-axis bias
	/// broadcast: `values[index] += bias[index % columns]` over
	/// `rows * columns` flat elements. `[values, bias]` bindings with
	/// values read-write + `[rows, columns]` scalars ABI.
	pub(in crate::runtime) fn conv_1d_bias_add() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // flat index
			/* 5 */ E::Push(0), // rows
			/* 6 */ E::Push(1), // columns
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7), // out-of-range guard
			/* 9 */ E::ModU32(4, 6), // column
			/* 10 */ E::LoadF32 {
				binding: 0,
				address: 4,
			},
			/* 11 */ E::LoadF32 {
				binding: 1,
				address: 9,
			},
			/* 12 */ E::AddF32(10, 11),
		];
		Self::flat_program(
			vec![Binding::ReadWrite(0), Binding::Input(0)],
			vec!["rows".into(), "columns".into()],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::StoreScalar {
					binding: 0,
					address: 4,
					value: 12,
				},
			],
		)
	}

	/// `conv_1d` tail transpose — the donor's 32×32 shared-tile
	/// transpose, one workgroup per `(column tile, row tile, batch)`
	/// with `group_id.z` batching. The 1024-lane block keeps the
	/// donor's `(group_thread_id.y, .x)` decomposition as
	/// `lane / 32`/`lane % 32`; the `TILE + 1` stride preserves its
	/// bank-conflict padding. `[input, output]` +
	/// `[batch_size, rows, columns]` ABI.
	pub(in crate::runtime) fn conv_1d_transpose() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Lane,
			/* 1 */ E::ConstU32(32),
			/* 2 */ E::DivU32(0, 1), // thread_row (donor group_thread_id.y)
			/* 3 */ E::ModU32(0, 1), // thread_column (donor .x)
			/* 4 */ E::GroupX,
			/* 5 */ E::GroupY,
			/* 6 */ E::GroupZ, // batch
			/* 7 */ E::Push(0), // batch_size
			/* 8 */ E::Push(1), // rows
			/* 9 */ E::Push(2), // columns
			/* 10 */ E::MulU32(5, 1), // group_y * TILE
			/* 11 */ E::AddU32(10, 2), // input_row
			/* 12 */ E::MulU32(4, 1), // group_x * TILE
			/* 13 */ E::AddU32(12, 3), // input_column
			/* 14 */ E::LtU32(11, 8), // input_row < rows
			/* 15 */ E::LtU32(13, 9), // input_column < columns
			/* 16 */ E::AndBool(14, 15), // valid input tap
			/* 17 */ E::MulU32(8, 9), // rows * columns
			/* 18 */ E::MulU32(6, 17), // batch input base
			/* 19 */ E::MulU32(11, 9), // input_row * columns
			/* 20 */ E::AddU32(19, 13), // + input_column
			/* 21 */ E::AddU32(18, 20), // input address
			/* 22 */ E::LoadF32 {
				binding: 0,
				address: 21,
			},
			/* 23 */ E::ConstF32(0), // 0.0f
			/* 24 */ E::ConstU32(33), // TILE_PAD
			/* 25 */ E::MulU32(2, 24), // thread_row * TILE_PAD
			/* 26 */ E::AddU32(25, 3), // staged write offset
			/* 27 */ E::AddU32(12, 2), // output_row = group_x*TILE + thread_row
			/* 28 */ E::AddU32(10, 3), // output_column = group_y*TILE + thread_column
			/* 29 */ E::LtU32(27, 9), // output_row < columns
			/* 30 */ E::LtU32(28, 8), // output_column < rows
			/* 31 */ E::AndBool(29, 30), // valid output tap
			/* 32 */ E::MulU32(27, 8), // output_row * rows
			/* 33 */ E::AddU32(32, 28), // + output_column
			/* 34 */ E::AddU32(18, 33), // output address (same batch base)
			/* 35 */ E::MulU32(3, 24), // thread_column * TILE_PAD
			/* 36 */ E::AddU32(35, 2), // staged read offset
			/* 37 */ E::SharedF32 {
				tile: 0,
				offset: 36,
			},
		];
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["batch_size".into(), "rows".into(), "columns".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 1024,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![TileDecl::Shared {
				rows: 32,
				columns: 32,
				stride: 33,
			}],
			lane_exprs,
			ops: vec![
				TileOp::StoreShared {
					tile: 0,
					offset: 26,
					value: 23,
				},
				TileOp::LaneIf {
					cond: 16,
					body: vec![TileOp::StoreShared {
						tile: 0,
						offset: 26,
						value: 22,
					}],
				},
				TileOp::Barrier,
				TileOp::LaneIf {
					cond: 31,
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 34,
						value: 37,
					}],
				},
			],
		}
	}

	/// `conv_2d` direct NCHW grouped convolution — the donor's flat
	/// 256-lane output elementwise program: one output element per
	/// lane, a per-lane accumulator initialized from `bias[channel]`,
	/// and the `(channels/group) × kernel × kernel` triple loop with
	/// the donor's signed bounds spelled as the u32 pairs
	/// `base >= padding && base - padding < extent`. Nested
	/// `LaneLoopRange`s reach outer inductions through `LoopVar`.
	/// `[input, weight, bias, output]` + the donor's eleven u32
	/// scalars ABI.
	pub(in crate::runtime) fn conv_2d() -> Self {
		Self::conv_2d_program(false)
	}

	/// `conv_2d_relu` — the submit-time Conv2d→ReLU fusion donor:
	/// identical to `conv_2d` except the store applies
	/// `max(0.0f, value)`, so the raw conv output is never
	/// materialized.
	pub(in crate::runtime) fn conv_2d_relu() -> Self {
		Self::conv_2d_program(true)
	}

	fn conv_2d_program(relu: bool) -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // output_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::Push(10), // groups
			/* 16 */ E::MulU32(13, 14), // output_height*width
			/* 17 */ E::MulU32(7, 16), // output_channels*plane
			/* 18 */ E::MulU32(5, 17), // output_count
			/* 19 */ E::GeU32(4, 18), // out-of-range guard
			/* 20 */ E::ModU32(4, 14), // output_x
			/* 21 */ E::DivU32(4, 14), // quotient
			/* 22 */ E::ModU32(21, 13), // output_y
			/* 23 */ E::DivU32(21, 13), // quotient
			/* 24 */ E::ModU32(23, 7), // output_channel
			/* 25 */ E::DivU32(23, 7), // batch
			/* 26 */ E::DivU32(6, 15), // input_channels_per_group
			/* 27 */ E::DivU32(7, 15), // output_channels_per_group
			/* 28 */ E::DivU32(24, 27), // group
			/* 29 */ E::MulU32(28, 26), // input_channel_base
			/* 30 */ E::LoadF32 {
				binding: 2,
				address: 24,
			}, // bias[output_channel]
			/* 31 */ E::MulU32(22, 11), // output_y*stride
			/* 32 */ E::MulU32(20, 11), // output_x*stride
			/* 33 */ E::LoopVar(3), // kernel_y
			/* 34 */ E::AddU32(70, 33), // input_y pre-shift base
			/* 35 */ E::GeU32(34, 12), // base >= padding
			/* 36 */ E::SubU32(34, 12), // input_y
			/* 37 */ E::LtU32(36, 8), // input_y < input_height
			/* 38 */ E::LoopVar(5), // kernel_x
			/* 39 */ E::AddU32(71, 38), // input_x pre-shift base
			/* 40 */ E::GeU32(39, 12), // base >= padding
			/* 41 */ E::SubU32(39, 12), // input_x
			/* 42 */ E::LtU32(41, 9), // input_x < input_width
			/* 43 */ E::LoopVar(2), // local_input_channel
			/* 44 */ E::AddU32(72, 43), // input_channel
			/* 45 */ E::MulU32(25, 6), // batch*input_channels
			/* 46 */ E::MulU32(44, 8), // input_channel * input_height
			/* 47 */ E::AddU32(46, 36), // + input_y
			/* 48 */ E::MulU32(47, 9), // * input_width
			/* 49 */ E::AddU32(48, 41), // + input_x
			/* 50 */ E::AddU32(73, 49), // input_offset = in_base + taps
			/* 51 */ E::MulU32(24, 26), // oc*input_channels_per_group
			/* 52 */ E::AddU32(74, 43), // oc_icpg + local_input_channel
			/* 53 */ E::MulU32(52, 10), // * kernel_size
			/* 54 */ E::AddU32(53, 33), // + kernel_y
			/* 55 */ E::MulU32(54, 10), // * kernel_size
			/* 56 */ E::AddU32(55, 38), // weight_offset
			/* 57 */ E::LoadF32 {
				binding: 0,
				address: 50,
			},
			/* 58 */ E::LoadF32 {
				binding: 1,
				address: 56,
			},
			/* 59 */ E::MulF32(57, 58), // tap product
			/* 60 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 61 */ E::AddF32(60, 59), // acc + product
			/* 62 */ E::AndBool(35, 37), // input_y in range
			/* 63 */ E::AndBool(40, 42), // input_x in range
			/* 64 */ E::AndBool(62, 63), // valid tap
			/* 65 */ E::ConstU32(0), // loop starts
			/* 66 */ E::ConstF32(0), // 0.0f
			/* 67 */ E::MaxF32(66, 60), // max(0.0f, acc)
			/* 68 */ E::MulU32(45, 8), // batch * ic * input_height
			/* 69 */ E::MulU32(68, 9), // in_base = batch * ic * ih * iw
			/* 70 */ E::RegisterU32 { tile: 1, index: 0 }, // output_y * stride
			/* 71 */ E::RegisterU32 { tile: 1, index: 1 }, // output_x * stride
			/* 72 */ E::RegisterU32 { tile: 1, index: 2 }, // input_channel_base
			/* 73 */ E::RegisterU32 { tile: 1, index: 3 }, // in_base
			/* 74 */ E::RegisterU32 { tile: 1, index: 4 }, // oc * icpg
			/* 75 */ E::RegisterU32 { tile: 1, index: 5 }, // output_index
		];
		let store_value = if relu { 67 } else { 60 };
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
				"groups".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 6 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 19 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 30,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 31,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 32,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 29,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 69,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 51,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 5,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 65,
					step: 1,
					extent: Extent::Expr(26),
					body: vec![TileOp::LaneLoopRange {
						start: 65,
						step: 1,
						extent: Extent::Push(5),
						body: vec![TileOp::LaneIf {
							cond: 62,
							body: vec![TileOp::LaneLoopRange {
								start: 65,
								step: 1,
								extent: Extent::Push(5),
								body: vec![TileOp::LaneIf {
									cond: 63,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 61,
									}],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 3,
					address: 75,
					value: store_value,
				},
			],
		)
	}

	/// `conv_1d` parameter adjoint — the donor's deterministic
	/// weight/bias gradient program. Lanes below `weight_count`
	/// decompose to `(oc, ic, kernel)` and accumulate
	/// `input × output_gradient` over `batch × output_length` serial
	/// taps with the donor's signed `input_position` spelled as the u32
	/// pair `base >= padding && base - padding < input_length`; lanes at
	/// or above it write the per-channel bias gradient sum.
	/// `[input, output_gradient, weight_gradient, bias_gradient]` +
	/// ten u32 scalars ABI.
	pub(in crate::runtime) fn conv_1d_parameter_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // parameter_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_length
			/* 9 */ E::Push(4), // kernel_size
			/* 10 */ E::Push(5), // stride
			/* 11 */ E::Push(6), // padding
			/* 12 */ E::Push(7), // dilation
			/* 13 */ E::Push(8), // output_length
			/* 14 */ E::Push(9), // parameter_count
			/* 15 */ E::GeU32(4, 14), // out-of-range guard
			/* 16 */ E::MulU32(7, 6), // oc*ic
			/* 17 */ E::MulU32(16, 9), // weight_count
			/* 18 */ E::GeU32(4, 17), // bias-gradient lane
			/* 19 */ E::LtU32(4, 17), // weight-gradient lane
			/* 20 */ E::SubU32(4, 17), // bias output_channel
			/* 21 */ E::LoopVar(3), // batch
			/* 22 */ E::LoopVar(4), // output_position
			/* 23 */ E::MulU32(21, 7), // batch*oc
			/* 24 */ E::AddU32(23, 58), // +oc
			/* 25 */ E::MulU32(24, 13), // *ol
			/* 26 */ E::AddU32(25, 22), // bias output_offset
			/* 27 */ E::LoadF32 {
				binding: 1,
				address: 26,
			},
			/* 28 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 29 */ E::AddF32(28, 27), // acc + grad
			/* 30 */ E::ModU32(4, 9), // kernel
			/* 31 */ E::DivU32(4, 9), // quotient
			/* 32 */ E::ModU32(31, 6), // input_channel
			/* 33 */ E::DivU32(31, 6), // weight output_channel
			/* 34 */ E::MulU32(22, 10), // op*stride
			/* 35 */ E::MulU32(30, 12), // kernel*dilation
			/* 36 */ E::AddU32(34, 57), // input_position pre-shift base
			/* 37 */ E::GeU32(36, 11), // base >= padding
			/* 38 */ E::SubU32(36, 11), // input_position
			/* 39 */ E::LtU32(38, 8), // < input_length
			/* 40 */ E::AndBool(37, 39), // valid tap
			/* 41 */ E::MulU32(21, 6), // batch*ic
			/* 42 */ E::AddU32(41, 55), // +ic
			/* 43 */ E::MulU32(42, 8), // *il
			/* 44 */ E::AddU32(43, 38), // input_offset
			/* 45 */ E::MulU32(21, 7), // batch*oc
			/* 46 */ E::AddU32(45, 56), // +oc
			/* 47 */ E::MulU32(46, 13), // *ol
			/* 48 */ E::AddU32(47, 22), // weight output_offset
			/* 49 */ E::LoadF32 {
				binding: 0,
				address: 44,
			},
			/* 50 */ E::LoadF32 {
				binding: 1,
				address: 48,
			},
			/* 51 */ E::MulF32(49, 50), // tap product
			/* 52 */ E::AddF32(28, 51), // acc + product
			/* 53 */ E::ConstU32(0), // loop starts
			/* 54 */ E::ConstF32(0), // 0.0f
			/* 55 */ E::RegisterU32 { tile: 1, index: 0 }, // input_channel
			/* 56 */ E::RegisterU32 { tile: 1, index: 1 }, // output_channel
			/* 57 */ E::RegisterU32 { tile: 1, index: 2 }, // kernel * dilation
			/* 58 */ E::RegisterU32 { tile: 1, index: 3 }, // bias output_channel
			/* 59 */ E::RegisterU32 { tile: 1, index: 4 }, // parameter_index
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Output(0),
				Binding::Output(1),
			],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_length".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"dilation".into(),
				"output_length".into(),
				"parameter_count".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 5 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 15 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 54,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 32,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 33,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 35,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 20,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 4,
				},
				TileOp::LaneIf {
					cond: 18,
					body: vec![
						TileOp::LaneLoopRange {
							start: 53,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 53,
								step: 1,
								extent: Extent::Push(8),
								body: vec![TileOp::AssignScalar {
									tile: 0,
									index: 0,
									value: 29,
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 58,
							value: 28,
						},
					],
				},
				TileOp::LaneIf {
					cond: 19,
					body: vec![
						TileOp::LaneLoopRange {
							start: 53,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 53,
								step: 1,
								extent: Extent::Push(8),
								body: vec![TileOp::LaneIf {
									cond: 40,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 52,
									}],
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 2,
							address: 59,
							value: 28,
						},
					],
				},
			],
		)
	}

	/// `conv_1d` input adjoint — the donor's direct transposed-conv
	/// gather: one input element per lane accumulating
	/// `output_gradient × weight` over `output_channels × kernel`
	/// serial taps. The donor's signed `output_numerator` divisibility
	/// test is spelled as the u32 triple `base >= kernel*dilation`,
	/// `(base - kernel*dilation) % stride == 0` and the quotient's
	/// `output_length` bound — identical under the donor's
	/// `num >= 0` admit.
	/// `[output_gradient, weight, input_gradient]` + nine u32 scalars.
	pub(in crate::runtime) fn conv_1d_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // input_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_length
			/* 9 */ E::Push(4), // kernel_size
			/* 10 */ E::Push(5), // stride
			/* 11 */ E::Push(6), // padding
			/* 12 */ E::Push(7), // dilation
			/* 13 */ E::Push(8), // output_length
			/* 14 */ E::MulU32(5, 6), // batch*ic
			/* 15 */ E::MulU32(14, 8), // input_count
			/* 16 */ E::GeU32(4, 15), // out-of-range guard
			/* 17 */ E::ModU32(4, 8), // input_position
			/* 18 */ E::DivU32(4, 8), // quotient
			/* 19 */ E::ModU32(18, 6), // input_channel
			/* 20 */ E::DivU32(18, 6), // batch
			/* 21 */ E::LoopVar(2), // output_channel
			/* 22 */ E::LoopVar(3), // kernel
			/* 23 */ E::AddU32(17, 11), // ip + padding (pre-shift base)
			/* 24 */ E::MulU32(22, 12), // kernel*dilation
			/* 25 */ E::GeU32(49, 24), // numerator >= 0
			/* 26 */ E::SubU32(49, 24), // output_numerator
			/* 27 */ E::ModU32(26, 10), // % stride
			/* 28 */ E::ConstU32(0),
			/* 29 */ E::EqU32(27, 28), // divisible by stride
			/* 30 */ E::DivU32(26, 10), // output_position
			/* 31 */ E::LtU32(30, 13), // < output_length
			/* 32 */ E::AndBool(25, 29), // nonneg && divisible
			/* 33 */ E::AndBool(32, 31), // valid tap
			/* 34 */ E::MulU32(20, 7), // batch*oc
			/* 35 */ E::AddU32(51, 21), // +oc
			/* 36 */ E::MulU32(35, 13), // *ol
			/* 37 */ E::AddU32(36, 30), // output_offset
			/* 38 */ E::MulU32(21, 6), // oc*ic
			/* 39 */ E::AddU32(38, 50), // +ic
			/* 40 */ E::MulU32(39, 9), // *k
			/* 41 */ E::AddU32(40, 22), // weight_offset
			/* 42 */ E::LoadF32 {
				binding: 0,
				address: 37,
			},
			/* 43 */ E::LoadF32 {
				binding: 1,
				address: 41,
			},
			/* 44 */ E::MulF32(42, 43), // tap product
			/* 45 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 46 */ E::AddF32(45, 44), // acc + product
			/* 47 */ E::ConstU32(0), // loop starts
			/* 48 */ E::ConstF32(0), // 0.0f
			/* 49 */ E::RegisterU32 { tile: 1, index: 0 }, // ip + padding
			/* 50 */ E::RegisterU32 { tile: 1, index: 1 }, // input_channel
			/* 51 */ E::RegisterU32 { tile: 1, index: 2 }, // batch * oc
			/* 52 */ E::RegisterU32 { tile: 1, index: 3 }, // input_index
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_length".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"dilation".into(),
				"output_length".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 16 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 48,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 23,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 19,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 34,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 47,
					step: 1,
					extent: Extent::Push(2),
					body: vec![TileOp::LaneLoopRange {
						start: 47,
						step: 1,
						extent: Extent::Push(4),
						body: vec![TileOp::LaneIf {
							cond: 33,
							body: vec![TileOp::AssignScalar {
								tile: 0,
								index: 0,
								value: 46,
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 52,
					value: 45,
				},
			],
		)
	}

	/// Flat `conv_transpose_1d` template — one transposed-conv output
	/// element per 256-lane flat index, accumulating over input channels
	/// and kernel taps. Donor: `ml/nn/conv/conv_transpose_1d.slang`
	/// (`MlConvTranspose1dF32`; nine `ConvTranspose1dGeometry` scalars).
	/// The signed `input_numerator` spells as the same u32 predicate pair
	/// used by `conv_1d_backward`.
	pub(in crate::runtime) fn conv_transpose_1d() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // output_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_length
			/* 9 */ E::Push(4), // kernel_size
			/* 10 */ E::Push(5), // stride
			/* 11 */ E::Push(6), // padding
			/* 12 */ E::Push(7), // dilation
			/* 13 */ E::Push(8), // output_length
			/* 14 */ E::MulU32(5, 7), // batch*oc
			/* 15 */ E::MulU32(14, 13), // output_count
			/* 16 */ E::GeU32(4, 15), // out-of-range guard
			/* 17 */ E::ModU32(4, 13), // output_position
			/* 18 */ E::DivU32(4, 13), // quotient
			/* 19 */ E::ModU32(18, 7), // output_channel
			/* 20 */ E::DivU32(18, 7), // batch
			/* 21 */ E::LoopVar(2), // input_channel
			/* 22 */ E::LoopVar(3), // kernel
			/* 23 */ E::AddU32(17, 11), // op + padding (pre-shift base)
			/* 24 */ E::MulU32(22, 12), // kernel*dilation
			/* 25 */ E::GeU32(49, 24), // numerator >= 0
			/* 26 */ E::SubU32(49, 24), // input_numerator
			/* 27 */ E::ModU32(26, 10), // % stride
			/* 28 */ E::ConstU32(0),
			/* 29 */ E::EqU32(27, 28), // divisible by stride
			/* 30 */ E::DivU32(26, 10), // input_position
			/* 31 */ E::LtU32(30, 8), // < input_length
			/* 32 */ E::AndBool(25, 29), // nonneg && divisible
			/* 33 */ E::AndBool(32, 31), // valid tap
			/* 34 */ E::MulU32(20, 6), // batch*ic
			/* 35 */ E::AddU32(51, 21), // +ic
			/* 36 */ E::MulU32(35, 8), // *il
			/* 37 */ E::AddU32(36, 30), // input_offset
			/* 38 */ E::MulU32(21, 7), // ic*oc
			/* 39 */ E::AddU32(38, 50), // +oc
			/* 40 */ E::MulU32(39, 9), // *k
			/* 41 */ E::AddU32(40, 22), // weight_offset
			/* 42 */ E::LoadF32 {
				binding: 0,
				address: 37,
			},
			/* 43 */ E::LoadF32 {
				binding: 1,
				address: 41,
			},
			/* 44 */ E::MulF32(42, 43), // tap product
			/* 45 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 46 */ E::AddF32(45, 44), // acc + product
			/* 47 */ E::ConstU32(0), // loop starts
			/* 48 */ E::ConstF32(0), // 0.0f
			/* 49 */ E::RegisterU32 { tile: 1, index: 0 }, // op + padding
			/* 50 */ E::RegisterU32 { tile: 1, index: 1 }, // output_channel
			/* 51 */ E::RegisterU32 { tile: 1, index: 2 }, // batch * ic
			/* 52 */ E::RegisterU32 { tile: 1, index: 3 }, // output_index
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_length".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"dilation".into(),
				"output_length".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 16 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 48,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 23,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 19,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 34,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 47,
					step: 1,
					extent: Extent::Push(1),
					body: vec![TileOp::LaneLoopRange {
						start: 47,
						step: 1,
						extent: Extent::Push(4),
						body: vec![TileOp::LaneIf {
							cond: 33,
							body: vec![TileOp::AssignScalar {
								tile: 0,
								index: 0,
								value: 46,
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 45,
				},
			],
		)
	}

	/// Flat `conv_2d` input adjoint — one NCHW input-gradient element
	/// per 256-lane flat index, accumulating `output_gradient × weight`
	/// over `output_channels_per_group × kernel × kernel` serial taps.
	/// Donor: `ml/nn/conv/conv_2d_backward.slang`
	/// (`MlConv2dBackwardF32`; eleven `Conv2dGeometry` scalars). The
	/// donor's signed `output_{y,x}_numerator` divisibility tests spell
	/// as the u32 `base >= kernel && num % stride == 0` pairs used by
	/// `conv_1d_backward`; the per-group extents reach the computed
	/// `output_channels_per_group` through `Extent::Expr`.
	pub(in crate::runtime) fn conv_2d_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // input_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::Push(10), // groups
			/* 16 */ E::MulU32(5, 6), // batch*ic
			/* 17 */ E::MulU32(16, 8), // *ih
			/* 18 */ E::MulU32(17, 9), // input_count
			/* 19 */ E::GeU32(4, 18), // out-of-range guard
			/* 20 */ E::ModU32(4, 9), // input_x
			/* 21 */ E::DivU32(4, 9), // quotient
			/* 22 */ E::ModU32(21, 8), // input_y
			/* 23 */ E::DivU32(21, 8), // quotient
			/* 24 */ E::ModU32(23, 6), // input_channel
			/* 25 */ E::DivU32(23, 6), // batch
			/* 26 */ E::DivU32(6, 15), // input_channels_per_group
			/* 27 */ E::DivU32(7, 15), // output_channels_per_group
			/* 28 */ E::DivU32(24, 26), // group
			/* 29 */ E::MulU32(28, 26), // group*icpg
			/* 30 */ E::SubU32(24, 29), // local_input_channel
			/* 31 */ E::MulU32(28, 27), // output_channel_base
			/* 32 */ E::LoopVar(2), // local_output_channel
			/* 33 */ E::LoopVar(3), // kernel_y
			/* 34 */ E::LoopVar(5), // kernel_x
			/* 35 */ E::AddU32(79, 32), // output_channel
			/* 36 */ E::AddU32(76, 12), // iy + padding (pre-shift base)
			/* 37 */ E::GeU32(36, 33), // oy_numerator >= 0
			/* 38 */ E::SubU32(36, 33), // output_y_numerator
			/* 39 */ E::ModU32(38, 11), // % stride
			/* 40 */ E::ConstU32(0),
			/* 41 */ E::EqU32(39, 40), // y divisible
			/* 42 */ E::DivU32(38, 11), // output_y
			/* 43 */ E::LtU32(42, 13), // < output_height
			/* 44 */ E::AddU32(75, 12), // ix + padding (pre-shift base)
			/* 45 */ E::GeU32(44, 34), // ox_numerator >= 0
			/* 46 */ E::SubU32(44, 34), // output_x_numerator
			/* 47 */ E::ModU32(46, 11), // % stride
			/* 48 */ E::EqU32(47, 40), // x divisible
			/* 49 */ E::DivU32(46, 11), // output_x
			/* 50 */ E::LtU32(49, 14), // < output_width
			/* 51 */ E::AndBool(37, 41), // y nonneg && divisible
			/* 52 */ E::AndBool(51, 43), // y valid
			/* 53 */ E::AndBool(45, 48), // x nonneg && divisible
			/* 54 */ E::AndBool(53, 50), // x valid
			/* 55 */ E::AndBool(52, 54), // valid tap
			/* 56 */ E::MulU32(77, 7), // batch*oc
			/* 57 */ E::AddU32(56, 35), // + oc
			/* 58 */ E::MulU32(57, 13), // * oh
			/* 59 */ E::AddU32(58, 42), // + oy
			/* 60 */ E::MulU32(59, 14), // * ow
			/* 61 */ E::AddU32(60, 49), // output_offset
			/* 62 */ E::MulU32(35, 26), // oc*icpg
			/* 63 */ E::AddU32(62, 78), // + local_ic
			/* 64 */ E::MulU32(63, 10), // * k
			/* 65 */ E::AddU32(64, 33), // + ky
			/* 66 */ E::MulU32(65, 10), // * k
			/* 67 */ E::AddU32(66, 34), // weight_offset
			/* 68 */ E::LoadF32 {
				binding: 0,
				address: 61,
			},
			/* 69 */ E::LoadF32 {
				binding: 1,
				address: 67,
			},
			/* 70 */ E::MulF32(68, 69), // tap product
			/* 71 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 72 */ E::AddF32(71, 70), // acc + product
			/* 73 */ E::ConstU32(0), // loop starts
			/* 74 */ E::ConstF32(0), // 0.0f
			/* 75 */ E::RegisterU32 { tile: 1, index: 0 }, // input_x
			/* 76 */ E::RegisterU32 { tile: 1, index: 1 }, // input_y
			/* 77 */ E::RegisterU32 { tile: 1, index: 2 }, // batch
			/* 78 */ E::RegisterU32 { tile: 1, index: 3 }, // local_input_channel
			/* 79 */ E::RegisterU32 { tile: 1, index: 4 }, // output_channel_base
			/* 80 */ E::RegisterU32 { tile: 1, index: 5 }, // input_index
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
				"groups".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 6 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 19 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 74,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 20,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 22,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 25,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 30,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 31,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 5,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 73,
					step: 1,
					extent: Extent::Expr(27),
					body: vec![TileOp::LaneLoopRange {
						start: 73,
						step: 1,
						extent: Extent::Push(5),
						body: vec![TileOp::LaneIf {
							cond: 52,
							body: vec![TileOp::LaneLoopRange {
								start: 73,
								step: 1,
								extent: Extent::Push(5),
								body: vec![TileOp::LaneIf {
									cond: 54,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 72,
									}],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 80,
					value: 71,
				},
			],
		)
	}

	/// Flat `conv_2d` parameter adjoint — the donor's deterministic
	/// grouped weight/bias gradient program. Lanes below `weight_count`
	/// decompose to `(oc, local_ic, kernel_y, kernel_x)` and accumulate
	/// `input × output_gradient` over `batch × output_height ×
	/// output_width` serial taps; lanes at or above it decompose to the
	/// bias `output_channel` and reduce `output_gradient` over the same
	/// grid. Donor: `ml/nn/conv/conv_2d_parameter_backward.slang`
	/// (`MlConv2dParameterBackwardF32`; eleven geometry scalars plus
	/// `parameter_count`). `LaneIf` advances the emitter depth, so the
	/// triple loop reads `loop3`/`loop4`/`loop5`.
	pub(in crate::runtime) fn conv_2d_parameter_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // parameter_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::Push(10), // groups
			/* 16 */ E::Push(11), // parameter_count
			/* 17 */ E::GeU32(4, 16), // out-of-range guard
			/* 18 */ E::DivU32(6, 15), // input_channels_per_group
			/* 19 */ E::DivU32(7, 15), // output_channels_per_group
			/* 20 */ E::MulU32(7, 18), // oc*icpg
			/* 21 */ E::MulU32(20, 10), // *k
			/* 22 */ E::MulU32(21, 10), // weight_count
			/* 23 */ E::GeU32(4, 22), // bias-gradient lane
			/* 24 */ E::LtU32(4, 22), // weight-gradient lane
			/* 25 */ E::SubU32(4, 22), // bias output_channel
			/* 26 */ E::LoopVar(3), // batch
			/* 27 */ E::LoopVar(4), // output_y
			/* 28 */ E::LoopVar(5), // output_x
			/* 29 */ E::MulU32(26, 7), // batch*oc
			/* 30 */ E::AddU32(29, 83), // + bias oc
			/* 31 */ E::MulU32(30, 13), // * oh
			/* 32 */ E::AddU32(31, 27), // + oy
			/* 33 */ E::MulU32(32, 14), // * ow
			/* 34 */ E::AddU32(33, 28), // bias output_offset
			/* 35 */ E::LoadF32 {
				binding: 1,
				address: 34,
			},
			/* 36 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 37 */ E::AddF32(36, 35), // acc + grad
			/* 38 */ E::ModU32(4, 10), // kernel_x
			/* 39 */ E::DivU32(4, 10), // quotient
			/* 40 */ E::ModU32(39, 10), // kernel_y
			/* 41 */ E::DivU32(39, 10), // quotient
			/* 42 */ E::ModU32(41, 18), // local_input_channel
			/* 43 */ E::DivU32(41, 18), // output_channel
			/* 44 */ E::DivU32(43, 19), // group
			/* 45 */ E::MulU32(44, 18), // group*icpg
			/* 46 */ E::AddU32(45, 42), // input_channel
			/* 47 */ E::MulU32(27, 11), // oy*stride
			/* 48 */ E::AddU32(47, 80), // + ky (input_y base)
			/* 49 */ E::GeU32(48, 12), // base >= padding
			/* 50 */ E::SubU32(48, 12), // input_y
			/* 51 */ E::LtU32(50, 8), // < input_height
			/* 52 */ E::MulU32(84, 11), // ox*stride
			/* 53 */ E::AddU32(52, 79), // + kx (input_x base)
			/* 54 */ E::GeU32(53, 12), // base >= padding
			/* 55 */ E::SubU32(53, 12), // input_x
			/* 56 */ E::LtU32(55, 9), // < input_width
			/* 57 */ E::AndBool(49, 51), // y valid
			/* 58 */ E::AndBool(54, 56), // x valid
			/* 59 */ E::AndBool(57, 58), // valid tap
			/* 60 */ E::MulU32(26, 6), // batch*ic
			/* 61 */ E::AddU32(60, 81), // + ic
			/* 62 */ E::MulU32(61, 8), // * ih
			/* 63 */ E::AddU32(62, 50), // + iy
			/* 64 */ E::MulU32(63, 9), // * iw
			/* 65 */ E::AddU32(64, 55), // input_offset
			/* 66 */ E::MulU32(26, 7), // batch*oc
			/* 67 */ E::AddU32(66, 82), // + oc
			/* 68 */ E::MulU32(67, 13), // * oh
			/* 69 */ E::AddU32(68, 27), // + oy
			/* 70 */ E::MulU32(69, 14), // * ow
			/* 71 */ E::AddU32(70, 84), // weight output_offset
			/* 72 */ E::LoadF32 {
				binding: 0,
				address: 65,
			},
			/* 73 */ E::LoadF32 {
				binding: 1,
				address: 71,
			},
			/* 74 */ E::MulF32(72, 73), // tap product
			/* 75 */ E::AddF32(36, 74), // acc + product
			/* 76 */ E::ConstU32(0), // loop starts
			/* 77 */ E::ConstF32(0), // 0.0f
			/* 78 */ E::RegisterU32 { tile: 1, index: 0 }, // parameter_index
			/* 79 */ E::RegisterU32 { tile: 1, index: 1 }, // kernel_x
			/* 80 */ E::RegisterU32 { tile: 1, index: 2 }, // kernel_y
			/* 81 */ E::RegisterU32 { tile: 1, index: 3 }, // input_channel
			/* 82 */ E::RegisterU32 { tile: 1, index: 4 }, // output_channel
			/* 83 */ E::RegisterU32 { tile: 1, index: 5 }, // bias output_channel
			/* 84 */ E::LoopVar(6), // weight-branch output_x
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Output(0),
				Binding::Output(1),
			],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
				"groups".into(),
				"parameter_count".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 6 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 17 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 77,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 4,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 38,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 40,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 46,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 43,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 5,
					value: 25,
				},
				TileOp::LaneIf {
					cond: 23,
					body: vec![
						TileOp::LaneLoopRange {
							start: 76,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 76,
								step: 1,
								extent: Extent::Push(8),
								body: vec![TileOp::LaneLoopRange {
									start: 76,
									step: 1,
									extent: Extent::Push(9),
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 37,
									}],
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 83,
							value: 36,
						},
					],
				},
				TileOp::LaneIf {
					cond: 24,
					body: vec![
						TileOp::LaneLoopRange {
							start: 76,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 76,
								step: 1,
								extent: Extent::Push(8),
								body: vec![TileOp::LaneIf {
									cond: 57,
									body: vec![TileOp::LaneLoopRange {
										start: 76,
										step: 1,
										extent: Extent::Push(9),
										body: vec![TileOp::LaneIf {
											cond: 58,
											body: vec![TileOp::AssignScalar {
												tile: 0,
												index: 0,
												value: 75,
											}],
										}],
									}],
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 2,
							address: 78,
							value: 36,
						},
					],
				},
			],
		)
	}

	/// Flat `conv_transpose_2d` forward — one NCHW output element per
	/// 256-lane flat index, accumulating `input × weight` over
	/// `input_channels × kernel × kernel` serial taps with the
	/// transposed divisibility predicates. Donor:
	/// `ml/nn/conv/conv_transpose_2d.slang` (`MlConvTranspose2dF32`;
	/// ten `ConvTranspose2dGeometry` scalars, IOKK weight layout).
	pub(in crate::runtime) fn conv_transpose_2d() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // output_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::MulU32(5, 7), // batch*oc
			/* 16 */ E::MulU32(15, 13), // *oh
			/* 17 */ E::MulU32(16, 14), // output_count
			/* 18 */ E::GeU32(4, 17), // out-of-range guard
			/* 19 */ E::ModU32(4, 14), // output_x
			/* 20 */ E::DivU32(4, 14), // quotient
			/* 21 */ E::ModU32(20, 13), // output_y
			/* 22 */ E::DivU32(20, 13), // quotient
			/* 23 */ E::ModU32(22, 7), // output_channel
			/* 24 */ E::DivU32(22, 7), // batch
			/* 25 */ E::LoopVar(2), // input_channel
			/* 26 */ E::LoopVar(3), // kernel_y
			/* 27 */ E::LoopVar(5), // kernel_x
			/* 28 */ E::AddU32(21, 12), // oy + padding
			/* 29 */ E::GeU32(67, 26), // iy_numerator >= 0
			/* 30 */ E::SubU32(67, 26), // input_y_numerator
			/* 31 */ E::ModU32(30, 11), // % stride
			/* 32 */ E::ConstU32(0),
			/* 33 */ E::EqU32(31, 32), // y divisible
			/* 34 */ E::DivU32(30, 11), // input_y
			/* 35 */ E::LtU32(34, 8), // < input_height
			/* 36 */ E::AddU32(19, 12), // ox + padding
			/* 37 */ E::GeU32(68, 27), // ix_numerator >= 0
			/* 38 */ E::SubU32(68, 27), // input_x_numerator
			/* 39 */ E::ModU32(38, 11), // % stride
			/* 40 */ E::EqU32(39, 32), // x divisible
			/* 41 */ E::DivU32(38, 11), // input_x
			/* 42 */ E::LtU32(41, 9), // < input_width
			/* 43 */ E::AndBool(29, 33), // y nonneg && divisible
			/* 44 */ E::AndBool(43, 35), // y valid
			/* 45 */ E::AndBool(37, 40), // x nonneg && divisible
			/* 46 */ E::AndBool(45, 42), // x valid
			/* 47 */ E::AndBool(44, 46), // valid tap
			/* 48 */ E::MulU32(24, 6), // batch*ic
			/* 49 */ E::AddU32(70, 25), // + ic
			/* 50 */ E::MulU32(49, 8), // * ih
			/* 51 */ E::AddU32(50, 34), // + iy
			/* 52 */ E::MulU32(51, 9), // * iw
			/* 53 */ E::AddU32(52, 41), // input_offset
			/* 54 */ E::MulU32(25, 7), // ic*oc
			/* 55 */ E::AddU32(54, 69), // + oc
			/* 56 */ E::MulU32(55, 10), // * k
			/* 57 */ E::AddU32(56, 26), // + ky
			/* 58 */ E::MulU32(57, 10), // * k
			/* 59 */ E::AddU32(58, 27), // weight_offset
			/* 60 */ E::LoadF32 {
				binding: 0,
				address: 53,
			},
			/* 61 */ E::LoadF32 {
				binding: 1,
				address: 59,
			},
			/* 62 */ E::MulF32(60, 61), // tap product
			/* 63 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 64 */ E::AddF32(63, 62), // acc + product
			/* 65 */ E::ConstU32(0), // loop starts
			/* 66 */ E::ConstF32(0), // 0.0f
			/* 67 */ E::RegisterU32 { tile: 1, index: 0 }, // oy + padding
			/* 68 */ E::RegisterU32 { tile: 1, index: 1 }, // ox + padding
			/* 69 */ E::RegisterU32 { tile: 1, index: 2 }, // output_channel
			/* 70 */ E::RegisterU32 { tile: 1, index: 3 }, // batch * ic
			/* 71 */ E::RegisterU32 { tile: 1, index: 4 }, // output_index
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 5 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 18 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 66,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 36,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 23,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 48,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 65,
					step: 1,
					extent: Extent::Push(1),
					body: vec![TileOp::LaneLoopRange {
						start: 65,
						step: 1,
						extent: Extent::Push(5),
						body: vec![TileOp::LaneIf {
							cond: 44,
							body: vec![TileOp::LaneLoopRange {
								start: 65,
								step: 1,
								extent: Extent::Push(5),
								body: vec![TileOp::LaneIf {
									cond: 46,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 64,
									}],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 63,
				},
			],
		)
	}

	/// Flat `conv_transpose_2d_bias_add` epilogue — the donor's in-place
	/// NCHW channel-bias read-modify-write. Donor:
	/// `ml/nn/conv/conv_transpose_2d_bias_add.slang`
	/// (`MlConvTranspose2dBiasAddF32`; four scalars).
	pub(in crate::runtime) fn conv_transpose_2d_bias_add() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // channels
			/* 7 */ E::Push(2), // height
			/* 8 */ E::Push(3), // width
			/* 9 */ E::MulU32(7, 8), // plane_size
			/* 10 */ E::MulU32(5, 6), // batch*channels
			/* 11 */ E::MulU32(10, 9), // element_count
			/* 12 */ E::GeU32(4, 11), // out-of-range guard
			/* 13 */ E::DivU32(4, 9), // index / plane_size
			/* 14 */ E::ModU32(13, 6), // channel
			/* 15 */ E::LoadF32 {
				binding: 1,
				address: 14,
			},
			/* 16 */ E::LoadF32 {
				binding: 0,
				address: 4,
			},
			/* 17 */ E::AddF32(16, 15), // value + bias
		];
		Self::flat_program(
			vec![Binding::ReadWrite(0), Binding::Input(1)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
			],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 12 },
				TileOp::StoreScalar {
					binding: 0,
					address: 4,
					value: 17,
				},
			],
		)
	}

	/// Flat `conv_transpose_2d` input adjoint — one NCHW input-gradient
	/// element per 256-lane flat index, accumulating
	/// `output_gradient × weight` over `output_channels × kernel ×
	/// kernel` taps. Unlike `conv_2d_backward`, the transposed output
	/// coordinate `input * stride + kernel - padding` needs only the
	/// signed bounds check, not divisibility. Donor:
	/// `ml/nn/conv/conv_transpose_2d_backward.slang`
	/// (`MlConvTranspose2dBackwardF32`; ten scalars, IOKK weight layout).
	pub(in crate::runtime) fn conv_transpose_2d_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // input_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::MulU32(5, 6), // batch*ic
			/* 16 */ E::MulU32(15, 8), // *ih
			/* 17 */ E::MulU32(16, 9), // input_count
			/* 18 */ E::GeU32(4, 17), // out-of-range guard
			/* 19 */ E::ModU32(4, 9), // input_x
			/* 20 */ E::DivU32(4, 9), // quotient
			/* 21 */ E::ModU32(20, 8), // input_y
			/* 22 */ E::DivU32(20, 8), // quotient
			/* 23 */ E::ModU32(22, 6), // input_channel
			/* 24 */ E::DivU32(22, 6), // batch
			/* 25 */ E::LoopVar(2), // output_channel
			/* 26 */ E::LoopVar(3), // kernel_y
			/* 27 */ E::LoopVar(5), // kernel_x
			/* 28 */ E::MulU32(21, 11), // iy*stride
			/* 29 */ E::AddU32(60, 26), // + ky (oy base)
			/* 30 */ E::GeU32(29, 12), // base >= padding
			/* 31 */ E::SubU32(29, 12), // output_y
			/* 32 */ E::LtU32(31, 13), // < output_height
			/* 33 */ E::MulU32(19, 11), // ix*stride
			/* 34 */ E::AddU32(61, 27), // + kx (ox base)
			/* 35 */ E::GeU32(34, 12), // base >= padding
			/* 36 */ E::SubU32(34, 12), // output_x
			/* 37 */ E::LtU32(36, 14), // < output_width
			/* 38 */ E::AndBool(30, 32), // y valid
			/* 39 */ E::AndBool(35, 37), // x valid
			/* 40 */ E::AndBool(38, 39), // valid tap
			/* 41 */ E::MulU32(24, 7), // batch*oc
			/* 42 */ E::AddU32(62, 25), // + oc
			/* 43 */ E::MulU32(42, 13), // * oh
			/* 44 */ E::AddU32(43, 31), // + oy
			/* 45 */ E::MulU32(44, 14), // * ow
			/* 46 */ E::AddU32(45, 36), // output_offset
			/* 47 */ E::MulU32(23, 7), // ic*oc
			/* 48 */ E::AddU32(63, 25), // + oc
			/* 49 */ E::MulU32(48, 10), // * k
			/* 50 */ E::AddU32(49, 26), // + ky
			/* 51 */ E::MulU32(50, 10), // * k
			/* 52 */ E::AddU32(51, 27), // weight_offset
			/* 53 */ E::LoadF32 {
				binding: 0,
				address: 46,
			},
			/* 54 */ E::LoadF32 {
				binding: 1,
				address: 52,
			},
			/* 55 */ E::MulF32(53, 54), // tap product
			/* 56 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 57 */ E::AddF32(56, 55), // acc + product
			/* 58 */ E::ConstU32(0), // loop starts
			/* 59 */ E::ConstF32(0), // 0.0f
			/* 60 */ E::RegisterU32 { tile: 1, index: 0 }, // iy * stride
			/* 61 */ E::RegisterU32 { tile: 1, index: 1 }, // ix * stride
			/* 62 */ E::RegisterU32 { tile: 1, index: 2 }, // batch * oc
			/* 63 */ E::RegisterU32 { tile: 1, index: 3 }, // ic * oc
			/* 64 */ E::RegisterU32 { tile: 1, index: 4 }, // input_index
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 5 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 18 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 59,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 33,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 41,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 47,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 4,
				},
				TileOp::LaneLoopRange {
					start: 58,
					step: 1,
					extent: Extent::Push(2),
					body: vec![TileOp::LaneLoopRange {
						start: 58,
						step: 1,
						extent: Extent::Push(5),
						body: vec![TileOp::LaneIf {
							cond: 38,
							body: vec![TileOp::LaneLoopRange {
								start: 58,
								step: 1,
								extent: Extent::Push(5),
								body: vec![TileOp::LaneIf {
									cond: 39,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 57,
									}],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 56,
				},
			],
		)
	}

	/// Flat `conv_transpose_2d` parameter adjoint — the donor's
	/// deterministic IOKK weight/bias gradient program. Weight lanes
	/// decompose to `(input_channel, output_channel, ky, kx)` and
	/// accumulate `input × output_gradient` over `batch × input_height
	/// × input_width` serial taps — the transposed input-space walk
	/// with `input * stride + kernel - padding` bounds checks; bias
	/// lanes reduce `output_gradient` over `batch × oh × ow`. Donor:
	/// `ml/nn/conv/conv_transpose_2d_parameter_backward.slang`
	/// (`MlConvTranspose2dParameterBackwardF32`; ten geometry scalars
	/// plus `parameter_count`).
	pub(in crate::runtime) fn conv_transpose_2d_parameter_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // parameter_index
			/* 5 */ E::Push(0), // batch_size
			/* 6 */ E::Push(1), // input_channels
			/* 7 */ E::Push(2), // output_channels
			/* 8 */ E::Push(3), // input_height
			/* 9 */ E::Push(4), // input_width
			/* 10 */ E::Push(5), // kernel_size
			/* 11 */ E::Push(6), // stride
			/* 12 */ E::Push(7), // padding
			/* 13 */ E::Push(8), // output_height
			/* 14 */ E::Push(9), // output_width
			/* 15 */ E::Push(10), // parameter_count
			/* 16 */ E::GeU32(4, 15), // out-of-range guard
			/* 17 */ E::MulU32(6, 7), // ic*oc
			/* 18 */ E::MulU32(17, 10), // *k
			/* 19 */ E::MulU32(18, 10), // weight_count (IOKK)
			/* 20 */ E::GeU32(4, 19), // bias-gradient lane
			/* 21 */ E::LtU32(4, 19), // weight-gradient lane
			/* 22 */ E::SubU32(4, 19), // bias output_channel
			/* 23 */ E::LoopVar(3), // batch
			/* 24 */ E::LoopVar(4), // output_y / input_y
			/* 25 */ E::LoopVar(5), // output_x
			/* 26 */ E::MulU32(23, 7), // batch*oc
			/* 27 */ E::AddU32(26, 72), // + bias oc
			/* 28 */ E::MulU32(27, 13), // * oh
			/* 29 */ E::AddU32(28, 24), // + oy
			/* 30 */ E::MulU32(29, 14), // * ow
			/* 31 */ E::AddU32(30, 25), // bias output_offset
			/* 32 */ E::LoadF32 {
				binding: 1,
				address: 31,
			},
			/* 33 */ E::RegisterF32 { tile: 0, index: 0 }, // acc
			/* 34 */ E::AddF32(33, 32), // acc + grad
			/* 35 */ E::ModU32(4, 10), // kernel_x
			/* 36 */ E::DivU32(4, 10), // quotient
			/* 37 */ E::ModU32(36, 10), // kernel_y
			/* 38 */ E::DivU32(36, 10), // quotient
			/* 39 */ E::ModU32(38, 7), // output_channel
			/* 40 */ E::DivU32(38, 7), // input_channel
			/* 41 */ E::MulU32(24, 11), // iy*stride
			/* 42 */ E::AddU32(41, 74), // + ky (oy base)
			/* 43 */ E::GeU32(42, 12), // base >= padding
			/* 44 */ E::SubU32(42, 12), // output_y
			/* 45 */ E::LtU32(44, 13), // < output_height
			/* 46 */ E::MulU32(78, 11), // ix*stride
			/* 47 */ E::AddU32(46, 73), // + kx (ox base)
			/* 48 */ E::GeU32(47, 12), // base >= padding
			/* 49 */ E::SubU32(47, 12), // output_x
			/* 50 */ E::LtU32(49, 14), // < output_width
			/* 51 */ E::AndBool(43, 45), // y valid
			/* 52 */ E::AndBool(48, 50), // x valid
			/* 53 */ E::AndBool(51, 52), // valid tap
			/* 54 */ E::MulU32(23, 6), // batch*ic
			/* 55 */ E::AddU32(54, 76), // + ic
			/* 56 */ E::MulU32(55, 8), // * ih
			/* 57 */ E::AddU32(56, 24), // + iy
			/* 58 */ E::MulU32(57, 9), // * iw
			/* 59 */ E::AddU32(58, 78), // input_offset
			/* 60 */ E::MulU32(23, 7), // batch*oc
			/* 61 */ E::AddU32(60, 75), // + oc
			/* 62 */ E::MulU32(61, 13), // * oh
			/* 63 */ E::AddU32(62, 44), // + oy
			/* 64 */ E::MulU32(63, 14), // * ow
			/* 65 */ E::AddU32(64, 49), // output_offset
			/* 66 */ E::LoadF32 {
				binding: 0,
				address: 59,
			},
			/* 67 */ E::LoadF32 {
				binding: 1,
				address: 65,
			},
			/* 68 */ E::MulF32(66, 67), // tap product
			/* 69 */ E::AddF32(33, 68), // acc + product
			/* 70 */ E::ConstU32(0), // loop starts
			/* 71 */ E::ConstF32(0), // 0.0f
			/* 72 */ E::RegisterU32 { tile: 1, index: 0 }, // bias output_channel
			/* 73 */ E::RegisterU32 { tile: 1, index: 1 }, // kernel_x
			/* 74 */ E::RegisterU32 { tile: 1, index: 2 }, // kernel_y
			/* 75 */ E::RegisterU32 { tile: 1, index: 3 }, // output_channel
			/* 76 */ E::RegisterU32 { tile: 1, index: 4 }, // input_channel
			/* 77 */ E::RegisterU32 { tile: 1, index: 5 }, // parameter_index
			/* 78 */ E::LoopVar(6), // input_x
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Output(0),
				Binding::Output(1),
			],
			vec![
				"batch_size".into(),
				"input_channels".into(),
				"output_channels".into(),
				"input_height".into(),
				"input_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
				"output_height".into(),
				"output_width".into(),
				"parameter_count".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 6 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 16 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 71,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 22,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 35,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 37,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 39,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 40,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 5,
					value: 4,
				},
				TileOp::LaneIf {
					cond: 20,
					body: vec![
						TileOp::LaneLoopRange {
							start: 70,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 70,
								step: 1,
								extent: Extent::Push(8),
								body: vec![TileOp::LaneLoopRange {
									start: 70,
									step: 1,
									extent: Extent::Push(9),
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 34,
									}],
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 72,
							value: 33,
						},
					],
				},
				TileOp::LaneIf {
					cond: 21,
					body: vec![
						TileOp::LaneLoopRange {
							start: 70,
							step: 1,
							extent: Extent::Push(0),
							body: vec![TileOp::LaneLoopRange {
								start: 70,
								step: 1,
								extent: Extent::Push(3),
								body: vec![TileOp::LaneIf {
									cond: 51,
									body: vec![TileOp::LaneLoopRange {
										start: 70,
										step: 1,
										extent: Extent::Push(4),
										body: vec![TileOp::LaneIf {
											cond: 52,
											body: vec![TileOp::AssignScalar {
												tile: 0,
												index: 0,
												value: 69,
											}],
										}],
									}],
								}],
							}],
						},
						TileOp::StoreScalar {
							binding: 2,
							address: 77,
							value: 33,
						},
					],
				},
			],
		)
	}

	/// `conv_1d_bias_transpose` fused-tail family — the 1024-lane 32×32
	/// shared-tile NCL transpose with the channel bias folded into the
	/// tile load. Donor: `ml/nn/conv/conv_1d_bias_transpose.slang`
	/// (`MlConv1dBiasTransposeF32`; three scalars, `[projected, bias,
	/// output]`).
	pub(in crate::runtime) fn conv_1d_bias_transpose() -> Self {
		Self::conv_1d_bias_transpose_program(ConvTailStore::Plain)
	}

	/// `conv_1d_bias_transpose_relu` — the same fused transpose with
	/// `max(0.0f, ·)` at the store
	/// (`MlConv1dBiasTransposeReluF32`).
	pub(in crate::runtime) fn conv_1d_bias_transpose_relu() -> Self {
		Self::conv_1d_bias_transpose_program(ConvTailStore::Relu)
	}

	/// `conv_1d_bias_transpose_residual_relu` — the ReLU store plus a
	/// second `sum` binding receiving `activated + residual` at the
	/// same offset (`MlConv1dBiasTransposeResidualReluF32`; five
	/// bindings `[projected, bias, output, residual, sum]`).
	pub(in crate::runtime) fn conv_1d_bias_transpose_residual_relu() -> Self {
		Self::conv_1d_bias_transpose_program(ConvTailStore::ResidualRelu)
	}

	fn conv_1d_bias_transpose_program(store: ConvTailStore) -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Lane,
			/* 1 */ E::ConstU32(32),
			/* 2 */ E::DivU32(0, 1), // thread_row (donor group_thread_id.y)
			/* 3 */ E::ModU32(0, 1), // thread_column (donor .x)
			/* 4 */ E::GroupX,
			/* 5 */ E::GroupY,
			/* 6 */ E::GroupZ, // batch
			/* 7 */ E::Push(0), // batch_size
			/* 8 */ E::Push(1), // rows
			/* 9 */ E::Push(2), // columns
			/* 10 */ E::MulU32(5, 1), // group_y * TILE
			/* 11 */ E::AddU32(10, 2), // input_row
			/* 12 */ E::MulU32(4, 1), // group_x * TILE
			/* 13 */ E::AddU32(12, 3), // input_column
			/* 14 */ E::LtU32(11, 8), // input_row < rows
			/* 15 */ E::LtU32(13, 9), // input_column < columns
			/* 16 */ E::AndBool(14, 15), // valid input tap
			/* 17 */ E::MulU32(8, 9), // rows * columns
			/* 18 */ E::MulU32(6, 17), // batch input base
			/* 19 */ E::MulU32(11, 9), // input_row * columns
			/* 20 */ E::AddU32(19, 13), // + input_column
			/* 21 */ E::AddU32(18, 20), // input address
			/* 22 */ E::LoadF32 {
				binding: 0,
				address: 21,
			},
			/* 23 */ E::ConstF32(0), // 0.0f
			/* 24 */ E::ConstU32(33), // TILE_PAD
			/* 25 */ E::MulU32(2, 24), // thread_row * TILE_PAD
			/* 26 */ E::AddU32(25, 3), // staged write offset
			/* 27 */ E::AddU32(12, 2), // output_row = group_x*TILE + thread_row
			/* 28 */ E::AddU32(10, 3), // output_column = group_y*TILE + thread_column
			/* 29 */ E::LtU32(27, 9), // output_row < columns
			/* 30 */ E::LtU32(28, 8), // output_column < rows
			/* 31 */ E::AndBool(29, 30), // valid output tap
			/* 32 */ E::MulU32(27, 8), // output_row * rows
			/* 33 */ E::AddU32(32, 28), // + output_column
			/* 34 */ E::AddU32(18, 33), // output address (same batch base)
			/* 35 */ E::MulU32(3, 24), // thread_column * TILE_PAD
			/* 36 */ E::AddU32(35, 2), // staged read offset
			/* 37 */ E::SharedF32 {
				tile: 0,
				offset: 36,
			},
			/* 38 */ E::LoadF32 {
				binding: 1,
				address: 13,
			}, // bias[input_column]
			/* 39 */ E::AddF32(22, 38), // biased tile value
			/* 40 */ E::MaxF32(37, 23), // activated
			/* 41 */
			E::LoadF32 {
				binding: 3,
				address: 34,
			}, // residual[output_offset]
			/* 42 */ E::AddF32(40, 41), // activated + residual
		];
		let mut bindings = vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)];
		if store == ConvTailStore::ResidualRelu {
			bindings.push(Binding::Input(2));
			bindings.push(Binding::Output(1));
		}
		let epilogue = match store {
			ConvTailStore::Plain => vec![TileOp::StoreScalar {
				binding: 2,
				address: 34,
				value: 37,
			}],
			ConvTailStore::Relu => vec![TileOp::StoreScalar {
				binding: 2,
				address: 34,
				value: 40,
			}],
			ConvTailStore::ResidualRelu => vec![
				TileOp::StoreScalar {
					binding: 2,
					address: 34,
					value: 40,
				},
				TileOp::StoreScalar {
					binding: 4,
					address: 34,
					value: 42,
				},
			],
		};
		Self {
			bindings,
			push_scalars: vec!["batch_size".into(), "rows".into(), "columns".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 1024,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![TileDecl::Shared {
				rows: 32,
				columns: 32,
				stride: 33,
			}],
			lane_exprs,
			ops: vec![
				TileOp::StoreShared {
					tile: 0,
					offset: 26,
					value: 23,
				},
				TileOp::LaneIf {
					cond: 16,
					body: vec![TileOp::StoreShared {
						tile: 0,
						offset: 26,
						value: 39,
					}],
				},
				TileOp::Barrier,
				TileOp::LaneIf {
					cond: 31,
					body: epilogue,
				},
			],
		}
	}

	/// `conv_1d_bias_copy` — the out-of-place bias-add the tail fusion
	/// substitutes when the consumer is the inverse transpose. Donor:
	/// `ml/nn/conv/conv_1d_bias_copy.slang` (`MlConv1dBiasCopyF32`;
	/// `[input, bias, output]` + `[rows, columns]`).
	pub(in crate::runtime) fn conv_1d_bias_copy() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // index
			/* 5 */ E::Push(0), // rows
			/* 6 */ E::Push(1), // columns
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7), // out-of-range guard
			/* 9 */ E::ModU32(4, 6), // column
			/* 10 */ E::LoadF32 {
				binding: 1,
				address: 9,
			},
			/* 11 */ E::LoadF32 {
				binding: 0,
				address: 4,
			},
			/* 12 */ E::AddF32(11, 10), // input + bias[column]
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec!["rows".into(), "columns".into()],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 12,
				},
			],
		)
	}

	/// `conv_1d_bias_transpose_norm` — the row-per-workgroup fused
	/// channel normalization: each 256-lane group folds the biased conv
	/// row `input + conv_bias` lane-strided, tree-reduces mean and
	/// variance with the donor's explicit inter-pass barrier, then
	/// republishes both the biased conv output and the normalized (or
	/// ReLU-normalized) output at `[B, C, T]` offsets. Donor:
	/// `ml/nn/conv/conv_1d_bias_transpose_norm{,_relu}.slang`
	/// (`MlConv1dBiasTransposeNorm{,Relu}F32`; three u32 scalars plus
	/// `epsilon`, six bindings). The biased value is recomputed in the
	/// variance and store passes rather than held in a per-lane array —
	/// the loads are identical, so the result is bit-exact.
	pub(in crate::runtime) fn conv_1d_bias_transpose_norm(relu: bool) -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // batch
			/* 1 */ E::Push(1), // channels
			/* 2 */ E::Push(2), // sequence_length
			/* 3 */ E::GroupX, // row
			/* 4 */ E::MulU32(0, 2), // batch * sequence_length (bound)
			/* 5 */ E::MulU32(3, 1), // row_base = row * channels
			/* 6 */ E::Induction, // channel
			/* 7 */ E::AddU32(5, 6), // row_base + channel
			/* 8 */ E::LoadF32 {
				binding: 0,
				address: 7,
			},
			/* 9 */ E::LoadF32 {
				binding: 1,
				address: 6,
			}, // conv_bias[channel]
			/* 10 */ E::AddF32(8, 9), // biased value
			/* 11 */ E::ConstF32(0), // 0.0f
			/* 12 */ E::Lane,
			/* 13 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 14 */ E::ConstU32(0),
			/* 15 */ E::SharedF32 {
				tile: 0,
				offset: 14,
			}, // reduction[0]
			/* 16 */ E::F32FromU32(1), // float(channels)
			/* 17 */ E::DivF32(15, 16), // mean
			/* 18 */ E::RegisterF32 { tile: 2, index: 0 }, // mean reg
			/* 19 */ E::SubF32(10, 18), // centered
			/* 20 */ E::MulF32(19, 19), // centered²
			/* 21 */ E::DivF32(15, 16), // variance (post second reduce)
			/* 22 */ E::PushF32(0), // epsilon
			/* 23 */ E::AddF32(21, 22), // variance + eps
			/* 24 */ E::RsqrtF32(23), // inverse_stddev
			/* 25 */ E::RegisterF32 { tile: 3, index: 0 }, // inv reg
			/* 26 */ E::LoopVar(2), // store channel
			/* 27 */ E::DivU32(3, 2), // row / sequence_length
			/* 28 */ E::MulU32(27, 1), // * channels
			/* 29 */ E::MulU32(28, 2), // * sequence_length
			/* 30 */ E::ModU32(3, 2), // row % sequence_length
			/* 31 */ E::AddU32(29, 30), // conv_base
			/* 32 */ E::MulU32(26, 2), // channel * sequence_length
			/* 33 */ E::AddU32(31, 32), // element
			/* 34 */ E::AddU32(5, 26), // row_base + channel
			/* 35 */ E::LoadF32 {
				binding: 0,
				address: 34,
			},
			/* 36 */ E::LoadF32 {
				binding: 1,
				address: 26,
			},
			/* 37 */ E::AddF32(35, 36), // biased (republished value)
			/* 38 */ E::SubF32(37, 18), // normalized pre-scale
			/* 39 */ E::MulF32(38, 25), // normalized
			/* 40 */ E::LoadF32 {
				binding: 2,
				address: 26,
			}, // weight[channel]
			/* 41 */ E::MulF32(39, 40), // n * w
			/* 42 */ E::LoadF32 {
				binding: 3,
				address: 26,
			}, // bias[channel]
			/* 43 */ E::AddF32(41, 42), // output
			/* 44 */ E::MaxF32(43, 11), // relu output
		];
		let output_value = if relu { 44 } else { 43 };
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec!["batch".into(), "channels".into(), "sequence_length".into()],
			push_f32_scalars: vec!["epsilon".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 256,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 256,
					stride: 256,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 4 },
				TileOp::LaneFold {
					acc: 1,
					init: 11,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 10,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 12,
					value: 13,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 17,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 1,
					init: 11,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 20,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 12,
					value: 13,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 24,
				},
				TileOp::LaneLoopRange {
					start: 12,
					step: 256,
					extent: Extent::Push(1),
					body: vec![
						TileOp::StoreScalar {
							binding: 4,
							address: 33,
							value: output_value,
						},
						TileOp::StoreScalar {
							binding: 5,
							address: 33,
							value: 37,
						},
					],
				},
			],
		}
	}

	/// `avg_pool_2d` — the donor's NCHW average pool: the flat 256-lane
	/// workgroup spells the donor's `numthreads(16,16,1)` dispatch as
	/// `output_y = group_id.x * 16 + lane % 16`,
	/// `output_x = group_id.y * 16 + lane / 16`,
	/// `batch_channel = group_id.z`. Signed window starts are the
	/// established u32 pair `base >= padding && base - padding < extent`;
	/// the valid-tap `count` lives in a u32 register and the store
	/// selects `count > 0 ? sum / count : 0` exactly like the donor.
	/// `[input, output]` + nine u32 scalars ABI.
	pub(in crate::runtime) fn avg_pool_2d() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // output_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // output_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::Push(6), // kernel_size
			/* 18 */ E::Push(7), // stride
			/* 19 */ E::Push(8), // padding
			/* 20 */ E::MulU32(11, 12), // batch * channels
			/* 21 */ E::GeU32(2, 20),
			/* 22 */ E::GeU32(7, 15),
			/* 23 */ E::GeU32(10, 16),
			/* 24 */ E::OrBool(21, 22),
			/* 25 */ E::OrBool(24, 23), // out-of-range guard
			/* 26 */ E::DivU32(2, 12), // batch
			/* 27 */ E::ModU32(2, 12), // channel
			/* 28 */ E::MulU32(7, 18), // output_y * stride
			/* 29 */ E::MulU32(10, 18), // output_x * stride
			/* 30 */ E::LoopVar(2), // kernel_y
			/* 31 */ E::AddU32(67, 30), // input_y pre-shift base
			/* 32 */ E::GeU32(31, 19), // base >= padding
			/* 33 */ E::SubU32(31, 19), // input_y
			/* 34 */ E::LtU32(33, 13), // input_y < input_height
			/* 35 */ E::AndBool(32, 34), // input_y in range
			/* 36 */ E::LoopVar(4), // kernel_x
			/* 37 */ E::AddU32(68, 36), // input_x pre-shift base
			/* 38 */ E::GeU32(37, 19), // base >= padding
			/* 39 */ E::SubU32(37, 19), // input_x
			/* 40 */ E::LtU32(39, 14), // input_x < input_width
			/* 41 */ E::AndBool(38, 40), // input_x in range
			/* 42 */ E::AndBool(35, 41), // valid tap
			/* 43 */ E::MulU32(26, 12), // batch * channels
			/* 44 */ E::AddU32(43, 27), // + channel → batch_channel
			/* 45 */ E::MulU32(33, 14), // input_y * input_width
			/* 46 */ E::AddU32(45, 39), // + input_x
			/* 47 */ E::AddU32(69, 46), // bc plane base + taps
			/* 48 */ E::AddU32(47, 55), // input_offset
			/* 49 */ E::LoadF32 {
				binding: 0,
				address: 48,
			},
			/* 50 */ E::RegisterF32 { tile: 0, index: 0 }, // sum
			/* 51 */ E::AddF32(50, 49), // sum + tap
			/* 52 */ E::RegisterU32 { tile: 1, index: 0 }, // count
			/* 53 */ E::ConstU32(1),
			/* 54 */ E::AddU32(52, 53), // count + 1
			/* 55 */ E::ConstU32(0),
			/* 56 */ E::LtU32(55, 52), // count > 0
			/* 57 */ E::F32FromU32(52), // float(count)
			/* 58 */ E::DivF32(50, 57), // sum / count
			/* 59 */ E::ConstF32(0), // 0.0f
			/* 60 */
			E::SelectF32 {
				cond: 56,
				if_true: 58,
				if_false: 59,
			}, // average
			/* 61 */ E::MulU32(44, 15), // bc * output_height
			/* 62 */ E::AddU32(61, 7), // + output_y
			/* 63 */ E::MulU32(62, 16), // * output_width
			/* 64 */ E::AddU32(63, 10), // output_offset
			/* 65 */ E::MulU32(44, 13), // bc * input_height
			/* 66 */ E::MulU32(65, 14), // bc * ih * iw
			/* 67 */ E::RegisterU32 { tile: 1, index: 1 }, // output_y * stride
			/* 68 */ E::RegisterU32 { tile: 1, index: 2 }, // output_x * stride
			/* 69 */ E::RegisterU32 { tile: 1, index: 3 }, // bc * ih * iw
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 25 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 59,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 55,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 29,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 66,
				},
				TileOp::LaneLoopRange {
					start: 55,
					step: 1,
					extent: Extent::Push(6),
					body: vec![TileOp::LaneIf {
						cond: 35,
						body: vec![TileOp::LaneLoopRange {
							start: 55,
							step: 1,
							extent: Extent::Push(6),
							body: vec![TileOp::LaneIf {
								cond: 41,
								body: vec![
									TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 51,
									},
									TileOp::AssignScalarU32 {
										tile: 1,
										index: 0,
										value: 54,
									},
								],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 64,
					value: 60,
				},
			],
		)
	}

	/// `max_pool_2d` — the donor's NCHW max pool plus arg index: same
	/// 16×16 3D-decomposed lanes and window predicates as `avg_pool_2d`,
	/// but each lane tracks `maximum` (seeded `-FLT_MAX`) and the
	/// absolute `maximum_index` behind the donor's strict
	/// `value > maximum` update so ties keep the earliest tap.
	/// `[input, output, indices]` + nine u32 scalars ABI.
	pub(in crate::runtime) fn max_pool_2d() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // output_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // output_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::Push(6), // kernel_size
			/* 18 */ E::Push(7), // stride
			/* 19 */ E::Push(8), // padding
			/* 20 */ E::MulU32(11, 12), // batch * channels
			/* 21 */ E::GeU32(2, 20),
			/* 22 */ E::GeU32(7, 15),
			/* 23 */ E::GeU32(10, 16),
			/* 24 */ E::OrBool(21, 22),
			/* 25 */ E::OrBool(24, 23), // out-of-range guard
			/* 26 */ E::DivU32(2, 12), // batch
			/* 27 */ E::ModU32(2, 12), // channel
			/* 28 */ E::MulU32(7, 18), // output_y * stride
			/* 29 */ E::MulU32(10, 18), // output_x * stride
			/* 30 */ E::LoopVar(2), // kernel_y
			/* 31 */ E::AddU32(61, 30), // input_y pre-shift base
			/* 32 */ E::GeU32(31, 19), // base >= padding
			/* 33 */ E::SubU32(31, 19), // input_y
			/* 34 */ E::LtU32(33, 13), // input_y < input_height
			/* 35 */ E::AndBool(32, 34), // input_y in range
			/* 36 */ E::LoopVar(4), // kernel_x
			/* 37 */ E::AddU32(62, 36), // input_x pre-shift base
			/* 38 */ E::GeU32(37, 19), // base >= padding
			/* 39 */ E::SubU32(37, 19), // input_x
			/* 40 */ E::LtU32(39, 14), // input_x < input_width
			/* 41 */ E::AndBool(38, 40), // input_x in range
			/* 42 */ E::AndBool(35, 41), // valid tap
			/* 43 */ E::MulU32(26, 12), // batch * channels
			/* 44 */ E::AddU32(43, 27), // + channel → batch_channel
			/* 45 */ E::MulU32(33, 14), // input_y * input_width
			/* 46 */ E::AddU32(45, 39), // + input_x
			/* 47 */ E::AddU32(63, 46), // bc plane base + taps
			/* 48 */ E::AddU32(47, 53), // input_offset
			/* 49 */ E::LoadF32 {
				binding: 0,
				address: 48,
			},
			/* 50 */ E::RegisterF32 { tile: 0, index: 0 }, // maximum
			/* 51 */ E::GtF32(49, 50), // value > maximum
			/* 52 */ E::RegisterU32 { tile: 1, index: 0 }, // maximum_index
			/* 53 */ E::ConstU32(0),
			/* 54 */ E::ConstF32(0xff7fffff), // -3.402823466e+38f
			/* 55 */ E::MulU32(44, 15), // bc * output_height
			/* 56 */ E::AddU32(55, 7), // + output_y
			/* 57 */ E::MulU32(56, 16), // * output_width
			/* 58 */ E::AddU32(57, 10), // output_offset
			/* 59 */ E::MulU32(44, 13), // bc * input_height
			/* 60 */ E::MulU32(59, 14), // bc * ih * iw
			/* 61 */ E::RegisterU32 { tile: 1, index: 1 }, // output_y * stride
			/* 62 */ E::RegisterU32 { tile: 1, index: 2 }, // output_x * stride
			/* 63 */ E::RegisterU32 { tile: 1, index: 3 }, // bc * ih * iw
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0), Binding::Output(1)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 25 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 54,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 53,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 29,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 60,
				},
				TileOp::LaneLoopRange {
					start: 53,
					step: 1,
					extent: Extent::Push(6),
					body: vec![TileOp::LaneIf {
						cond: 35,
						body: vec![TileOp::LaneLoopRange {
							start: 53,
							step: 1,
							extent: Extent::Push(6),
							body: vec![TileOp::LaneIf {
								cond: 41,
								body: vec![TileOp::LaneIf {
									cond: 51,
									body: vec![
										TileOp::AssignScalar {
											tile: 0,
											index: 0,
											value: 49,
										},
										TileOp::AssignScalarU32 {
											tile: 1,
											index: 0,
											value: 48,
										},
									],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 58,
					value: 50,
				},
				TileOp::StoreScalarU32 {
					binding: 2,
					address: 58,
					value: 52,
				},
			],
		)
	}

	/// `adaptive_avg_pool_2d` — the donor's computed-window average pool:
	/// same lane decomposition, but the tap range is the unsigned
	/// `start = output * input / output_extent` /
	/// `end = (output + 1) * input + extent - 1) / extent` window folded
	/// per lane, with `count = window_height * window_width`.
	/// `[input, output]` + six u32 scalars ABI.
	pub(in crate::runtime) fn adaptive_avg_pool_2d() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // output_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // output_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::MulU32(11, 12), // batch * channels
			/* 18 */ E::GeU32(2, 17),
			/* 19 */ E::GeU32(7, 15),
			/* 20 */ E::GeU32(10, 16),
			/* 21 */ E::OrBool(18, 19),
			/* 22 */ E::OrBool(21, 20), // out-of-range guard
			/* 23 */ E::DivU32(2, 12), // batch
			/* 24 */ E::ModU32(2, 12), // channel
			/* 25 */ E::ConstU32(1),
			/* 26 */ E::MulU32(7, 13), // output_y * input_height
			/* 27 */ E::DivU32(26, 15), // input_y_start
			/* 28 */ E::MulU32(10, 14), // output_x * input_width
			/* 29 */ E::DivU32(28, 16), // input_x_start
			/* 30 */ E::AddU32(7, 25), // output_y + 1
			/* 31 */ E::MulU32(30, 13), // * input_height
			/* 32 */ E::AddU32(31, 15), // + output_height
			/* 33 */ E::SubU32(32, 25), // - 1
			/* 34 */ E::DivU32(33, 15), // input_y_end
			/* 35 */ E::AddU32(10, 25), // output_x + 1
			/* 36 */ E::MulU32(35, 14), // * input_width
			/* 37 */ E::AddU32(36, 16), // + output_width
			/* 38 */ E::SubU32(37, 25), // - 1
			/* 39 */ E::DivU32(38, 16), // input_x_end
			/* 40 */ E::LoopVar(2), // input_y
			/* 41 */ E::LoopVar(3), // input_x
			/* 42 */ E::MulU32(23, 12), // batch * channels
			/* 43 */ E::AddU32(42, 24), // + channel → batch_channel
			/* 44 */ E::MulU32(40, 14), // input_y * input_width
			/* 45 */ E::AddU32(44, 41), // + input_x
			/* 46 */ E::AddU32(63, 45), // bc plane base + taps
			/* 47 */ E::AddU32(63, 45), // input_offset
			/* 48 */ E::LoadF32 {
				binding: 0,
				address: 47,
			},
			/* 49 */ E::RegisterF32 { tile: 0, index: 0 }, // sum
			/* 50 */ E::AddF32(49, 48), // sum + tap
			/* 51 */ E::SubU32(34, 27), // window height
			/* 52 */ E::SubU32(39, 29), // window width
			/* 53 */ E::MulU32(51, 52), // count
			/* 54 */ E::F32FromU32(53), // float(count)
			/* 55 */ E::DivF32(49, 54), // average
			/* 56 */ E::MulU32(43, 15), // bc * output_height
			/* 57 */ E::AddU32(56, 7), // + output_y
			/* 58 */ E::MulU32(57, 16), // * output_width
			/* 59 */ E::AddU32(58, 10), // output_offset
			/* 60 */ E::ConstF32(0), // 0.0f
			/* 61 */ E::MulU32(43, 13), // bc * input_height
			/* 62 */ E::MulU32(61, 14), // bc * ih * iw
			/* 63 */ E::RegisterU32 { tile: 1, index: 0 }, // bc * ih * iw
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 22 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 60,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 62,
				},
				TileOp::LaneLoopRange {
					start: 27,
					step: 1,
					extent: Extent::Expr(34),
					body: vec![TileOp::LaneLoopRange {
						start: 29,
						step: 1,
						extent: Extent::Expr(39),
						body: vec![TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: 50,
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 59,
					value: 55,
				},
			],
		)
	}

	/// `avg_pool_2d_backward` — the donor's input adjoint: each lane owns
	/// one input element and scans every output window, accumulating
	/// `output_gradient / count` where membership is the u32-shifted
	/// `input + padding ∈ [output * stride, output * stride + kernel)`
	/// pair and `count` is recomputed per window by the inner signed-tap
	/// loop. `[output_gradient, input_gradient]` + nine u32 scalars ABI.
	pub(in crate::runtime) fn avg_pool_2d_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // input_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // input_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::Push(6), // kernel_size
			/* 18 */ E::Push(7), // stride
			/* 19 */ E::Push(8), // padding
			/* 20 */ E::MulU32(11, 12), // batch * channels
			/* 21 */ E::GeU32(2, 20),
			/* 22 */ E::GeU32(7, 13),
			/* 23 */ E::GeU32(10, 14),
			/* 24 */ E::OrBool(21, 22),
			/* 25 */ E::OrBool(24, 23), // out-of-range guard
			/* 26 */ E::DivU32(2, 12), // batch
			/* 27 */ E::ModU32(2, 12), // channel
			/* 28 */ E::AddU32(7, 19), // input_y + padding
			/* 29 */ E::AddU32(10, 19), // input_x + padding
			/* 30 */ E::LoopVar(2), // output_y
			/* 31 */ E::MulU32(30, 18), // output_y * stride
			/* 32 */ E::GeU32(77, 31), // input_y >= window start
			/* 33 */ E::AddU32(31, 17), // + kernel_size
			/* 34 */ E::LtU32(77, 33), // input_y < window end
			/* 35 */ E::AndBool(32, 34), // input_y member
			/* 36 */ E::LoopVar(4), // output_x
			/* 37 */ E::MulU32(36, 18), // output_x * stride
			/* 38 */ E::GeU32(78, 37), // input_x >= window start
			/* 39 */ E::AddU32(37, 17), // + kernel_size
			/* 40 */ E::LtU32(78, 39), // input_x < window end
			/* 41 */ E::AndBool(38, 40), // input_x member
			/* 42 */ E::AndBool(35, 41), // in window
			/* 43 */ E::LoopVar(6), // kernel_y
			/* 44 */ E::AddU32(31, 43), // tap y pre-shift base
			/* 45 */ E::GeU32(44, 19), // base >= padding
			/* 46 */ E::SubU32(44, 19), // tap y
			/* 47 */ E::LtU32(46, 13), // tap y < input_height
			/* 48 */ E::AndBool(45, 47), // tap y valid
			/* 49 */ E::LoopVar(7), // kernel_x
			/* 50 */ E::AddU32(37, 49), // tap x pre-shift base
			/* 51 */ E::GeU32(50, 19), // base >= padding
			/* 52 */ E::SubU32(50, 19), // tap x
			/* 53 */ E::LtU32(52, 14), // tap x < input_width
			/* 54 */ E::AndBool(51, 53), // tap x valid
			/* 55 */ E::AndBool(48, 54), // valid tap
			/* 56 */ E::RegisterU32 { tile: 1, index: 0 }, // count
			/* 57 */ E::ConstU32(1),
			/* 58 */ E::AddU32(56, 57), // count + 1
			/* 59 */ E::MulU32(26, 12), // batch * channels
			/* 60 */ E::AddU32(59, 27), // + channel → batch_channel
			/* 61 */ E::MulU32(60, 15), // * output_height
			/* 62 */ E::AddU32(61, 30), // + output_y
			/* 63 */ E::MulU32(62, 16), // * output_width
			/* 64 */ E::AddU32(81, 36), // output_offset = out_base + oy*ow + ox
			/* 65 */ E::LoadF32 {
				binding: 0,
				address: 64,
			},
			/* 66 */ E::F32FromU32(56), // float(count)
			/* 67 */ E::DivF32(65, 66), // output_gradient / count
			/* 68 */ E::RegisterF32 { tile: 0, index: 0 }, // gradient_sum
			/* 69 */ E::AddF32(68, 67), // gradient_sum + term
			/* 70 */ E::ConstF32(0), // 0.0f
			/* 71 */ E::MulU32(60, 13), // bc * input_height
			/* 72 */ E::AddU32(71, 7), // + input_y
			/* 73 */ E::MulU32(72, 14), // * input_width
			/* 74 */ E::AddU32(73, 10), // input_offset
			/* 75 */ E::ConstU32(0),
			/* 76 */ E::MulU32(30, 16), // output_y * output_width
			/* 77 */ E::RegisterU32 { tile: 1, index: 1 }, // input_y + padding
			/* 78 */ E::RegisterU32 { tile: 1, index: 2 }, // input_x + padding
			/* 79 */ E::RegisterU32 { tile: 1, index: 3 }, // input_offset
			/* 80 */ E::RegisterU32 { tile: 1, index: 4 }, // bc * oh * ow
			/* 81 */ E::AddU32(80, 76), // out_base + oy * ow
			/* 82 */ E::MulU32(61, 16), // bc * output_height * output_width
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 5 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 25 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 70,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 29,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 74,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 4,
					value: 82,
				},
				TileOp::LaneLoopRange {
					start: 75,
					step: 1,
					extent: Extent::Push(4),
					body: vec![TileOp::LaneIf {
						cond: 35,
						body: vec![TileOp::LaneLoopRange {
							start: 75,
							step: 1,
							extent: Extent::Push(5),
							body: vec![TileOp::LaneIf {
								cond: 41,
								body: vec![
									TileOp::AssignScalarU32 {
										tile: 1,
										index: 0,
										value: 75,
									},
									TileOp::LaneLoopRange {
										start: 75,
										step: 1,
										extent: Extent::Push(6),
										body: vec![TileOp::LaneLoopRange {
											start: 75,
											step: 1,
											extent: Extent::Push(6),
											body: vec![TileOp::LaneIf {
												cond: 55,
												body: vec![TileOp::AssignScalarU32 {
													tile: 1,
													index: 0,
													value: 58,
												}],
											}],
										}],
									},
									TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 69,
									},
								],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 79,
					value: 68,
				},
			],
		)
	}

	/// `max_pool_2d_backward` — the donor's input adjoint: each lane owns
	/// one input element, scans every output window, and adds the
	/// window's output gradient when the saved indices element equals
	/// this input's absolute offset. `relu` selects the fused
	/// `max_pool_2d_relu_backward` store: `saved_output > 0 ? sum : 0`.
	/// `[indices, output_gradient, input_gradient]` (+ `saved_output`
	/// for `relu`) + nine u32 scalars ABI.
	pub(in crate::runtime) fn max_pool_2d_backward() -> Self {
		Self::max_pool_2d_backward_program(false)
	}

	/// `max_pool_2d_relu_backward` — the submit-time fusion donor:
	/// identical gather to `max_pool_2d_backward`, but the store applies
	/// the ReLU mask against the saved forward output so the unmasked
	/// intermediate is never materialized.
	pub(in crate::runtime) fn max_pool_2d_relu_backward() -> Self {
		Self::max_pool_2d_backward_program(true)
	}

	fn max_pool_2d_backward_program(relu: bool) -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // input_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // input_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::Push(6), // kernel_size
			/* 18 */ E::Push(7), // stride
			/* 19 */ E::Push(8), // padding
			/* 20 */ E::MulU32(11, 12), // batch * channels
			/* 21 */ E::GeU32(2, 20),
			/* 22 */ E::GeU32(7, 13),
			/* 23 */ E::GeU32(10, 14),
			/* 24 */ E::OrBool(21, 22),
			/* 25 */ E::OrBool(24, 23), // out-of-range guard
			/* 26 */ E::DivU32(2, 12), // batch
			/* 27 */ E::ModU32(2, 12), // channel
			/* 28 */ E::AddU32(7, 19), // input_y + padding
			/* 29 */ E::AddU32(10, 19), // input_x + padding
			/* 30 */ E::LoopVar(2), // output_y
			/* 31 */ E::MulU32(30, 18), // output_y * stride
			/* 32 */ E::GeU32(64, 31), // input_y >= window start
			/* 33 */ E::AddU32(31, 17), // + kernel_size
			/* 34 */ E::LtU32(64, 33), // input_y < window end
			/* 35 */ E::AndBool(32, 34), // input_y member
			/* 36 */ E::LoopVar(4), // output_x
			/* 37 */ E::MulU32(36, 18), // output_x * stride
			/* 38 */ E::GeU32(65, 37), // input_x >= window start
			/* 39 */ E::AddU32(37, 17), // + kernel_size
			/* 40 */ E::LtU32(65, 39), // input_x < window end
			/* 41 */ E::AndBool(38, 40), // input_x member
			/* 42 */ E::AndBool(35, 41), // in window
			/* 43 */ E::MulU32(26, 12), // batch * channels
			/* 44 */ E::AddU32(43, 27), // + channel → batch_channel
			/* 45 */ E::MulU32(44, 15), // * output_height
			/* 46 */ E::AddU32(45, 30), // + output_y
			/* 47 */ E::MulU32(46, 16), // * output_width
			/* 48 */ E::AddU32(68, 36), // output_offset = out_base + oy*ow + ox
			/* 49 */
			E::BufferU32 {
				binding: 0,
				index: 48,
			}, // indices[output_offset]
			/* 50 */ E::MulU32(44, 13), // bc * input_height
			/* 51 */ E::AddU32(50, 7), // + input_y
			/* 52 */ E::MulU32(51, 14), // * input_width
			/* 53 */ E::AddU32(52, 10), // input_offset
			/* 54 */ E::EqU32(49, 66), // indices[out] == input_offset
			/* 55 */ E::LoadF32 {
				binding: 1,
				address: 48,
			},
			/* 56 */ E::RegisterF32 { tile: 0, index: 0 }, // gradient_sum
			/* 57 */ E::AddF32(56, 55), // gradient_sum + term
			/* 58 */ E::ConstF32(0), // 0.0f
			/* 59 */ E::ConstU32(0),
			/* 60 */
			E::LoadF32 {
				binding: 2,
				address: 66,
			}, // saved_output[input_offset]
			/* 61 */ E::GtF32(60, 58), // saved_output > 0
			/* 62 */
			E::SelectF32 {
				cond: 61,
				if_true: 56,
				if_false: 58,
			}, // relu-masked gradient
			/* 63 */ E::MulU32(30, 16), // output_y * output_width
			/* 64 */ E::RegisterU32 { tile: 1, index: 0 }, // input_y + padding
			/* 65 */ E::RegisterU32 { tile: 1, index: 1 }, // input_x + padding
			/* 66 */ E::RegisterU32 { tile: 1, index: 2 }, // input_offset
			/* 67 */ E::RegisterU32 { tile: 1, index: 3 }, // bc * oh * ow
			/* 68 */ E::AddU32(67, 63), // out_base + oy * ow
			/* 69 */ E::MulU32(45, 16), // bc * output_height * output_width
		];
		let bindings = if relu {
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			]
		} else {
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)]
		};
		let store = if relu { 3 } else { 2 };
		let value = if relu { 62 } else { 56 };
		Self::flat_program(
			bindings,
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"kernel_size".into(),
				"stride".into(),
				"padding".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 25 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 58,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 28,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 29,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 53,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 69,
				},
				TileOp::LaneLoopRange {
					start: 59,
					step: 1,
					extent: Extent::Push(4),
					body: vec![TileOp::LaneIf {
						cond: 35,
						body: vec![TileOp::LaneLoopRange {
							start: 59,
							step: 1,
							extent: Extent::Push(5),
							body: vec![TileOp::LaneIf {
								cond: 41,
								body: vec![TileOp::LaneIf {
									cond: 54,
									body: vec![TileOp::AssignScalar {
										tile: 0,
										index: 0,
										value: 57,
									}],
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: store,
					address: 66,
					value,
				},
			],
		)
	}

	/// `adaptive_avg_pool_2d_backward` — the donor's input adjoint: each
	/// lane owns one input element and scans every output window, adding
	/// `output_gradient / count` while the input sits inside the
	/// window's unsigned `start/end` range; `count` is the window area
	/// recomputed per output. `[output_gradient, input_gradient]` + six
	/// u32 scalars ABI.
	pub(in crate::runtime) fn adaptive_avg_pool_2d_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::GroupY,
			/* 2 */ E::GroupZ, // batch_channel
			/* 3 */ E::Lane,
			/* 4 */ E::ConstU32(16),
			/* 5 */ E::ModU32(3, 4), // group_thread_id.x
			/* 6 */ E::MulU32(0, 4),
			/* 7 */ E::AddU32(6, 5), // input_y
			/* 8 */ E::DivU32(3, 4), // group_thread_id.y
			/* 9 */ E::MulU32(1, 4),
			/* 10 */ E::AddU32(9, 8), // input_x
			/* 11 */ E::Push(0), // batch_size
			/* 12 */ E::Push(1), // channels
			/* 13 */ E::Push(2), // input_height
			/* 14 */ E::Push(3), // input_width
			/* 15 */ E::Push(4), // output_height
			/* 16 */ E::Push(5), // output_width
			/* 17 */ E::MulU32(11, 12), // batch * channels
			/* 18 */ E::GeU32(2, 17),
			/* 19 */ E::GeU32(7, 13),
			/* 20 */ E::GeU32(10, 14),
			/* 21 */ E::OrBool(18, 19),
			/* 22 */ E::OrBool(21, 20), // out-of-range guard
			/* 23 */ E::DivU32(2, 12), // batch
			/* 24 */ E::ModU32(2, 12), // channel
			/* 25 */ E::LoopVar(2), // output_y
			/* 26 */ E::MulU32(25, 13), // output_y * input_height
			/* 27 */ E::DivU32(26, 15), // input_y_start
			/* 28 */ E::GeU32(70, 27), // input_y >= start
			/* 29 */ E::ConstU32(1),
			/* 30 */ E::AddU32(25, 29), // output_y + 1
			/* 31 */ E::MulU32(30, 13), // * input_height
			/* 32 */ E::AddU32(31, 15), // + output_height
			/* 33 */ E::SubU32(32, 29), // - 1
			/* 34 */ E::DivU32(33, 15), // input_y_end
			/* 35 */ E::LtU32(70, 34), // input_y < end
			/* 36 */ E::AndBool(28, 35), // input_y member
			/* 37 */ E::LoopVar(4), // output_x
			/* 38 */ E::MulU32(37, 14), // output_x * input_width
			/* 39 */ E::DivU32(38, 16), // input_x_start
			/* 40 */ E::GeU32(71, 39), // input_x >= start
			/* 41 */ E::AddU32(37, 29), // output_x + 1
			/* 42 */ E::MulU32(41, 14), // * input_width
			/* 43 */ E::AddU32(42, 16), // + output_width
			/* 44 */ E::SubU32(43, 29), // - 1
			/* 45 */ E::DivU32(44, 16), // input_x_end
			/* 46 */ E::LtU32(71, 45), // input_x < end
			/* 47 */ E::AndBool(40, 46), // input_x member
			/* 48 */ E::AndBool(36, 47), // in window
			/* 49 */ E::SubU32(34, 27), // window height
			/* 50 */ E::SubU32(45, 39), // window width
			/* 51 */ E::MulU32(49, 50), // count
			/* 52 */ E::F32FromU32(51), // float(count)
			/* 53 */ E::MulU32(23, 12), // batch * channels
			/* 54 */ E::AddU32(53, 24), // + channel → batch_channel
			/* 55 */ E::MulU32(54, 15), // * output_height
			/* 56 */ E::AddU32(55, 25), // + output_y
			/* 57 */ E::MulU32(56, 16), // * output_width
			/* 58 */ E::AddU32(74, 37), // out_base + oy*ow + output_x
			/* 59 */ E::LoadF32 {
				binding: 0,
				address: 58,
			},
			/* 60 */ E::DivF32(59, 52), // output_gradient / count
			/* 61 */ E::RegisterF32 { tile: 0, index: 0 }, // gradient_sum
			/* 62 */ E::AddF32(61, 60), // gradient_sum + term
			/* 63 */ E::ConstF32(0), // 0.0f
			/* 64 */ E::MulU32(54, 13), // bc * input_height
			/* 65 */ E::AddU32(64, 7), // + input_y
			/* 66 */ E::MulU32(65, 14), // * input_width
			/* 67 */ E::AddU32(66, 10), // input_offset
			/* 68 */ E::ConstU32(0),
			/* 69 */ E::MulU32(25, 16), // output_y * output_width
			/* 70 */ E::RegisterU32 { tile: 1, index: 0 }, // input_y
			/* 71 */ E::RegisterU32 { tile: 1, index: 1 }, // input_x
			/* 72 */ E::RegisterU32 { tile: 1, index: 2 }, // input_offset
			/* 73 */ E::RegisterU32 { tile: 1, index: 3 }, // bc * oh * ow
			/* 74 */ E::AddU32(73, 69), // out_base + oy * ow
			/* 75 */ E::MulU32(55, 16), // bc * output_height * output_width
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
			],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 22 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 63,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 0,
					value: 7,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 1,
					value: 10,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 2,
					value: 67,
				},
				TileOp::AssignScalarU32 {
					tile: 1,
					index: 3,
					value: 75,
				},
				TileOp::LaneLoopRange {
					start: 68,
					step: 1,
					extent: Extent::Push(4),
					body: vec![TileOp::LaneIf {
						cond: 36,
						body: vec![TileOp::LaneLoopRange {
							start: 68,
							step: 1,
							extent: Extent::Push(5),
							body: vec![TileOp::LaneIf {
								cond: 47,
								body: vec![TileOp::AssignScalar {
									tile: 0,
									index: 0,
									value: 62,
								}],
							}],
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 1,
					address: 72,
					value: 61,
				},
			],
		)
	}
	/// `split_heads` — the donor's flat NHWC→NHSC repack: one lane per
	/// output element decomposing `(b, h, s, d)` and gathering
	/// `input[((b*s + s)*h + h)*d + f]`. Donor: `attention/split_heads.slang`
	/// (`MlSplitHeadsF32`; `[input, output]` + four u32 scalars).
	pub(in crate::runtime) fn split_heads() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // output_index
			/* 5 */ E::Push(0), // batch
			/* 6 */ E::Push(1), // sequence_length
			/* 7 */ E::Push(2), // num_heads
			/* 8 */ E::Push(3), // head_dim
			/* 9 */ E::MulU32(5, 6),
			/* 10 */ E::MulU32(9, 7),
			/* 11 */ E::MulU32(10, 8), // count
			/* 12 */ E::GeU32(4, 11), // out-of-range guard
			/* 13 */ E::ModU32(4, 8), // feature
			/* 14 */ E::DivU32(4, 8), // quotient
			/* 15 */ E::ModU32(14, 6), // sequence
			/* 16 */ E::DivU32(14, 6), // quotient
			/* 17 */ E::ModU32(16, 7), // head
			/* 18 */ E::DivU32(16, 7), // batch
			/* 19 */ E::MulU32(18, 6), // batch*sequence
			/* 20 */ E::AddU32(19, 15), // +sequence
			/* 21 */ E::MulU32(20, 7), // *heads
			/* 22 */ E::AddU32(21, 17), // +head
			/* 23 */ E::MulU32(22, 8), // *dim
			/* 24 */ E::AddU32(23, 13), // input_index
			/* 25 */ E::LoadF32 {
				binding: 0,
				address: 24,
			},
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch".into(),
				"sequence_length".into(),
				"num_heads".into(),
				"head_dim".into(),
			],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 12 },
				TileOp::StoreScalar {
					binding: 1,
					address: 4,
					value: 25,
				},
			],
		)
	}

	/// `merge_heads` — the mirror repack NHSC→NHWC: the same flat
	/// decomposition with head/sequence roles swapped so it gathers
	/// `input[((b*h + h)*s + s)*d + f]`. Donor: `attention/merge_heads.slang`
	/// (`MlMergeHeadsF32`; `[input, output]` + four u32 scalars).
	pub(in crate::runtime) fn merge_heads() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // output_index
			/* 5 */ E::Push(0), // batch
			/* 6 */ E::Push(1), // sequence_length
			/* 7 */ E::Push(2), // num_heads
			/* 8 */ E::Push(3), // head_dim
			/* 9 */ E::MulU32(5, 6),
			/* 10 */ E::MulU32(9, 7),
			/* 11 */ E::MulU32(10, 8), // count
			/* 12 */ E::GeU32(4, 11), // out-of-range guard
			/* 13 */ E::ModU32(4, 8), // feature
			/* 14 */ E::DivU32(4, 8), // quotient
			/* 15 */ E::ModU32(14, 7), // head
			/* 16 */ E::DivU32(14, 7), // quotient
			/* 17 */ E::ModU32(16, 6), // sequence
			/* 18 */ E::DivU32(16, 6), // batch
			/* 19 */ E::MulU32(18, 7), // batch*heads
			/* 20 */ E::AddU32(19, 15), // +head
			/* 21 */ E::MulU32(20, 6), // *sequence
			/* 22 */ E::AddU32(21, 17), // +sequence
			/* 23 */ E::MulU32(22, 8), // *dim
			/* 24 */ E::AddU32(23, 13), // input_index
			/* 25 */ E::LoadF32 {
				binding: 0,
				address: 24,
			},
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch".into(),
				"sequence_length".into(),
				"num_heads".into(),
				"head_dim".into(),
			],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 12 },
				TileOp::StoreScalar {
					binding: 1,
					address: 4,
					value: 25,
				},
			],
		)
	}

	/// `sdpa_softmax` — the donor's workgroup-per-row softmax over
	/// attention scores: lane-strided `score * scale (+ mask)`
	/// transformed taps with the causal `column > query` blank, a
	/// shared tree maximum, a shared tree exponent sum and the
	/// normalize-store pass. Donor: `attention/sdpa_softmax.slang`
	/// (`MlSdpaSoftmaxF32` at 256 lanes, `MlSdpaSoftmaxN32F32` at 32).
	/// `[scores, mask, probability]` + `[rows, columns, has_mask,
	/// causal]` u32 scalars + `scale` f32 — the specialization repacks
	/// the donor's interleaved `[rows, columns, scale, has_mask,
	/// causal]` record order.
	pub(in crate::runtime) fn sdpa_softmax(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"sdpa_softmax lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::Push(2), // has_mask
			/* 3 */ E::Push(3), // causal
			/* 4 */ E::PushF32(0), // scale
			/* 5 */ E::GroupX, // row
			/* 6 */ E::MulU32(5, 1), // base = row*columns
			/* 7 */ E::ModU32(5, 1), // query = row%columns
			/* 8 */ E::Induction, // column
			/* 9 */ E::AddU32(33, 8), // address
			/* 10 */ E::LoadF32 {
				binding: 0,
				address: 9,
			}, // score
			/* 11 */ E::MulF32(10, 4), // score*scale
			/* 12 */ E::LoadF32 {
				binding: 1,
				address: 9,
			}, // mask
			/* 13 */ E::AddF32(11, 12), // scaled + mask
			/* 14 */ E::ConstU32(0),
			/* 15 */ E::Lane,
			/* 16 */ E::NeqU32(2, 14), // has_mask != 0
			/* 17 */ E::AddF32(11, 35), // masked score
			/* 18 */ E::NeqU32(3, 14), // causal != 0
			/* 19 */ E::LtU32(34, 8), // query < column
			/* 20 */ E::AndBool(18, 19), // causal blank applies
			/* 21 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 22 */
			E::SelectF32 {
				cond: 20,
				if_true: 21,
				if_false: 17,
			}, // transformed
			/* 23 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 24 */ E::RegisterF32 { tile: 2, index: 0 }, // row_max
			/* 25 */ E::SubF32(22, 24), // transformed - row_max
			/* 26 */ E::ExpF32(25),
			/* 27 */ E::SharedF32 {
				tile: 0,
				offset: 14,
			}, // shared[0]
			/* 28 */ E::RegisterF32 { tile: 3, index: 0 }, // inverse_sum
			/* 29 */ E::MulF32(26, 28), // probability
			/* 30 */ E::ConstF32(0x0000_0000), // 0.0
			/* 31 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 32 */ E::DivF32(31, 27), // 1.0 / sum
			/* 33 */ E::RegisterU32 { tile: 4, index: 0 }, // base
			/* 34 */ E::RegisterU32 { tile: 4, index: 1 }, // query
			/* 35 */
			E::SelectF32 {
				cond: 16,
				if_true: 12,
				if_false: 30,
			}, // mask or 0
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec![
				"rows".into(),
				"columns".into(),
				"has_mask".into(),
				"causal".into(),
			],
			push_f32_scalars: vec!["scale".into()],
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 2 },
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::AssignScalarU32 {
					tile: 4,
					index: 0,
					value: 6,
				},
				TileOp::AssignScalarU32 {
					tile: 4,
					index: 1,
					value: 7,
				},
				TileOp::LaneFold {
					acc: 1,
					init: 21,
					extent: Extent::Push(1),
					op: LaneReduce::Max,
					element: 22,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 23,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Max,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 27,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 1,
					init: 30,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 26,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 23,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 32,
				},
				TileOp::Barrier,
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 9,
						value: 29,
					}],
				},
			],
		})
	}

	/// `small_causal_sdpa` — the seven-node packed causal-attention
	/// composite: stage Q/K/V from packed `[B*S, H*D]` storage into shared
	/// `[S, D]` tiles while republishing the split-head outputs, then per
	/// batch-head compute scaled causal scores, a serial row softmax and
	/// the weighted value contraction, republishing scores, probabilities,
	/// context and the merged output. Donor:
	/// `attention/small_causal_sdpa.slang` (`MlSmallCausalSdpaF32`; ten
	/// storage bindings + `[batch, sequence, heads, dim]` u32 scalars and
	/// the `scale` f32).
	///
	/// The donor's tile loop `first += queries_per_tile` and context stride
	/// `d += row_lanes` carry runtime steps the `LaneLoopRange` literal
	/// step cannot spell; both are rephrased as uniform serial loops over
	/// `ceil(extent / step)` tiles with the lane-invariant bases hoisted
	/// into u32 registers (the same generated-code discipline as the
	/// convolution/pool ports).
	pub(in crate::runtime) fn small_causal_sdpa() -> Self {
		use LaneExpr as E;
		// u32 register tile 10: [batch_idx, head, count, row_lanes, qpt,
		// slot, lane_row, offset, stage_base, bh*seq, query, query*dim,
		// lane_row*dim, row_index, ctx_base, packed_base, d].
		let lane_exprs = vec![
			/* 0 */ E::GroupX, // bh
			/* 1 */ E::Push(0), // batch
			/* 2 */ E::Push(1), // sequence_length
			/* 3 */ E::Push(2), // num_heads
			/* 4 */ E::Push(3), // head_dim
			/* 5 */ E::PushF32(0), // scale
			/* 6 */ E::MulU32(1, 3), // batch*heads
			/* 7 */ E::DivU32(0, 3), // batch_idx
			/* 8 */ E::ModU32(0, 3), // head
			/* 9 */ E::MulU32(2, 4), // count = seq*dim
			/* 10 */ E::ConstU32(128),
			/* 11 */ E::MinU32(2, 10), // row_lanes
			/* 12 */ E::ConstU32(256),
			/* 13 */ E::DivU32(12, 11), // queries_per_tile
			/* 14 */ E::Lane,
			/* 15 */ E::DivU32(14, 11), // slot
			/* 16 */ E::ModU32(14, 11), // lane_row
			/* 17 */ E::MulU32(15, 11), // offset
			/* 18 */ E::MulU32(0, 9), // stage_base = bh*count
			/* 19 */ E::MulU32(0, 2), // bh*seq
			/* 20 */ E::Induction,
			/* 21 */ E::ModU32(20, 4), // d
			/* 22 */ E::DivU32(20, 4), // s
			/* 23 */ E::MulU32(111, 2), // batch_idx*seq
			/* 24 */ E::AddU32(23, 22),
			/* 25 */ E::MulU32(24, 3),
			/* 26 */ E::AddU32(25, 112),
			/* 27 */ E::MulU32(26, 4),
			/* 28 */ E::AddU32(27, 21), // packed
			/* 29 */ E::LoadF32 {
				binding: 0,
				address: 28,
			},
			/* 30 */ E::LoadF32 {
				binding: 1,
				address: 28,
			},
			/* 31 */ E::LoadF32 {
				binding: 2,
				address: 28,
			},
			/* 32 */ E::AddU32(18, 20), // stage address
			/* 33 */ E::ConstU32(0),
			/* 34 */ E::ConstU32(1),
			/* 35 */ E::ConstF32(0x0000_0000),
			/* 36 */ E::ConstF32(0x3f80_0000),
			/* 37 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 38 */ E::MulU32(20, 114), // first
			/* 39 */ E::AddU32(38, 115), // query
			/* 40 */ E::LtU32(39, 2), // query < seq
			/* 41 */ E::LtU32(58, 2), // lane_row < seq
			/* 42 */ E::AndBool(40, 41), // active
			/* 43 */ E::MulU32(39, 4), // query*dim
			/* 44 */ E::MulU32(58, 4), // lane_row*dim
			/* 45 */ E::AddU32(19, 39), // bh*seq + query
			/* 46 */ E::MulU32(45, 2), // row_index
			/* 47 */ E::MulU32(45, 4), // ctx_base
			/* 48 */ E::AddU32(23, 39),
			/* 49 */ E::MulU32(48, 3),
			/* 50 */ E::AddU32(49, 8),
			/* 51 */ E::MulU32(50, 4), // packed_base
			/* 52 */ E::AddU32(2, 114),
			/* 53 */ E::SubU32(52, 34),
			/* 54 */ E::DivU32(53, 114), // outer tiles = ceil(seq/qpt)
			/* 55 */ E::AddU32(4, 113),
			/* 56 */ E::SubU32(55, 34),
			/* 57 */ E::DivU32(56, 113), // d tiles = ceil(dim/row_lanes)
			/* 58 */ E::RegisterU32 { tile: 10, index: 6 }, // lane_row
			/* 59 */ E::RegisterU32 { tile: 10, index: 7 }, // offset
			/* 60 */ E::RegisterU32 {
				tile: 10,
				index: 10,
			}, // query
			/* 61 */ E::RegisterU32 {
				tile: 10,
				index: 11,
			}, // query*dim
			/* 62 */ E::RegisterU32 {
				tile: 10,
				index: 12,
			}, // lane_row*dim
			/* 63 */ E::RegisterU32 {
				tile: 10,
				index: 13,
			}, // row_index
			/* 64 */ E::RegisterU32 {
				tile: 10,
				index: 14,
			}, // ctx_base
			/* 65 */ E::RegisterU32 {
				tile: 10,
				index: 15,
			}, // packed_base
			/* 66 */ E::RegisterU32 {
				tile: 10,
				index: 16,
			}, // d
			/* 67 */ E::AddU32(59, 58), // weights index
			/* 68 */ E::AddU32(59, 20), // offset + key
			/* 69 */ E::SharedF32 {
				tile: 3,
				offset: 68,
			},
			/* 70 */ E::AddU32(61, 20), // qd + key
			/* 71 */ E::SharedF32 {
				tile: 0,
				offset: 70,
			},
			/* 72 */ E::AddU32(62, 20), // ld + key
			/* 73 */ E::SharedF32 {
				tile: 1,
				offset: 72,
			},
			/* 74 */ E::MulF32(71, 73), // dot term
			/* 75 */ E::AddU32(63, 58), // score address
			/* 76 */ E::RegisterF32 { tile: 4, index: 0 }, // dot
			/* 77 */ E::MulF32(76, 5), // dot*scale
			/* 78 */ E::LtU32(60, 58), // query < lane_row
			/* 79 */
			E::SelectF32 {
				cond: 78,
				if_true: 37,
				if_false: 77,
			}, // masked score
			/* 80 */ E::SharedF32 {
				tile: 3,
				offset: 59,
			}, // weights[offset]
			/* 81 */ E::RegisterF32 { tile: 6, index: 0 }, // maximum
			/* 82 */ E::MaxF32(81, 69),
			/* 83 */ E::SubF32(69, 81),
			/* 84 */ E::ExpF32(83),
			/* 85 */ E::RegisterF32 { tile: 7, index: 0 }, // sum
			/* 86 */ E::AddF32(85, 84),
			/* 87 */ E::DivF32(36, 85), // inverse sum
			/* 88 */ E::RegisterF32 { tile: 8, index: 0 }, // inverse
			/* 89 */ E::MulF32(69, 88), // probability
			/* 90 */ E::AddU32(63, 20), // row_index + key
			/* 91 */ E::MulU32(20, 113), // i*row_lanes
			/* 92 */ E::AddU32(58, 91), // d
			/* 93 */ E::MulU32(20, 4), // key*dim
			/* 94 */ E::AddU32(93, 66), // key*dim + d
			/* 95 */ E::SharedF32 {
				tile: 2,
				offset: 94,
			},
			/* 96 */ E::MulF32(69, 95), // context term
			/* 97 */ E::RegisterF32 { tile: 9, index: 0 }, // result
			/* 98 */ E::AddU32(64, 66), // context address
			/* 99 */ E::AddU32(65, 66), // packed output address
			/* 100 */ E::RegisterF32 { tile: 5, index: 0 }, // score
			/* 101 */ E::EqU32(58, 33), // lane_row == 0
			/* 102 */ E::AndBool(101, 40), // softmax owner
			/* 103 */ E::LtU32(66, 4), // d < head_dim
			/* 104 */ E::AddF32(85, 100), // sum + e
			/* 105 */ E::RegisterU32 {
				tile: 10,
				index: 17,
			}, // packed
			/* 106 */ E::RegisterU32 {
				tile: 10,
				index: 18,
			}, // stage address
			/* 107 */ E::RegisterF32 { tile: 11, index: 0 }, // staged value
			/* 108 */ E::LoadF32 {
				binding: 0,
				address: 105,
			},
			/* 109 */ E::LoadF32 {
				binding: 1,
				address: 105,
			},
			/* 110 */ E::LoadF32 {
				binding: 2,
				address: 105,
			},
			/* 111 */ E::RegisterU32 { tile: 10, index: 0 }, // batch_idx
			/* 112 */ E::RegisterU32 { tile: 10, index: 1 }, // head
			/* 113 */ E::RegisterU32 { tile: 10, index: 3 }, // row_lanes
			/* 114 */ E::RegisterU32 { tile: 10, index: 4 }, // qpt
			/* 115 */ E::RegisterU32 { tile: 10, index: 5 }, // slot
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
				Binding::Output(3),
				Binding::Output(4),
				Binding::Output(5),
				Binding::Output(6),
			],
			push_scalars: vec![
				"batch".into(),
				"sequence_length".into(),
				"num_heads".into(),
				"head_dim".into(),
			],
			push_f32_scalars: vec!["scale".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 256,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 2048,
					stride: 2048,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 2048,
					stride: 2048,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 2048,
					stride: 2048,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 1024,
					stride: 1024,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 19 },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 6 },
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 0,
					value: 7,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 1,
					value: 8,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 2,
					value: 9,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 3,
					value: 11,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 4,
					value: 13,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 5,
					value: 15,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 6,
					value: 16,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 7,
					value: 17,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 8,
					value: 18,
				},
				TileOp::AssignScalarU32 {
					tile: 10,
					index: 9,
					value: 19,
				},
				TileOp::LaneLoop {
					extent: Extent::Expr(9),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 17,
							value: 28,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 18,
							value: 32,
						},
						TileOp::AssignScalar {
							tile: 11,
							index: 0,
							value: 108,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 20,
							value: 107,
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 106,
							value: 107,
						},
						TileOp::AssignScalar {
							tile: 11,
							index: 0,
							value: 109,
						},
						TileOp::StoreShared {
							tile: 1,
							offset: 20,
							value: 107,
						},
						TileOp::StoreScalar {
							binding: 4,
							address: 106,
							value: 107,
						},
						TileOp::AssignScalar {
							tile: 11,
							index: 0,
							value: 110,
						},
						TileOp::StoreShared {
							tile: 2,
							offset: 20,
							value: 107,
						},
						TileOp::StoreScalar {
							binding: 5,
							address: 106,
							value: 107,
						},
					],
				},
				TileOp::Barrier,
				TileOp::LaneLoopRange {
					start: 33,
					step: 1,
					extent: Extent::Expr(54),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 10,
							value: 39,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 11,
							value: 43,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 12,
							value: 44,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 13,
							value: 46,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 14,
							value: 47,
						},
						TileOp::AssignScalarU32 {
							tile: 10,
							index: 15,
							value: 51,
						},
						TileOp::AssignScalar {
							tile: 5,
							index: 0,
							value: 37,
						},
						TileOp::LaneIf {
							cond: 42,
							body: vec![
								TileOp::LaneFold {
									acc: 4,
									init: 35,
									extent: Extent::Push(3),
									op: LaneReduce::Sum,
									element: 74,
									serial: true,
								},
								TileOp::StoreScalar {
									binding: 6,
									address: 75,
									value: 76,
								},
								TileOp::AssignScalar {
									tile: 5,
									index: 0,
									value: 79,
								},
							],
						},
						TileOp::StoreShared {
							tile: 3,
							offset: 67,
							value: 100,
						},
						TileOp::Barrier,
						TileOp::LaneIf {
							cond: 102,
							body: vec![
								TileOp::AssignScalar {
									tile: 6,
									index: 0,
									value: 80,
								},
								TileOp::LaneLoopRange {
									start: 34,
									step: 1,
									extent: Extent::Push(1),
									body: vec![TileOp::AssignScalar {
										tile: 6,
										index: 0,
										value: 82,
									}],
								},
								TileOp::AssignScalar {
									tile: 7,
									index: 0,
									value: 35,
								},
								TileOp::LaneLoopRange {
									start: 33,
									step: 1,
									extent: Extent::Push(1),
									body: vec![
										TileOp::AssignScalar {
											tile: 5,
											index: 0,
											value: 84,
										},
										TileOp::StoreShared {
											tile: 3,
											offset: 68,
											value: 100,
										},
										TileOp::AssignScalar {
											tile: 7,
											index: 0,
											value: 104,
										},
									],
								},
								TileOp::AssignScalar {
									tile: 8,
									index: 0,
									value: 87,
								},
								TileOp::LaneLoopRange {
									start: 33,
									step: 1,
									extent: Extent::Push(1),
									body: vec![
										TileOp::AssignScalar {
											tile: 5,
											index: 0,
											value: 89,
										},
										TileOp::StoreShared {
											tile: 3,
											offset: 68,
											value: 100,
										},
										TileOp::StoreScalar {
											binding: 7,
											address: 90,
											value: 100,
										},
									],
								},
							],
						},
						TileOp::Barrier,
						TileOp::LaneIf {
							cond: 40,
							body: vec![TileOp::LaneLoopRange {
								start: 33,
								step: 1,
								extent: Extent::Expr(57),
								body: vec![
									TileOp::AssignScalarU32 {
										tile: 10,
										index: 16,
										value: 92,
									},
									TileOp::LaneIf {
										cond: 103,
										body: vec![
											TileOp::LaneFold {
												acc: 9,
												init: 35,
												extent: Extent::Push(1),
												op: LaneReduce::Sum,
												element: 96,
												serial: true,
											},
											TileOp::StoreScalar {
												binding: 8,
												address: 98,
												value: 97,
											},
											TileOp::StoreScalar {
												binding: 9,
												address: 99,
												value: 97,
											},
										],
									},
								],
							}],
						},
						TileOp::Barrier,
					],
				},
			],
		}
	}

	/// Flash-attention forward donor: one workgroup per `(batch_head,
	/// query)` row at 64 lanes — strided causal score pass tracked into a
	/// cooperative max, shared-weight exp/normalize pass, cooperative sum,
	/// `log_sum_exp` store and a strided per-dimension context fold.
	pub(in crate::runtime) fn flash_causal() -> Self {
		use LaneExpr as E;
		// u32 register tile 6: [key_or_dimension].
		let lane_exprs = vec![
			/* 0 */ E::GroupX, // row
			/* 1 */ E::Push(0), // batch_heads
			/* 2 */ E::Push(1), // sequence_length
			/* 3 */ E::Push(2), // head_dim
			/* 4 */ E::PushF32(0), // scale
			/* 5 */ E::MulU32(1, 2), // row_count
			/* 6 */ E::DivU32(0, 2), // batch_head
			/* 7 */ E::MulU32(6, 2), // bh*seq
			/* 8 */ E::SubU32(0, 7), // query
			/* 9 */ E::MulU32(0, 3), // query_base
			/* 10 */ E::MulU32(7, 3), // kv_head_base
			/* 11 */ E::Lane,
			/* 12 */ E::ConstU32(0),
			/* 13 */ E::ConstU32(1),
			/* 14 */ E::AddU32(8, 13), // query + 1
			/* 15 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 16 */ E::ConstF32(0x0000_0000), // 0.0
			/* 17 */ E::Induction,
			/* 18 */ E::RegisterU32 { tile: 6, index: 0 }, // key / d
			/* 19 */ E::MulU32(18, 3), // key*dim
			/* 20 */ E::AddU32(49, 19), // key_base
			/* 21 */ E::AddU32(48, 17), // qb + d
			/* 22 */ E::AddU32(20, 17), // kvb + d
			/* 23 */ E::LoadF32 {
				binding: 0,
				address: 21,
			},
			/* 24 */ E::LoadF32 {
				binding: 1,
				address: 22,
			},
			/* 25 */ E::MulF32(23, 24), // score term
			/* 26 */ E::RegisterF32 { tile: 2, index: 0 }, // score/weight/acc
			/* 27 */ E::MulF32(26, 4), // score*scale
			/* 28 */ E::RegisterF32 { tile: 3, index: 0 }, // local_max
			/* 29 */ E::MaxF32(28, 27),
			/* 30 */ E::SharedF32 {
				tile: 0,
				offset: 17,
			}, // sw[key]
			/* 31 */ E::SharedF32 {
				tile: 1,
				offset: 12,
			}, // reduce[0]
			/* 32 */ E::RegisterF32 { tile: 5, index: 0 }, // row_max
			/* 33 */ E::SubF32(30, 32), // sw - max
			/* 34 */ E::ExpF32(33), // weight
			/* 35 */ E::RegisterF32 { tile: 4, index: 0 }, // local_sum
			/* 36 */ E::AddF32(35, 26), // sum + weight
			/* 37 */ E::EqU32(11, 12), // lane == 0
			/* 38 */ E::LogF32(31), // log(sum)
			/* 39 */ E::AddF32(32, 38), // max + log(sum)
			/* 40 */ E::MulU32(17, 3), // fold key*dim
			/* 41 */ E::AddU32(40, 18), // key*dim + d
			/* 42 */ E::AddU32(49, 41), // v address
			/* 43 */ E::LoadF32 {
				binding: 2,
				address: 42,
			},
			/* 44 */ E::SharedF32 {
				tile: 0,
				offset: 17,
			}, // sw[key]
			/* 45 */ E::DivF32(44, 31), // sw/sum
			/* 46 */ E::MulF32(45, 43), // ctx term
			/* 47 */ E::AddU32(48, 18), // qb + d
			/* 48 */ E::RegisterU32 { tile: 6, index: 1 }, // query_base
			/* 49 */ E::RegisterU32 { tile: 6, index: 2 }, // kv_head_base
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch_heads".into(),
				"sequence_length".into(),
				"head_dim".into(),
			],
			push_f32_scalars: vec!["scale".into()],
			shape: BlockShape {
				lanes: 64,
				thread_columns: 64,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 1024,
					stride: 1024,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 64,
					stride: 64,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 3 },
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 5 },
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 16,
				},
				TileOp::AssignScalarU32 {
					tile: 6,
					index: 0,
					value: 12,
				},
				TileOp::AssignScalarU32 {
					tile: 6,
					index: 1,
					value: 9,
				},
				TileOp::AssignScalarU32 {
					tile: 6,
					index: 2,
					value: 10,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 15,
				},
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Expr(14),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 6,
							index: 0,
							value: 17,
						},
						TileOp::LaneFold {
							acc: 2,
							init: 16,
							extent: Extent::Push(2),
							op: LaneReduce::Sum,
							element: 25,
							serial: true,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 18,
							value: 27,
						},
						TileOp::AssignScalar {
							tile: 3,
							index: 0,
							value: 29,
						},
					],
				},
				TileOp::StoreShared {
					tile: 1,
					offset: 11,
					value: 28,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Max,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: 31,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 16,
				},
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Expr(14),
					body: vec![
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 34,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 17,
							value: 26,
						},
						TileOp::AssignScalar {
							tile: 4,
							index: 0,
							value: 36,
						},
					],
				},
				TileOp::Barrier,
				TileOp::StoreShared {
					tile: 1,
					offset: 11,
					value: 35,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Sum,
				},
				TileOp::LaneIf {
					cond: 37,
					body: vec![TileOp::StoreScalar {
						binding: 4,
						address: 0,
						value: 39,
					}],
				},
				TileOp::Barrier,
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Push(2),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 6,
							index: 0,
							value: 17,
						},
						TileOp::LaneFold {
							acc: 2,
							init: 16,
							extent: Extent::Expr(14),
							op: LaneReduce::Sum,
							element: 46,
							serial: true,
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 47,
							value: 26,
						},
					],
				},
			],
		}
	}

	/// Flash-attention backward-Q donor: cooperative `delta` reduction,
	/// then a strided per-key score/probability-gradient recomputation into
	/// shared storage and a strided per-dimension `query_gradient` fold.
	pub(in crate::runtime) fn flash_causal_backward_q() -> Self {
		use LaneExpr as E;
		// u32 register tile 7: [key, dimension].
		let lane_exprs = vec![
			/* 0 */ E::GroupX, // row
			/* 1 */ E::Push(0), // batch_heads
			/* 2 */ E::Push(1), // sequence_length
			/* 3 */ E::Push(2), // head_dim
			/* 4 */ E::PushF32(0), // scale
			/* 5 */ E::MulU32(1, 2), // row_count
			/* 6 */ E::DivU32(0, 2), // batch_head
			/* 7 */ E::MulU32(6, 2), // bh*seq
			/* 8 */ E::SubU32(0, 7), // query
			/* 9 */ E::MulU32(0, 3), // query_base
			/* 10 */ E::MulU32(7, 3), // kv_head_base
			/* 11 */ E::Lane,
			/* 12 */ E::ConstU32(0),
			/* 13 */ E::ConstU32(1),
			/* 14 */ E::AddU32(8, 13), // query + 1
			/* 15 */ E::ConstF32(0x0000_0000), // 0.0
			/* 16 */ E::Induction,
			/* 17 */ E::AddU32(53, 16), // qb + d
			/* 18 */ E::LoadF32 {
				binding: 4,
				address: 17,
			}, // out_grad[qb+d]
			/* 19 */ E::LoadF32 {
				binding: 3,
				address: 17,
			}, // output[qb+d]
			/* 20 */ E::MulF32(18, 19), // delta term
			/* 21 */ E::RegisterU32 { tile: 7, index: 0 }, // key
			/* 22 */ E::MulU32(21, 3), // key*dim
			/* 23 */ E::AddU32(54, 22), // key_base
			/* 24 */ E::AddU32(23, 16), // kvb + d
			/* 25 */ E::LoadF32 {
				binding: 0,
				address: 17,
			}, // q[qb+d]
			/* 26 */ E::LoadF32 {
				binding: 1,
				address: 24,
			}, // k[kvb+d]
			/* 27 */ E::MulF32(25, 26), // score term
			/* 28 */ E::LoadF32 {
				binding: 2,
				address: 24,
			}, // v[kvb+d]
			/* 29 */ E::MulF32(18, 28), // pg term
			/* 30 */ E::RegisterF32 { tile: 2, index: 0 }, // score/delta/acc
			/* 31 */ E::MulF32(30, 4), // score*scale
			/* 32 */ E::RegisterF32 { tile: 4, index: 0 }, // lse
			/* 33 */ E::SubF32(31, 32),
			/* 34 */ E::ExpF32(33), // probability
			/* 35 */ E::RegisterF32 { tile: 3, index: 0 }, // pg
			/* 36 */ E::RegisterF32 { tile: 5, index: 0 }, // delta
			/* 37 */ E::SubF32(35, 36), // pg - delta
			/* 38 */ E::RegisterF32 { tile: 6, index: 0 }, // probability
			/* 39 */ E::MulF32(38, 37), // prob*(pg-delta)
			/* 40 */ E::MulF32(39, 4), // *scale
			/* 41 */ E::MulU32(16, 3), // fold key*dim
			/* 42 */ E::RegisterU32 { tile: 7, index: 1 }, // d
			/* 43 */ E::AddU32(41, 42), // key*dim + d
			/* 44 */ E::AddU32(54, 43), // kvh + key*dim + d
			/* 45 */ E::LoadF32 {
				binding: 1,
				address: 44,
			},
			/* 46 */ E::SharedF32 {
				tile: 0,
				offset: 16,
			}, // ssg[key]
			/* 47 */ E::MulF32(46, 45), // ssg*k term
			/* 48 */ E::SharedF32 {
				tile: 1,
				offset: 12,
			}, // reduce[0]
			/* 49 */ E::LoadF32 {
				binding: 5,
				address: 0,
			}, // lse[row]
			/* 50 */ E::AddU32(53, 42), // qb + d
			/* 51 */ E::AddF32(30, 27), // score + term
			/* 52 */ E::AddF32(35, 29), // pg + term
			/* 53 */ E::RegisterU32 { tile: 7, index: 2 }, // query_base
			/* 54 */ E::RegisterU32 { tile: 7, index: 3 }, // kv_head_base
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Output(0),
			],
			push_scalars: vec![
				"batch_heads".into(),
				"sequence_length".into(),
				"head_dim".into(),
			],
			push_f32_scalars: vec!["scale".into()],
			shape: BlockShape {
				lanes: 64,
				thread_columns: 64,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 1024,
					stride: 1024,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 64,
					stride: 64,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 4 },
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 5 },
				TileOp::AssignScalarU32 {
					tile: 7,
					index: 0,
					value: 12,
				},
				TileOp::AssignScalarU32 {
					tile: 7,
					index: 2,
					value: 9,
				},
				TileOp::AssignScalarU32 {
					tile: 7,
					index: 3,
					value: 10,
				},
				TileOp::LaneFold {
					acc: 2,
					init: 15,
					extent: Extent::Push(2),
					op: LaneReduce::Sum,
					element: 20,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 1,
					offset: 11,
					value: 30,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: 48,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 49,
				},
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Expr(14),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 7,
							index: 0,
							value: 16,
						},
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 15,
						},
						TileOp::AssignScalar {
							tile: 3,
							index: 0,
							value: 15,
						},
						TileOp::LaneLoopRange {
							start: 12,
							step: 1,
							extent: Extent::Push(2),
							body: vec![
								TileOp::AssignScalar {
									tile: 2,
									index: 0,
									value: 51,
								},
								TileOp::AssignScalar {
									tile: 3,
									index: 0,
									value: 52,
								},
							],
						},
						TileOp::AssignScalar {
							tile: 6,
							index: 0,
							value: 34,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 21,
							value: 40,
						},
					],
				},
				TileOp::Barrier,
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Push(2),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 7,
							index: 1,
							value: 16,
						},
						TileOp::LaneFold {
							acc: 2,
							init: 15,
							extent: Extent::Expr(14),
							op: LaneReduce::Sum,
							element: 47,
							serial: true,
						},
						TileOp::StoreScalar {
							binding: 6,
							address: 50,
							value: 30,
						},
					],
				},
			],
		}
	}

	/// Flash-attention backward-KV donor: one workgroup per
	/// `(batch_head, key)` row — strided per-query probability and
	/// score-gradient staging, then a strided per-dimension fold that
	/// accumulates `key_gradient` and `value_gradient` over `key..seq`.
	pub(in crate::runtime) fn flash_causal_backward_kv() -> Self {
		use LaneExpr as E;
		// u32 register tile 8: [query, dimension].
		let lane_exprs = vec![
			/* 0 */ E::GroupX, // row (key row)
			/* 1 */ E::Push(0), // batch_heads
			/* 2 */ E::Push(1), // sequence_length
			/* 3 */ E::Push(2), // head_dim
			/* 4 */ E::PushF32(0), // scale
			/* 5 */ E::MulU32(1, 2), // row_count
			/* 6 */ E::DivU32(0, 2), // batch_head
			/* 7 */ E::MulU32(6, 2), // bh*seq
			/* 8 */ E::SubU32(0, 7), // key
			/* 9 */ E::MulU32(0, 3), // kv_base = row*dim
			/* 10 */ E::MulU32(7, 3), // head_base
			/* 11 */ E::Lane,
			/* 12 */ E::AddU32(8, 11), // q start = key+lane
			/* 13 */ E::ConstU32(0),
			/* 14 */ E::ConstF32(0x0000_0000), // 0.0
			/* 15 */ E::Induction,
			/* 16 */ E::RegisterU32 { tile: 8, index: 0 }, // q
			/* 17 */ E::MulU32(16, 3), // q*dim
			/* 18 */ E::AddU32(59, 17), // qb
			/* 19 */ E::AddU32(18, 15), // qb + d
			/* 20 */ E::LoadF32 {
				binding: 0,
				address: 19,
			}, // q[qb+d]
			/* 21 */ E::AddU32(60, 15), // kvb + d
			/* 22 */ E::LoadF32 {
				binding: 1,
				address: 21,
			}, // k[kvb+d]
			/* 23 */ E::MulF32(20, 22), // score term
			/* 24 */ E::LoadF32 {
				binding: 4,
				address: 19,
			}, // og[qb+d]
			/* 25 */ E::LoadF32 {
				binding: 2,
				address: 21,
			}, // v[kvb+d]
			/* 26 */ E::MulF32(24, 25), // pg term
			/* 27 */ E::LoadF32 {
				binding: 3,
				address: 19,
			}, // out[qb+d]
			/* 28 */ E::MulF32(24, 27), // delta term
			/* 29 */ E::RegisterF32 { tile: 2, index: 0 }, // score
			/* 30 */ E::MulF32(29, 4), // score*scale
			/* 31 */ E::AddU32(61, 16), // lse index = bh*seq + q
			/* 32 */ E::LoadF32 {
				binding: 5,
				address: 31,
			}, // lse
			/* 33 */ E::SubF32(30, 32),
			/* 34 */ E::ExpF32(33), // probability
			/* 35 */ E::RegisterF32 { tile: 5, index: 0 }, // probability
			/* 36 */ E::RegisterF32 { tile: 3, index: 0 }, // pg
			/* 37 */ E::RegisterF32 { tile: 4, index: 0 }, // delta
			/* 38 */ E::SubF32(36, 37), // pg - delta
			/* 39 */ E::MulF32(35, 38), // prob*(pg-delta)
			/* 40 */ E::MulF32(39, 4), // ssg term
			/* 41 */ E::RegisterU32 { tile: 8, index: 1 }, // d
			/* 42 */ E::MulU32(15, 3), // inner q*dim
			/* 43 */ E::AddU32(59, 42), // hb + q*dim
			/* 44 */ E::AddU32(43, 41), // + d
			/* 45 */ E::LoadF32 {
				binding: 0,
				address: 44,
			},
			/* 46 */ E::SharedF32 {
				tile: 1,
				offset: 15,
			}, // ssg[q]
			/* 47 */ E::MulF32(46, 45), // kg term
			/* 48 */ E::LoadF32 {
				binding: 4,
				address: 44,
			}, // og
			/* 49 */ E::SharedF32 {
				tile: 0,
				offset: 15,
			}, // sp[q]
			/* 50 */ E::MulF32(49, 48), // vg term
			/* 51 */ E::RegisterF32 { tile: 6, index: 0 }, // kg
			/* 52 */ E::AddF32(51, 47), // kg + term
			/* 53 */ E::RegisterF32 { tile: 7, index: 0 }, // vg
			/* 54 */ E::AddF32(53, 50), // vg + term
			/* 55 */ E::AddU32(60, 41), // kvb + d
			/* 56 */ E::AddF32(29, 23), // score + term
			/* 57 */ E::AddF32(36, 26), // pg + term
			/* 58 */ E::AddF32(37, 28), // delta + term
			/* 59 */ E::RegisterU32 { tile: 8, index: 2 }, // head_base
			/* 60 */ E::RegisterU32 { tile: 8, index: 3 }, // kv_base
			/* 61 */ E::RegisterU32 { tile: 8, index: 4 }, // bh*seq
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch_heads".into(),
				"sequence_length".into(),
				"head_dim".into(),
			],
			push_f32_scalars: vec!["scale".into()],
			shape: BlockShape {
				lanes: 64,
				thread_columns: 64,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 1024,
					stride: 1024,
				},
				TileDecl::Shared {
					rows: 1,
					columns: 1024,
					stride: 1024,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 5 },
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 5 },
				TileOp::AssignScalarU32 {
					tile: 8,
					index: 0,
					value: 13,
				},
				TileOp::AssignScalarU32 {
					tile: 8,
					index: 2,
					value: 10,
				},
				TileOp::AssignScalarU32 {
					tile: 8,
					index: 3,
					value: 9,
				},
				TileOp::AssignScalarU32 {
					tile: 8,
					index: 4,
					value: 7,
				},
				TileOp::LaneLoopRange {
					start: 12,
					step: 64,
					extent: Extent::Push(1),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 8,
							index: 0,
							value: 15,
						},
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 14,
						},
						TileOp::AssignScalar {
							tile: 3,
							index: 0,
							value: 14,
						},
						TileOp::AssignScalar {
							tile: 4,
							index: 0,
							value: 14,
						},
						TileOp::LaneLoopRange {
							start: 13,
							step: 1,
							extent: Extent::Push(2),
							body: vec![
								TileOp::AssignScalar {
									tile: 2,
									index: 0,
									value: 56,
								},
								TileOp::AssignScalar {
									tile: 3,
									index: 0,
									value: 57,
								},
								TileOp::AssignScalar {
									tile: 4,
									index: 0,
									value: 58,
								},
							],
						},
						TileOp::AssignScalar {
							tile: 5,
							index: 0,
							value: 34,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 16,
							value: 35,
						},
						TileOp::StoreShared {
							tile: 1,
							offset: 16,
							value: 40,
						},
					],
				},
				TileOp::Barrier,
				TileOp::LaneLoopRange {
					start: 11,
					step: 64,
					extent: Extent::Push(2),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 8,
							index: 1,
							value: 15,
						},
						TileOp::AssignScalar {
							tile: 6,
							index: 0,
							value: 14,
						},
						TileOp::AssignScalar {
							tile: 7,
							index: 0,
							value: 14,
						},
						TileOp::LaneLoopRange {
							start: 8,
							step: 1,
							extent: Extent::Push(1),
							body: vec![
								TileOp::AssignScalar {
									tile: 6,
									index: 0,
									value: 52,
								},
								TileOp::AssignScalar {
									tile: 7,
									index: 0,
									value: 54,
								},
							],
						},
						TileOp::StoreScalar {
							binding: 6,
							address: 55,
							value: 51,
						},
						TileOp::StoreScalar {
							binding: 7,
							address: 55,
							value: 53,
						},
					],
				},
			],
		}
	}

	/// Tiled `linear_triple` template — three `linear` projections
	/// sharing input and geometry, z-dispatched exactly like the donor:
	/// the weight/bias/output descriptor fields are selected by a
	/// `group_id.z` ternary, so `node.workgroups` keeps `[…, …, 3]`.
	pub(in crate::runtime) fn linear_triple(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body_z(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		ops.push(TileOp::StoreTileBias {
			acc: 2,
			bias_binding: 4,
			binding: 7,
			z: true,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
			bias_origin: None,
			act: None,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Input(6),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_gated_act` template — the `linear` schedule with a
	/// third shared tile staging the packed up-half weight rows
	/// (`column_base + gate_width`), dual accumulators, and the donor's
	/// `swiglu` epilogue materializing both the packed linear output and
	/// the half-width act output. Only `act_kind` 9 reaches this fused
	/// node, so the gated schedule is unconditional.
	pub(in crate::runtime) fn linear_gated_act(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, mut tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		tiles.push(TileDecl::Shared {
			rows: tile_columns,
			columns: tile_k,
			stride: tile_k + 1,
		});
		tiles.push(TileDecl::Register {
			rows: fragment_rows,
			columns: fragment_columns,
		});
		// tiles: shared input 0, shared gate weight 1, register gate
		// accumulator 2, shared up weight 3, register up accumulator 4.
		ops.insert(1, TileOp::ClearTile { tile: 4 });
		let TileOp::Loop { body, .. } = &mut ops[2] else {
			return Err(Error::internal("gemm body did not produce the K loop"));
		};
		body.insert(
			2,
			TileOp::LoadShared {
				tile: 3,
				binding: 1,
				row_origin: Extent::GroupY(tile_columns),
				column_origin: Extent::Induction,
				row_extent: Extent::Push(2),
				column_extent: Extent::Push(1),
				vec,
				row_offset: Some(Extent::HalfPush(2)),
				mask_origin: None,
				z: false,
				column_offset: None,
				transposed: false,
			},
		);
		body.insert(
			5,
			TileOp::DotShared {
				acc: 4,
				lhs: 0,
				rhs: 3,
			},
		);
		ops.push(TileOp::StoreTileGated {
			gate_acc: 2,
			up_acc: 4,
			bias_binding: 2,
			linear_binding: 3,
			act_binding: 4,
			linear_stride: Extent::Push(2),
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			gate_width: Extent::HalfPush(2),
			bias_origin: None,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
				"act_kind".into(),
			],
			push_f32_scalars: vec!["alpha".into()],
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_concat` template — the `linear` schedule whose
	/// operand-`A` stage gathers the K axis across three part buffers by
	/// k range and materializes the concat output on column tile zero.
	/// Buffers `[first, second, third, weight, bias, concat, output]`,
	/// push scalars `[batch, input_features, output_features, w0, w1,
	/// w2]`.
	pub(in crate::runtime) fn linear_concat(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		let TileOp::Loop { body, .. } = &mut ops[1] else {
			return Err(Error::internal("gemm body did not produce the K loop"));
		};
		body[0] = TileOp::LoadSharedConcat {
			tile: 0,
			bindings: [0, 1, 2],
			widths: [3, 4, 5],
			materialize: 5,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::Induction,
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(1),
		};
		if let TileOp::LoadShared { binding, .. } = &mut body[1] {
			*binding = 3;
		}
		ops.push(TileOp::StoreTileBias {
			acc: 2,
			bias_binding: 4,
			binding: 6,
			z: false,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
			bias_origin: None,
			act: None,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
				"first_width".into(),
				"second_width".into(),
				"third_width".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Tiled `linear_residual_norm` template — the `linear` schedule plus
	/// the donor's residual epilogue staging every tile row's `sum` into
	/// a `BM × BN` shared tile, then one lane per row serially reduces
	/// mean/variance, and `lanes/BM` lanes per row write the normalized
	/// and weighted outputs. Legal only when a single column tile covers
	/// the row (`output_features <= tile_columns`) — the route proves it
	/// through `node.workgroups[1] == 1`. Buffers `[input, weight, bias,
	/// residual, norm_weight, norm_bias, linear_out, sum_out, output,
	/// normalized, inv_stddev]`, push `[batch, input, output]` plus f32
	/// `epsilon`.
	pub(in crate::runtime) fn linear_residual_norm(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, mut tiles, mut ops) = gemm_tile_body(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(1, 0, 2),
		)?;
		if !shape.lanes.is_multiple_of(tile_rows) {
			return Err(Error::invalid_argument(
				"residual_norm needs a whole number of lanes per tile row",
			));
		}
		let lanes_per_row = shape.lanes / tile_rows;
		tiles.push(TileDecl::Shared {
			rows: tile_rows,
			columns: tile_columns,
			stride: tile_columns,
		});
		tiles.push(TileDecl::Shared {
			rows: 1,
			columns: tile_rows,
			stride: tile_rows,
		});
		tiles.push(TileDecl::Shared {
			rows: 1,
			columns: tile_rows,
			stride: tile_rows,
		});
		for _ in 0..4 {
			tiles.push(TileDecl::Register {
				rows: 1,
				columns: 1,
			});
		}
		// tiles: shared input 0, shared weight 1, register acc 2,
		// shared sum tile 3, shared row mean 4, shared row inv-stddev 5,
		// register row sum 6, row square-sum 7, row mean 8, row inv 9.
		let mut e = Vec::new();
		macro_rules! x {
			($v:expr) => {{
				e.push($v);
				e.len() - 1
			}};
		}
		let lane = x!(LaneExpr::Lane);
		let gx = x!(LaneExpr::GroupX);
		let tr = x!(LaneExpr::ConstU32(tile_rows));
		let tc = x!(LaneExpr::ConstU32(tile_columns));
		let batch = x!(LaneExpr::Push(0));
		let out = x!(LaneExpr::Push(2));
		let row_base = x!(LaneExpr::MulU32(gx, tr));
		let remain = x!(LaneExpr::SubU32(batch, row_base));
		let valid = x!(LaneExpr::MinU32(tr, remain));
		let stat_cond = x!(LaneExpr::LtU32(lane, valid));
		let zero = x!(LaneExpr::ConstF32(0));
		let ind = x!(LaneExpr::Induction);
		let row_off = x!(LaneExpr::MulU32(lane, tc));
		let stat_idx = x!(LaneExpr::AddU32(row_off, ind));
		let stat_val = x!(LaneExpr::SharedF32 {
			tile: 3,
			offset: stat_idx
		});
		let out_f = x!(LaneExpr::F32FromU32(out));
		let rsum = x!(LaneExpr::RegisterF32 { tile: 6, index: 0 });
		let mean = x!(LaneExpr::DivF32(rsum, out_f));
		let rmean = x!(LaneExpr::RegisterF32 { tile: 8, index: 0 });
		let centered = x!(LaneExpr::SubF32(stat_val, rmean));
		let sq = x!(LaneExpr::MulF32(centered, centered));
		let rsq = x!(LaneExpr::RegisterF32 { tile: 7, index: 0 });
		let var = x!(LaneExpr::DivF32(rsq, out_f));
		let eps = x!(LaneExpr::PushF32(0));
		let veps = x!(LaneExpr::AddF32(var, eps));
		let inv = x!(LaneExpr::RsqrtF32(veps));
		let rinv = x!(LaneExpr::RegisterF32 { tile: 9, index: 0 });
		let stat_addr = x!(LaneExpr::AddU32(row_base, lane));
		let lpr = x!(LaneExpr::ConstU32(lanes_per_row));
		let nrow = x!(LaneExpr::DivU32(lane, lpr));
		let orow = x!(LaneExpr::AddU32(row_base, nrow));
		let ncond = x!(LaneExpr::LtU32(orow, batch));
		let nrow_off = x!(LaneExpr::MulU32(nrow, tc));
		let nidx = x!(LaneExpr::AddU32(nrow_off, ind));
		let nval = x!(LaneExpr::SharedF32 {
			tile: 3,
			offset: nidx
		});
		let nmean = x!(LaneExpr::SharedF32 {
			tile: 4,
			offset: nrow
		});
		let ninv = x!(LaneExpr::SharedF32 {
			tile: 5,
			offset: nrow
		});
		let dnorm = x!(LaneExpr::SubF32(nval, nmean));
		let normed = x!(LaneExpr::MulF32(dnorm, ninv));
		let elrow = x!(LaneExpr::MulU32(orow, out));
		let el = x!(LaneExpr::AddU32(elrow, ind));
		let nw = x!(LaneExpr::LoadF32 {
			binding: 4,
			address: ind,
		});
		let nb = x!(LaneExpr::LoadF32 {
			binding: 5,
			address: ind,
		});
		let nwmul = x!(LaneExpr::MulF32(normed, nw));
		let outv = x!(LaneExpr::AddF32(nwmul, nb));
		let start = x!(LaneExpr::ModU32(lane, lpr));
		ops.push(TileOp::StoreTileResidualNorm {
			acc: 2,
			bias_binding: 2,
			residual_binding: 3,
			linear_binding: 6,
			sum_binding: 7,
			tile: 3,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
		});
		ops.push(TileOp::Barrier);
		ops.push(TileOp::LaneIf {
			cond: stat_cond,
			body: vec![
				TileOp::LaneFold {
					acc: 6,
					init: zero,
					extent: Extent::Push(2),
					op: LaneReduce::Sum,
					element: stat_val,
					serial: true,
				},
				TileOp::AssignScalar {
					tile: 8,
					index: 0,
					value: mean,
				},
				TileOp::LaneFold {
					acc: 7,
					init: zero,
					extent: Extent::Push(2),
					op: LaneReduce::Sum,
					element: sq,
					serial: true,
				},
				TileOp::AssignScalar {
					tile: 9,
					index: 0,
					value: inv,
				},
				TileOp::StoreShared {
					tile: 4,
					offset: lane,
					value: rmean,
				},
				TileOp::StoreShared {
					tile: 5,
					offset: lane,
					value: rinv,
				},
				TileOp::StoreScalar {
					binding: 10,
					address: stat_addr,
					value: rinv,
				},
			],
		});
		ops.push(TileOp::Barrier);
		ops.push(TileOp::LaneLoopRange {
			start,
			step: lanes_per_row,
			extent: Extent::Push(2),
			body: vec![TileOp::LaneIf {
				cond: ncond,
				body: vec![
					TileOp::StoreScalar {
						binding: 9,
						address: el,
						value: normed,
					},
					TileOp::StoreScalar {
						binding: 8,
						address: el,
						value: outv,
					},
				],
			}],
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
				Binding::Output(3),
				Binding::Output(4),
			],
			push_scalars: vec![
				"batch".into(),
				"input_features".into(),
				"output_features".into(),
			],
			push_f32_scalars: vec!["epsilon".into()],
			shape,
			tiles,
			lane_exprs: e,
			ops,
		})
	}

	/// `qkv_projection_bias` template — the linear-family donor's
	/// three-weight staging schedule: one input tile plus query/key/
	/// value weight tiles feeding three accumulators, then three
	/// bias+store epilogues. Buffers `[input, qw, kw, vw, qb, kb, vb,
	/// qout, kout, vout]`, push `[m, n, k]`. Donor geometry is
	/// 32x32x32 with 1x4 fragments and scalar staging.
	pub(in crate::runtime) fn qkv_projection_bias(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body_multi(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(2, 0, 1),
			&[1, 2, 3],
		)?;
		// tiles: shared input 0, shared q/k/v 1..3, acc q/k/v 4..6.
		for (acc, bias_binding, binding) in [(4usize, 4usize, 7usize), (5, 5, 8), (6, 6, 9)] {
			ops.push(TileOp::StoreTileBias {
				acc,
				bias_binding,
				binding,
				z: false,
				row_origin: Extent::GroupX(tile_rows),
				column_origin: Extent::GroupY(tile_columns),
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(1),
				bias_origin: None,
				act: None,
			});
		}
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Input(6),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
			],
			push_scalars: vec!["m".into(), "n".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// `gate_up_swiglu_bias` template — two independent `[n, k]`
	/// projections (gate and up weights) feeding the donor's
	/// `silu(gate + bias) * (up + bias)` epilogue into one `[m, n]`
	/// output. Buffers `[input, gate_w, up_w, gate_b, up_b, output]`,
	/// push `[m, n, k]`. Donor geometry is 32x32x32 with 1x4 fragments.
	pub(in crate::runtime) fn gate_up_swiglu_bias(
		tile_rows: u32,
		tile_columns: u32,
		tile_k: u32,
		fragment_rows: u32,
		fragment_columns: u32,
		vec: u32,
	) -> Result<Self> {
		let (shape, tiles, mut ops) = gemm_tile_body_multi(
			(
				tile_rows,
				tile_columns,
				tile_k,
				fragment_rows,
				fragment_columns,
				vec,
			),
			(2, 0, 1),
			&[1, 2],
		)?;
		// tiles: shared input 0, shared gate/up 1..2, acc gate/up 3..4.
		ops.push(TileOp::StoreTileGateUp {
			gate_acc: 3,
			up_acc: 4,
			gate_bias_binding: 3,
			up_bias_binding: 4,
			binding: 5,
			row_origin: Extent::GroupX(tile_rows),
			column_origin: Extent::GroupY(tile_columns),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(1),
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Output(0),
			],
			push_scalars: vec!["m".into(), "n".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs: Vec::new(),
			ops,
		})
	}

	/// Axis-aware stable-softmax template — the tile-IR re-expression
	/// of `softmax.slang`: one workgroup per `(outer, inner)` slice,
	/// per-lane strided max/sum folds, cooperative tree reductions over
	/// a `lanes`-wide shared scratch, and a final strided normalize
	/// pass. The emitted program preserves the donor ABI exactly —
	/// `[outer_size, dim_size, inner_size]` push scalars behind two
	/// buffer indices — so the node's dispatch geometry is unchanged.
	///
	/// `lanes` is the workgroup width and the shared scratch length; it
	/// must be a power of two within the Vulkan workgroup floor.
	/// `Err` rejects anything else.
	pub(in crate::runtime) fn softmax(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"softmax lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // outer_size
			/* 1 */ E::Push(1), // dim_size
			/* 2 */ E::Push(2), // inner_size
			/* 3 */ E::GroupX, // group
			/* 4 */ E::MulU32(0, 2), // outer*inner bound
			/* 5 */ E::DivU32(3, 2), // outer index
			/* 6 */ E::ModU32(3, 2), // inner index
			/* 7 */ E::MulU32(5, 1), // outer*dim
			/* 8 */ E::MulU32(7, 2), // outer*dim*inner
			/* 9 */ E::AddU32(8, 6), // row base
			/* 10 */ E::Induction, // element index
			/* 11 */ E::MulU32(10, 2), // index*inner
			/* 12 */ E::AddU32(9, 11), // element address
			/* 13 */ E::LoadF32 {
				binding: 0,
				address: 12,
			}, // x
			/* 14 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 15 */ E::Lane, // lane
			/* 16 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 17 */ E::RegisterF32 { tile: 2, index: 0 }, // row_max
			/* 18 */ E::SubF32(13, 17), // x - row_max
			/* 19 */ E::ExpF32(18), // exp(x - row_max)
			/* 20 */ E::ConstF32(0x0000_0000), // 0.0
			/* 21 */ E::ConstU32(0),
			/* 22 */ E::SharedF32 {
				tile: 0,
				offset: 21,
			}, // shared[0]
			/* 23 */ E::RegisterF32 { tile: 3, index: 0 }, // inverse_sum
			/* 24 */ E::MulF32(19, 23), // exp * inverse_sum
			/* 25 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 26 */ E::DivF32(25, 22), // 1.0 / sum
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 4 },
				TileOp::LaneFold {
					acc: 1,
					init: 14,
					extent: Extent::Push(1),
					op: LaneReduce::Max,
					element: 13,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Max,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 22,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 1,
					init: 20,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 19,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 26,
				},
				TileOp::Barrier,
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 12,
						value: 24,
					}],
				},
			],
		})
	}

	/// Stable log-softmax — the `log_softmax.slang` donor schedule:
	/// identical max/sum reduction phases to `softmax`, with a
	/// `row_max + log(sum)` normalizer and a `x - log_normalizer`
	/// output pass. Same donor ABI.
	pub(in crate::runtime) fn log_softmax(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"log_softmax lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // outer_size
			/* 1 */ E::Push(1), // dim_size
			/* 2 */ E::Push(2), // inner_size
			/* 3 */ E::GroupX, // group
			/* 4 */ E::MulU32(0, 2), // outer*inner bound
			/* 5 */ E::DivU32(3, 2), // outer index
			/* 6 */ E::ModU32(3, 2), // inner index
			/* 7 */ E::MulU32(5, 1), // outer*dim
			/* 8 */ E::MulU32(7, 2), // outer*dim*inner
			/* 9 */ E::AddU32(8, 6), // row base
			/* 10 */ E::Induction, // element index
			/* 11 */ E::MulU32(10, 2), // index*inner
			/* 12 */ E::AddU32(9, 11), // element address
			/* 13 */ E::LoadF32 {
				binding: 0,
				address: 12,
			}, // x
			/* 14 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 15 */ E::Lane, // lane
			/* 16 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 17 */ E::RegisterF32 { tile: 2, index: 0 }, // row_max
			/* 18 */ E::SubF32(13, 17), // x - row_max
			/* 19 */ E::ExpF32(18), // exp(x - row_max)
			/* 20 */ E::ConstF32(0x0000_0000), // 0.0
			/* 21 */ E::ConstU32(0),
			/* 22 */ E::SharedF32 {
				tile: 0,
				offset: 21,
			}, // shared[0]
			/* 23 */ E::RegisterF32 { tile: 3, index: 0 }, // log_normalizer
			/* 24 */ E::LogF32(22), // log(sum)
			/* 25 */ E::AddF32(17, 24), // row_max + log(sum)
			/* 26 */ E::SubF32(13, 23), // x - log_normalizer
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 4 },
				TileOp::LaneFold {
					acc: 1,
					init: 14,
					extent: Extent::Push(1),
					op: LaneReduce::Max,
					element: 13,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Max,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 22,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 1,
					init: 20,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 19,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 25,
				},
				TileOp::Barrier,
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 12,
						value: 26,
					}],
				},
			],
		})
	}

	/// Output-saved softmax adjoint — the `softmax_backward.slang`
	/// donor schedule: one strided `sum(dy * y)` fold, a shared tree
	/// reduction, then `dx = y * (dy - dot)`. ABI: three buffers
	/// (`forward_output, output_gradient, input_gradient`),
	/// `[outer_size, dim_size, inner_size]` scalars.
	pub(in crate::runtime) fn softmax_backward(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"softmax_backward lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0),
			/* 1 */ E::Push(1),
			/* 2 */ E::Push(2),
			/* 3 */ E::GroupX,
			/* 4 */ E::MulU32(0, 2),
			/* 5 */ E::DivU32(3, 2),
			/* 6 */ E::ModU32(3, 2),
			/* 7 */ E::MulU32(5, 1),
			/* 8 */ E::MulU32(7, 2),
			/* 9 */ E::AddU32(8, 6),
			/* 10 */ E::Induction,
			/* 11 */ E::MulU32(10, 2),
			/* 12 */ E::AddU32(9, 11), // element address
			/* 13 */ E::LoadF32 {
				binding: 0,
				address: 12,
			}, // y
			/* 14 */ E::LoadF32 {
				binding: 1,
				address: 12,
			}, // dy
			/* 15 */ E::MulF32(14, 13), // dy * y
			/* 16 */ E::ConstF32(0x0000_0000), // 0.0
			/* 17 */ E::Lane,
			/* 18 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 19 */ E::ConstU32(0),
			/* 20 */ E::SharedF32 {
				tile: 0,
				offset: 19,
			}, // dot
			/* 21 */ E::RegisterF32 { tile: 2, index: 0 }, // dot reg
			/* 22 */ E::SubF32(14, 21), // dy - dot
			/* 23 */ E::MulF32(13, 22), // y * (dy - dot)
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 4 },
				TileOp::LaneFold {
					acc: 1,
					init: 16,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 15,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 17,
					value: 18,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 20,
				},
				TileOp::Barrier,
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 12,
						value: 23,
					}],
				},
			],
		})
	}

	/// Log-softmax adjoint — the `log_softmax_backward.slang` donor
	/// schedule: a strided `sum(dy)` fold, shared tree reduction, then
	/// `dx = dy - exp(y) * gradient_sum`. Same ABI as
	/// `softmax_backward`.
	pub(in crate::runtime) fn log_softmax_backward(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"log_softmax_backward lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0),
			/* 1 */ E::Push(1),
			/* 2 */ E::Push(2),
			/* 3 */ E::GroupX,
			/* 4 */ E::MulU32(0, 2),
			/* 5 */ E::DivU32(3, 2),
			/* 6 */ E::ModU32(3, 2),
			/* 7 */ E::MulU32(5, 1),
			/* 8 */ E::MulU32(7, 2),
			/* 9 */ E::AddU32(8, 6),
			/* 10 */ E::Induction,
			/* 11 */ E::MulU32(10, 2),
			/* 12 */ E::AddU32(9, 11), // element address
			/* 13 */ E::LoadF32 {
				binding: 1,
				address: 12,
			}, // dy
			/* 14 */ E::ConstF32(0x0000_0000), // 0.0
			/* 15 */ E::Lane,
			/* 16 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 17 */ E::ConstU32(0),
			/* 18 */ E::SharedF32 {
				tile: 0,
				offset: 17,
			}, // gradient_sum
			/* 19 */ E::RegisterF32 { tile: 2, index: 0 }, // sum reg
			/* 20 */ E::LoadF32 {
				binding: 0,
				address: 12,
			}, // y
			/* 21 */ E::ExpF32(20), // exp(y)
			/* 22 */ E::MulF32(21, 19), // exp(y) * sum
			/* 23 */ E::SubF32(13, 22), // dy - exp(y)*sum
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 4 },
				TileOp::LaneFold {
					acc: 1,
					init: 14,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 13,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 15,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 18,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 12,
						value: 23,
					}],
				},
			],
		})
	}

	/// RMSNorm forward — the `rms_norm.slang` donor schedule: one
	/// workgroup per row, a strided square-sum fold, shared tree
	/// reduction, `1 / sqrt(mean + eps)` and the weight-scaled
	/// normalize pass. ABI: three buffers,
	/// `[rows, columns]` u32 scalars plus `float epsilon`.
	pub(in crate::runtime) fn rms_norm(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"rms_norm lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::GroupX, // row
			/* 3 */ E::MulU32(2, 1), // row base
			/* 4 */ E::Induction, // column index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // x
			/* 7 */ E::MulF32(6, 6), // x*x
			/* 8 */ E::ConstF32(0x0000_0000), // 0.0
			/* 9 */ E::Lane, // lane
			/* 10 */ E::RegisterF32 { tile: 1, index: 0 }, // fold acc
			/* 11 */ E::ConstU32(0),
			/* 12 */ E::SharedF32 {
				tile: 0,
				offset: 11,
			}, // partial[0]
			/* 13 */ E::F32FromU32(1), // float(columns)
			/* 14 */ E::DivF32(12, 13), // mean square
			/* 15 */ E::PushF32(0), // epsilon
			/* 16 */ E::AddF32(14, 15), // mean + eps
			/* 17 */ E::SqrtF32(16), // sqrt
			/* 18 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 19 */ E::DivF32(18, 17), // inverse_rms
			/* 20 */ E::RegisterF32 { tile: 2, index: 0 }, // inverse_rms
			/* 21 */ E::LoadF32 {
				binding: 1,
				address: 4,
			}, // weight[column]
			/* 22 */ E::MulF32(6, 20), // x * inverse_rms
			/* 23 */ E::MulF32(22, 21), // * weight
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec!["rows".into(), "columns".into()],
			push_f32_scalars: vec!["epsilon".into()],
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneFold {
					acc: 1,
					init: 8,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 7,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 9,
					value: 10,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 19,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 5,
						value: 23,
					}],
				},
			],
		})
	}

	/// LayerNorm forward — the `layer_norm.slang` donor schedule: one
	/// workgroup per row, strided sum and centered square-sum folds over
	/// two shared scratch tiles, `rsqrt(variance + eps)`, an exclusive
	/// lane-0 `inverse_stddev` store and the affine normalize pass.
	/// ABI: six buffers (`input, weight, bias, output, normalized,
	/// inverse_stddev`), `[rows, columns]` u32 scalars plus
	/// `float epsilon`.
	pub(in crate::runtime) fn layer_norm(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"layer_norm lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::GroupX, // row
			/* 3 */ E::MulU32(2, 1), // row offset
			/* 4 */ E::Induction, // column index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // x
			/* 7 */ E::ConstF32(0x0000_0000), // 0.0
			/* 8 */ E::Lane, // lane
			/* 9 */ E::RegisterF32 { tile: 2, index: 0 }, // fold acc
			/* 10 */ E::ConstU32(0),
			/* 11 */ E::SharedF32 {
				tile: 0,
				offset: 10,
			}, // partial_sum[0]
			/* 12 */ E::F32FromU32(1), // float(columns)
			/* 13 */ E::DivF32(11, 12), // mean
			/* 14 */ E::RegisterF32 { tile: 3, index: 0 }, // mean
			/* 15 */ E::SubF32(6, 14), // centered
			/* 16 */ E::MulF32(15, 15), // centered^2
			/* 17 */ E::SharedF32 {
				tile: 1,
				offset: 10,
			}, // partial_sq[0]
			/* 18 */ E::DivF32(17, 12), // variance
			/* 19 */ E::PushF32(0), // epsilon
			/* 20 */ E::AddF32(18, 19), // variance + eps
			/* 21 */ E::RsqrtF32(20), // inverse_stddev
			/* 22 */ E::RegisterF32 { tile: 4, index: 0 }, // inv reg
			/* 23 */ E::MulF32(15, 22), // normalized
			/* 24 */ E::LoadF32 {
				binding: 1,
				address: 4,
			}, // weight
			/* 25 */ E::MulF32(23, 24), // n*w
			/* 26 */ E::LoadF32 {
				binding: 2,
				address: 4,
			}, // bias
			/* 27 */ E::AddF32(25, 26), // n*w + bias
		];
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
			],
			push_scalars: vec!["rows".into(), "columns".into()],
			push_f32_scalars: vec!["epsilon".into()],
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneFold {
					acc: 2,
					init: 7,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 6,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 8,
					value: 9,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 13,
				},
				TileOp::LaneFold {
					acc: 2,
					init: 7,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 16,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 1,
					offset: 8,
					value: 9,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 21,
				},
				TileOp::LaneGuard {
					lane: 10,
					body: vec![TileOp::StoreScalar {
						binding: 5,
						address: 2,
						value: 22,
					}],
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::StoreScalar {
							binding: 4,
							address: 5,
							value: 23,
						},
						TileOp::StoreScalar {
							binding: 3,
							address: 5,
							value: 27,
						},
					],
				},
			],
		})
	}

	/// `layer_norm_backward` — the row-cooperative LayerNorm data
	/// adjoint: dual strided folds (`sum(og*w)` and
	/// `sum(og*w*normalized)`) each reduced through the shared
	/// `partial[]` tree, the saved `inverse_stddev[row]` load, and the
	/// strided epilogue writing `input_gradient` and the
	/// `weight_contribution` feed. ABI: six buffers
	/// `[normalized, inverse_stddev, weight, output_gradient,
	/// input_gradient, weight_contribution]`, `[rows, columns]` push,
	/// `[rows, 1, 1]` grid. The donor's lane-0 shared-scalar handoffs
	/// collapse to direct `partial[0]` register reads; the reuse
	/// barrier is preserved.
	pub(in crate::runtime) fn layer_norm_backward(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"layer_norm_backward lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::GroupX, // row
			/* 3 */ E::MulU32(2, 1), // row base
			/* 4 */ E::Induction, // column index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 3,
				address: 5,
			}, // output_gradient
			/* 7 */ E::LoadF32 {
				binding: 2,
				address: 4,
			}, // weight[column]
			/* 8 */ E::MulF32(6, 7), // scaled_gradient
			/* 9 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // normalized
			/* 10 */ E::MulF32(8, 9), // scaled_gradient * normalized
			/* 11 */ E::ConstF32(0x0000_0000), // 0.0
			/* 12 */ E::Lane, // lane
			/* 13 */ E::ConstU32(0),
			/* 14 */ E::RegisterF32 { tile: 1, index: 0 }, // sg acc
			/* 15 */ E::RegisterF32 { tile: 2, index: 0 }, // sgn acc
			/* 16 */ E::SharedF32 {
				tile: 0,
				offset: 13,
			}, // partial[0]
			/* 17 */ E::RegisterF32 { tile: 3, index: 0 }, // sg total
			/* 18 */ E::RegisterF32 { tile: 4, index: 0 }, // sgn total
			/* 19 */ E::LoadF32 {
				binding: 1,
				address: 2,
			}, // inverse_stddev[row]
			/* 20 */ E::RegisterF32 { tile: 5, index: 0 }, // istddev reg
			/* 21 */ E::F32FromU32(1), // float(columns)
			/* 22 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 23 */ E::DivF32(22, 21), // inverse_columns
			/* 24 */ E::MulF32(17, 23), // sg * inverse_columns
			/* 25 */ E::MulF32(9, 18), // normalized * sgn
			/* 26 */ E::MulF32(25, 23), // * inverse_columns
			/* 27 */ E::SubF32(8, 24), // og*w - sg*inv_cols
			/* 28 */ E::SubF32(27, 26), // - n*sgn*inv_cols
			/* 29 */ E::MulF32(20, 28), // input_gradient
			/* 30 */ E::MulF32(6, 9), // weight_contribution
		];
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec!["rows".into(), "columns".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneFoldDual {
					acc: 1,
					acc2: 2,
					init: 11,
					extent: Extent::Push(1),
					element: 8,
					element2: 10,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 12,
					value: 14,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 16,
				},
				TileOp::Barrier,
				TileOp::StoreShared {
					tile: 0,
					offset: 12,
					value: 15,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 16,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: 19,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::StoreScalar {
							binding: 4,
							address: 5,
							value: 29,
						},
						TileOp::StoreScalar {
							binding: 5,
							address: 5,
							value: 30,
						},
					],
				},
			],
		})
	}

	/// `rms_norm_backward` — the row-cooperative RMSNorm data adjoint:
	/// a strided square-sum fold reduced to `inverse_rms`, then a
	/// strided `sum(og*w*x)` dot fold, and the epilogue writing
	/// `input_gradient` and `weight_contribution`. ABI: five buffers
	/// `[input, weight, output_gradient, input_gradient,
	/// weight_contribution]`, `[rows, columns]` plus `float epsilon`
	/// push, `[rows, 1, 1]` grid.
	pub(in crate::runtime) fn rms_norm_backward(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"rms_norm_backward lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::GroupX, // row
			/* 3 */ E::MulU32(2, 1), // row base
			/* 4 */ E::Induction, // column index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // input
			/* 7 */ E::MulF32(6, 6), // x*x
			/* 8 */ E::ConstF32(0x0000_0000), // 0.0
			/* 9 */ E::Lane, // lane
			/* 10 */ E::ConstU32(0),
			/* 11 */ E::RegisterF32 { tile: 1, index: 0 }, // sq acc
			/* 12 */ E::SharedF32 {
				tile: 0,
				offset: 10,
			}, // partial[0]
			/* 13 */ E::RegisterF32 { tile: 2, index: 0 }, // sq total
			/* 14 */ E::F32FromU32(1), // float(columns)
			/* 15 */ E::DivF32(13, 14), // mean square
			/* 16 */ E::PushF32(0), // epsilon
			/* 17 */ E::AddF32(15, 16), // mean + eps
			/* 18 */ E::SqrtF32(17), // sqrt
			/* 19 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 20 */ E::DivF32(19, 18), // inverse_rms
			/* 21 */ E::RegisterF32 { tile: 3, index: 0 }, // irms reg
			/* 22 */ E::LoadF32 {
				binding: 2,
				address: 5,
			}, // output_gradient
			/* 23 */ E::LoadF32 {
				binding: 1,
				address: 4,
			}, // weight[column]
			/* 24 */ E::MulF32(22, 23), // og*w
			/* 25 */ E::MulF32(24, 6), // og*w*x
			/* 26 */ E::RegisterF32 { tile: 4, index: 0 }, // dot acc
			/* 27 */ E::RegisterF32 { tile: 5, index: 0 }, // dot total
			/* 28 */ E::DivF32(19, 14), // inverse_columns
			/* 29 */ E::MulF32(6, 27), // x*dot
			/* 30 */ E::MulF32(29, 21), // *inverse_rms
			/* 31 */ E::MulF32(30, 21), // *inverse_rms
			/* 32 */ E::MulF32(31, 28), // *inverse_columns
			/* 33 */ E::SubF32(24, 32), // og*w - term
			/* 34 */ E::MulF32(21, 33), // input_gradient
			/* 35 */ E::MulF32(22, 6), // og*x
			/* 36 */ E::MulF32(35, 21), // *inverse_rms
		];
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec!["rows".into(), "columns".into()],
			push_f32_scalars: vec!["epsilon".into()],
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneFold {
					acc: 1,
					init: 8,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 7,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 9,
					value: 11,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 12,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 20,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 4,
					init: 8,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 25,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 9,
					value: 26,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: 12,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::StoreScalar {
							binding: 3,
							address: 5,
							value: 34,
						},
						TileOp::StoreScalar {
							binding: 4,
							address: 5,
							value: 36,
						},
					],
				},
			],
		})
	}

	/// `grouped_linear_m_act` — the MoE expert-grouped gate/up donor:
	/// a 32x32 output tile over gate columns, one 16-wide K tile, 64
	/// lanes, a 4x4 fragment, and scalar staging. Every lane runs the
	/// donor's `expert_for_row` while-scan twice (tile-first and
	/// tile-last row); tiles spanning an expert boundary take the
	/// serial `BoundaryGateUp` path and return, single-expert tiles
	/// stage gate/up weight rows under the shared `first_expert` offset
	/// and finish with the dual-bias swiglu epilogue. Descriptor
	/// bindings are the fused node's
	/// `[input, weight, bias, offsets, output, act_output]`.
	pub(in crate::runtime) fn grouped_linear_m_act() -> Result<Self> {
		Self::grouped_body(true, true)
	}

	/// `grouped_linear_m` — the grouped affine donor: five bindings
	/// `[input, weight, bias, offsets, output]`, push
	/// `[rows, outputs, inner, experts]`, and the donor's
	/// `[rows/32, outputs/32, 1]` grid.
	pub(in crate::runtime) fn grouped_linear_m() -> Result<Self> {
		Self::grouped_body(true, false)
	}

	/// `grouped_gemm_m` — the grouped projection without a bias
	/// epilogue: four bindings `[input, weight, offsets, output]` and
	/// the same push/grid contract as `grouped_linear_m`.
	pub(in crate::runtime) fn grouped_gemm_m() -> Result<Self> {
		Self::grouped_body(false, false)
	}

	/// Shared `grouped_*_m` body — 32x32x16 tiles, 64 lanes, 4x4
	/// fragments, scalar staging. `has_bias` selects the donor's bias
	/// epilogue and the extra bias binding; `has_act` selects the
	/// gate/up split (`swiglu` epilogue, half-width mask/output).
	fn grouped_body(has_bias: bool, has_act: bool) -> Result<Self> {
		let weights: &[usize] = if has_act { &[1, 1] } else { &[1] };
		let (shape, mut tiles, mut ops) =
			gemm_tile_body_multi((32, 32, 16, 4, 4, 1), (2, 0, 1), weights)?;
		// tiles: shared input 0, shared weight 1(, shared up weight 2),
		// accumulator 2(, up accumulator 4), expert scans last two.
		let acc = 1 + weights.len();
		let up_acc = acc + 1;
		let scan_first = tiles.len();
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		});
		let scan_last = tiles.len();
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		});
		let mut lane_exprs = vec![
			LaneExpr::GroupX,       // 0
			LaneExpr::ConstU32(32), // 1
			LaneExpr::MulU32(0, 1), // 2: row_base
			LaneExpr::Push(0),      // 3: rows
			LaneExpr::GeU32(2, 3),  // 4: row_base >= rows
			LaneExpr::GroupY,       // 5
			LaneExpr::MulU32(5, 1), // 6: column_base
			LaneExpr::Push(1),      // 7: output_features
		];
		// The column bound is `outputs` for the wide donors and
		// `outputs / 2` for the gate/up split.
		let column_bound = if has_act {
			lane_exprs.push(LaneExpr::ConstU32(2)); // 8
			lane_exprs.push(LaneExpr::DivU32(7, 8)); // 9: gate_width
			9
		} else {
			7
		};
		let column_oor = lane_exprs.len();
		lane_exprs.push(LaneExpr::GeU32(6, column_bound));
		let guard = lane_exprs.len();
		lane_exprs.push(LaneExpr::OrBool(4, column_oor));
		let base_plus = lane_exprs.len();
		lane_exprs.push(LaneExpr::AddU32(2, 1));
		let row_end = lane_exprs.len();
		lane_exprs.push(LaneExpr::MinU32(base_plus, 3));
		let one = lane_exprs.len();
		lane_exprs.push(LaneExpr::ConstU32(1));
		let last_row = lane_exprs.len();
		lane_exprs.push(LaneExpr::SubU32(row_end, one));
		let first_reg = lane_exprs.len();
		lane_exprs.push(LaneExpr::RegisterF32 {
			tile: scan_first,
			index: 0,
		});
		let first = lane_exprs.len();
		lane_exprs.push(LaneExpr::U32FromF32(first_reg));
		let last_reg = lane_exprs.len();
		lane_exprs.push(LaneExpr::RegisterF32 {
			tile: scan_last,
			index: 0,
		});
		let last = lane_exprs.len();
		lane_exprs.push(LaneExpr::U32FromF32(last_reg));
		let boundary_cond = lane_exprs.len();
		lane_exprs.push(LaneExpr::NeqU32(first, last));
		let expert_row = lane_exprs.len();
		lane_exprs.push(LaneExpr::MulU32(first, 7));
		let gate_origin = lane_exprs.len();
		lane_exprs.push(LaneExpr::AddU32(expert_row, 6));
		let up_origin = if has_act {
			lane_exprs.push(LaneExpr::AddU32(gate_origin, column_bound));
			Some(lane_exprs.len() - 1)
		} else {
			None
		};
		// Expert-offset staging: weight rows address
		// `(first_expert * outputs + column)` but mask against
		// `column < column_bound`, not `row < outputs`.
		let TileOp::Loop { body, .. } = ops.last_mut().unwrap() else {
			return Err(Error::internal("gemm body did not produce the K loop"));
		};
		for (index, origin) in [Some(gate_origin), up_origin].iter().flatten().enumerate() {
			let TileOp::LoadShared {
				row_origin,
				row_extent,
				mask_origin,
				..
			} = &mut body[1 + index]
			else {
				return Err(Error::internal("gemm body weight staging missing"));
			};
			*row_origin = Extent::Expr(*origin);
			*row_extent = Extent::Expr(column_bound);
			*mask_origin = Some(Extent::GroupY(32));
		}
		// Prefix: dispatch guard, the two expert scans, and the
		// boundary fallback that returns the whole workgroup.
		let boundary = if has_act {
			TileOp::BoundaryGateUp {
				acc,
				offsets: 3,
				input: 0,
				weight: 1,
				bias: 2,
				output: 4,
				act: 5,
				experts: Extent::Push(3),
				rows: Extent::Push(0),
				outputs: Extent::Push(1),
				inner: Extent::Push(2),
				gate_width: Extent::HalfPush(1),
				row_origin: Extent::GroupX(32),
				column_origin: Extent::GroupY(32),
			}
		} else {
			TileOp::BoundaryLinear {
				acc,
				offsets: if has_bias { 3 } else { 2 },
				input: 0,
				weight: 1,
				bias: has_bias.then_some(2),
				output: if has_bias { 4 } else { 3 },
				experts: Extent::Push(3),
				rows: Extent::Push(0),
				outputs: Extent::Push(1),
				inner: Extent::Push(2),
				column_extent: Extent::Expr(column_bound),
				row_origin: Extent::GroupX(32),
				column_origin: Extent::GroupY(32),
				transposed: false,
			}
		};
		let mut prefix = vec![
			TileOp::GuardIf { cond: guard },
			TileOp::ExpertScan {
				offsets: if has_bias { 3 } else { 2 },
				experts: Extent::Push(3),
				row: 2,
				acc: scan_first,
			},
			TileOp::ExpertScan {
				offsets: if has_bias { 3 } else { 2 },
				experts: Extent::Push(3),
				row: last_row,
				acc: scan_last,
			},
			TileOp::LaneIf {
				cond: boundary_cond,
				body: vec![boundary, TileOp::Return],
			},
		];
		prefix.append(&mut ops);
		ops = prefix;
		if has_act {
			ops.push(TileOp::StoreTileGated {
				gate_acc: acc,
				up_acc,
				bias_binding: 2,
				linear_binding: 4,
				act_binding: 5,
				linear_stride: Extent::Push(1),
				row_origin: Extent::GroupX(32),
				column_origin: Extent::GroupY(32),
				row_extent: Extent::Push(0),
				gate_width: Extent::HalfPush(1),
				bias_origin: Some(expert_row),
			});
		} else if has_bias {
			ops.push(TileOp::StoreTileBias {
				acc,
				bias_binding: 2,
				binding: 4,
				z: false,
				row_origin: Extent::GroupX(32),
				column_origin: Extent::GroupY(32),
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(1),
				bias_origin: Some(expert_row),
				act: None,
			});
		} else {
			ops.push(TileOp::StoreTile {
				acc,
				binding: 3,
				row_origin: Extent::GroupX(32),
				column_origin: Extent::GroupY(32),
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(1),
				mask_origin: None,
			});
		}
		let mut bindings = vec![Binding::Input(0), Binding::Input(1)];
		if has_bias {
			bindings.push(Binding::Input(2));
		}
		bindings.push(Binding::Input(if has_bias { 3 } else { 2 }));
		bindings.push(Binding::Output(0));
		if has_act {
			bindings.push(Binding::Output(1));
		}
		let mut push_scalars = vec![
			"rows".into(),
			"output_features".into(),
			"input_features".into(),
			"experts".into(),
		];
		if has_act {
			push_scalars.push("act_kind".into());
		}
		Ok(Self {
			bindings,
			push_scalars,
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs,
			ops,
		})
	}

	/// The `grouped_linear_m_data_backward` donor — the transposed
	/// grouped contraction `dX[row][k] = Σ_n dY[row][n] * W[(expert *
	/// outputs + n)][k]` over the same 32x32x16 / 64-lane / 4x4-fragment
	/// schedule as the forward grouped donors. The gradient tile stages
	/// like a forward input; the weight tile transposes in load
	/// (`column * inputs + row`) with the expert term carried on the
	/// load column. Boundary tiles take the transposed serial path.
	/// Push `[rows, outputs, inputs, experts]`; grid `[rows/32,
	/// inputs/32, 1]`.
	pub(in crate::runtime) fn grouped_linear_m_data_backward() -> Result<Self> {
		let (shape, mut tiles, mut ops) = gemm_tile_body_multi((32, 32, 16, 4, 4, 1), (1, 0, 2), &[1])?;
		// tiles: shared gradient 0, shared weight 1, accumulator 2,
		// expert scans last two.
		let acc = 2;
		let scan_first = tiles.len();
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		});
		let scan_last = tiles.len();
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		});
		let mut lane_exprs = vec![
			LaneExpr::GroupX,       // 0
			LaneExpr::ConstU32(32), // 1
			LaneExpr::MulU32(0, 1), // 2: row_base
			LaneExpr::Push(0),      // 3: rows
			LaneExpr::GeU32(2, 3),  // 4: row_base >= rows
			LaneExpr::GroupY,       // 5
			LaneExpr::MulU32(5, 1), // 6: column_base
			LaneExpr::Push(2),      // 7: input_features
		];
		let column_oor = lane_exprs.len();
		lane_exprs.push(LaneExpr::GeU32(6, 7)); // 8
		let guard = lane_exprs.len();
		lane_exprs.push(LaneExpr::OrBool(4, column_oor)); // 9
		let base_plus = lane_exprs.len();
		lane_exprs.push(LaneExpr::AddU32(2, 1)); // 10
		let row_end = lane_exprs.len();
		lane_exprs.push(LaneExpr::MinU32(base_plus, 3)); // 11
		let one = lane_exprs.len();
		lane_exprs.push(LaneExpr::ConstU32(1)); // 12
		let last_row = lane_exprs.len();
		lane_exprs.push(LaneExpr::SubU32(row_end, one)); // 13
		let first_reg = lane_exprs.len();
		lane_exprs.push(LaneExpr::RegisterF32 {
			tile: scan_first,
			index: 0,
		});
		let first = lane_exprs.len();
		lane_exprs.push(LaneExpr::U32FromF32(first_reg));
		let last_reg = lane_exprs.len();
		lane_exprs.push(LaneExpr::RegisterF32 {
			tile: scan_last,
			index: 0,
		});
		let last = lane_exprs.len();
		lane_exprs.push(LaneExpr::U32FromF32(last_reg));
		let boundary_cond = lane_exprs.len();
		lane_exprs.push(LaneExpr::NeqU32(first, last));
		let outputs = lane_exprs.len();
		lane_exprs.push(LaneExpr::Push(1));
		let expert_col = lane_exprs.len();
		lane_exprs.push(LaneExpr::MulU32(first, outputs));
		// The weight tile transposes in load: its staged element
		// `[k-local][reduction]` reads `(first_expert * outputs +
		// reduction) * inputs + k` — the expert term offsets the load
		// column while the mask keeps the raw reduction coordinate.
		let TileOp::Loop { body, .. } = ops.last_mut().unwrap() else {
			return Err(Error::internal("gemm body did not produce the K loop"));
		};
		let TileOp::LoadShared {
			column_offset,
			transposed,
			..
		} = &mut body[1]
		else {
			return Err(Error::internal("gemm body weight staging missing"));
		};
		*column_offset = Some(Extent::Expr(expert_col));
		*transposed = true;
		let mut prefix = vec![
			TileOp::GuardIf { cond: guard },
			TileOp::ExpertScan {
				offsets: 2,
				experts: Extent::Push(3),
				row: 2,
				acc: scan_first,
			},
			TileOp::ExpertScan {
				offsets: 2,
				experts: Extent::Push(3),
				row: last_row,
				acc: scan_last,
			},
			TileOp::LaneIf {
				cond: boundary_cond,
				body: vec![
					TileOp::BoundaryLinear {
						acc,
						offsets: 2,
						input: 0,
						weight: 1,
						bias: None,
						output: 3,
						experts: Extent::Push(3),
						rows: Extent::Push(0),
						outputs: Extent::Push(1),
						inner: Extent::Push(2),
						column_extent: Extent::Push(2),
						row_origin: Extent::GroupX(32),
						column_origin: Extent::GroupY(32),
						transposed: true,
					},
					TileOp::Return,
				],
			},
		];
		prefix.append(&mut ops);
		ops = prefix;
		ops.push(TileOp::StoreTile {
			acc,
			binding: 3,
			row_origin: Extent::GroupX(32),
			column_origin: Extent::GroupY(32),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(2),
			mask_origin: None,
		});
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec![
				"rows".into(),
				"output_features".into(),
				"input_features".into(),
				"experts".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs,
			ops,
		})
	}

	/// `grouped_linear_m_parameter_backward` — the z-dispatched
	/// expert-major adjoint `dW[e][n][k] = Σ_{row ∈ expert e}
	/// `MlLinearBackwardF32` — the donor's 32×32×16 tiled data adjoint
	/// `dX[m,k] = Σ_n dY[m,n] · W[n,k]`, 64 lanes of 4×4 fragments. The
	/// weight is staged transposed (`W` is `[out,in]`, the rhs tile is
	/// `[in][out]`); every boundary is masked.
	pub(in crate::runtime) fn linear_backward() -> Self {
		let (shape, tiles, mut ops) = linear_backward_prelude();
		ops.extend(linear_backward_ops(
			TileOp::LoadShared {
				tile: 0,
				z: false,
				binding: 1,
				row_origin: Extent::GroupX(32),
				column_origin: Extent::Induction,
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(2),
				row_offset: None,
				mask_origin: None,
				column_offset: None,
				transposed: false,
				vec: 1,
			},
			2,
			2,
		));
		Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec![
				String::from("batch"),
				String::from("input_features"),
				String::from("output_features"),
			],
			push_f32_scalars: vec![],
			shape,
			tiles,
			lane_exprs: vec![],
			ops,
		}
	}

	/// `MlLinearBackwardPrologueF32` — the same data adjoint with the
	/// producer's `tile_prologue_value` folded into the upstream staging
	/// and `dy` materialized under the `group_id.y == 0` gate.
	pub(in crate::runtime) fn linear_backward_prologue() -> Self {
		use LaneExpr as E;
		let (shape, tiles, mut ops) = linear_backward_prelude();
		ops.extend(linear_backward_ops(
			TileOp::LoadSharedPrologue {
				tile: 0,
				upstream: 1,
				saved: 2,
				dy: 3,
				row_origin: Extent::GroupX(32),
				column_origin: Extent::Induction,
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(2),
				gate: 1,
			},
			2,
			4,
		));
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				String::from("batch"),
				String::from("input_features"),
				String::from("output_features"),
				String::from("prologue_kind"),
			],
			push_f32_scalars: vec![String::from("alpha")],
			shape,
			tiles,
			lane_exprs: vec![
				/* 0 */ E::ConstU32(0),
				/* 1 */ E::EqU32(2, 0), // group_id.y == 0
				/* 2 */ E::GroupY,
			],
			ops,
		}
	}

	/// `MlLinearBackwardNormF32`/`MlLinearBackwardNormAddF32` — the
	/// layer-norm data adjoint fused into the linear data adjoint. A
	/// row-statistics pre-pass pairs the 64 lanes across the 32-row
	/// band (each lane folds `column = lane & 1; += 2` into
	/// `sum(upstream*gamma)` and `sum(upstream*gamma*normalized)`), a
	/// pairwise combine publishes `shared_row_stats`, then the GEMM
	/// staging computes the adjoint per element and republishes
	/// `norm_input_gradient`, `weight_contribution` (and `stage1` for
	/// the add variant) under `group_id.y == 0`. Tiles are
	/// `[dy, weight, stat_partials, row_stats, acc, stat_x, stat_y,
	/// scaled, inv]`.
	fn linear_backward_norm_program(add: bool) -> Self {
		use LaneExpr as E;
		let (shape, mut tiles, mut ops) = linear_backward_prelude();
		// Insert the two stats scratch tiles before the accumulator.
		tiles.insert(
			2,
			TileDecl::Shared {
				rows: 1,
				columns: 128,
				stride: 128,
			},
		);
		tiles.insert(
			3,
			TileDecl::Shared {
				rows: 1,
				columns: 64,
				stride: 64,
			},
		);
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		}); // 5: stat_x
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		}); // 6: stat_y
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		}); // 7: scaled
		tiles.push(TileDecl::Register {
			rows: 1,
			columns: 1,
		}); // 8: inv
		let lane_exprs = vec![
			/* 0 */ E::Lane,
			/* 1 */ E::Push(0), // batch
			/* 2 */ E::Push(2), // output_features
			/* 3 */ E::ConstU32(0),
			/* 4 */ E::ConstU32(1),
			/* 5 */ E::ConstU32(2),
			/* 6 */ E::ConstU32(32),
			/* 7 */ E::GroupX,
			/* 8 */ E::GroupY,
			/* 9 */ E::MulU32(7, 6), // row_base
			/* 10 */ E::DivU32(0, 5), // stat_lane
			/* 11 */ E::AddU32(9, 10), // stat_row
			/* 12 */ E::MulU32(11, 2), // row_offset
			/* 13 */ E::ModU32(0, 5), // lane & 1 (fold start)
			/* 14 */ E::LtU32(11, 1), // stat_row < batch
			/* 15 */ E::Induction, // column
			/* 16 */ E::AddU32(12, 15), // element address
			/* 17 */ E::LoadF32 {
				binding: 4,
				address: 16,
			}, // upstream
			/* 18 */ E::LoadF32 {
				binding: 3,
				address: 15,
			}, // gamma
			/* 19 */ E::MulF32(17, 18), // scaled
			/* 20 */ E::LoadF32 {
				binding: 1,
				address: 16,
			}, // normalized
			/* 21 */ E::RegisterF32 { tile: 5, index: 0 }, // stat_x
			/* 22 */ E::RegisterF32 { tile: 6, index: 0 }, // stat_y
			/* 23 */ E::RegisterF32 { tile: 7, index: 0 }, // staged scaled
			/* 24 */ E::AddF32(21, 23), // stat_x += scaled
			/* 25 */ E::MulF32(23, 20), // staged scaled * normalized
			/* 26 */ E::AddF32(22, 25), // stat_y += scaled*norm
			/* 27 */ E::MulU32(0, 5), // lane * 2
			/* 28 */ E::AddU32(27, 4), // lane * 2 + 1
			/* 29 */ E::AddU32(27, 5), // lane * 2 + 2
			/* 30 */ E::AddU32(29, 4), // lane * 2 + 3
			/* 31 */ E::EqU32(13, 3), // lane % 2 == 0
			/* 32 */ E::MulU32(10, 5), // stat_lane * 2
			/* 33 */ E::AddU32(32, 4), // stat_lane * 2 + 1
			/* 34 */ E::SharedF32 {
				tile: 2,
				offset: 27,
			}, // partials[thread].x
			/* 35 */ E::SharedF32 {
				tile: 2,
				offset: 29,
			}, // partials[thread + 1].x
			/* 36 */ E::SharedF32 {
				tile: 2,
				offset: 28,
			}, // partials[thread].y
			/* 37 */ E::SharedF32 {
				tile: 2,
				offset: 30,
			}, // partials[thread + 1].y
			/* 38 */ E::AddF32(34, 35), // row_stats.x
			/* 39 */ E::AddF32(36, 37), // row_stats.y
			/* 40 */ E::ConstF32(0x3f80_0000), // 1.0f
			/* 41 */ E::F32FromU32(2), // float(output_features)
			/* 42 */ E::DivF32(40, 41), // inverse_columns
			/* 43 */ E::EqU32(8, 3), // group_id.y == 0
			/* 44 */ E::RegisterF32 { tile: 8, index: 0 }, // inv read
			/* 45 */ E::ConstF32(0x0000_0000), // 0.0f
		];
		ops.extend([
			TileOp::AssignScalar {
				tile: 5,
				index: 0,
				value: 45,
			},
			TileOp::AssignScalar {
				tile: 6,
				index: 0,
				value: 45,
			},
			TileOp::LaneIf {
				cond: 14,
				body: vec![TileOp::LaneLoopRange {
					start: 13,
					step: 2,
					extent: Extent::Push(2),
					body: vec![
						TileOp::AssignScalar {
							tile: 7,
							index: 0,
							value: 19,
						},
						TileOp::AssignScalar {
							tile: 5,
							index: 0,
							value: 24,
						},
						TileOp::AssignScalar {
							tile: 6,
							index: 0,
							value: 26,
						},
					],
				}],
			},
			TileOp::StoreShared {
				tile: 2,
				offset: 27,
				value: 21,
			},
			TileOp::StoreShared {
				tile: 2,
				offset: 28,
				value: 22,
			},
			TileOp::Barrier,
			TileOp::LaneIf {
				cond: 31,
				body: vec![
					TileOp::StoreShared {
						tile: 3,
						offset: 32,
						value: 38,
					},
					TileOp::StoreShared {
						tile: 3,
						offset: 33,
						value: 39,
					},
				],
			},
			TileOp::Barrier,
			TileOp::AssignScalar {
				tile: 8,
				index: 0,
				value: 42,
			},
		]);
		let (norm_in, contrib, stage1, dx, addend) = if add {
			(6usize, 7usize, Some(8usize), 9usize, Some(5usize))
		} else {
			(5usize, 6usize, None, 7usize, None)
		};
		ops.extend(linear_backward_ops(
			TileOp::LoadSharedNorm {
				tile: 0,
				normalized: 1,
				inv_stddev: 2,
				norm_weight: 3,
				upstream: 4,
				addend,
				norm_input_gradient: norm_in,
				weight_contribution: contrib,
				stage1,
				stats: 3,
				inverse_columns: 44,
				row_origin: Extent::GroupX(32),
				column_origin: Extent::Induction,
				row_extent: Extent::Push(0),
				column_extent: Extent::Push(2),
				gate: 43,
			},
			4,
			dx,
		));
		let mut bindings = vec![
			Binding::Input(0),
			Binding::Input(1),
			Binding::Input(2),
			Binding::Input(3),
			Binding::Input(4),
		];
		if add {
			bindings.push(Binding::Input(5));
		}
		bindings.push(Binding::Output(0));
		bindings.push(Binding::Output(1));
		if add {
			bindings.push(Binding::Output(2));
		}
		bindings.push(Binding::Output(if add { 3 } else { 2 }));
		Self {
			bindings,
			push_scalars: vec![
				String::from("batch"),
				String::from("input_features"),
				String::from("output_features"),
			],
			push_f32_scalars: vec![],
			shape,
			tiles,
			lane_exprs,
			ops,
		}
	}

	/// `MlLinearBackwardNormF32` tile route.
	pub(in crate::runtime) fn linear_backward_norm() -> Self {
		Self::linear_backward_norm_program(false)
	}

	/// `MlLinearBackwardNormAddF32` tile route — the norm adjoint fused
	/// with the gradient-accumulation add and a `stage1` republish.
	pub(in crate::runtime) fn linear_backward_norm_add() -> Self {
		Self::linear_backward_norm_program(true)
	}

	/// `MlLinearBackwardBmmChainF32` tile route — the donor's
	/// `linear_backward_bmm` two-stage schedule on 256 lanes
	/// (`lane = ty * 16 + tx`): stage one recomputes the producer bmm
	/// row band into `stage1_tile` and republishes it, stage two
	/// multiplies the band by the staged transposed weight. `kind`
	/// selects the `[B,R,I]` or `[B,I,R]` left-operand layout.
	pub(in crate::runtime) fn linear_backward_bmm_chain() -> Self {
		use LaneExpr as E;
		let tiles = vec![
			TileDecl::Shared {
				rows: 16,
				columns: 16,
				stride: 17,
			},
			TileDecl::Shared {
				rows: 16,
				columns: 32,
				stride: 33,
			},
			TileDecl::Shared {
				rows: 16,
				columns: 32,
				stride: 33,
			},
			TileDecl::Shared {
				rows: 32,
				columns: 32,
				stride: 33,
			},
			TileDecl::Register {
				rows: 1,
				columns: 2,
			},
			TileDecl::Register {
				rows: 1,
				columns: 2,
			},
		];
		let ops = vec![
			TileOp::ClearTile { tile: 4 },
			TileOp::Loop {
				extent: Extent::Push(2),
				step: 16,
				start: None,
				body: vec![
					TileOp::StoreShared {
						tile: 0,
						offset: 36,
						value: 33,
					},
					TileOp::StoreShared {
						tile: 1,
						offset: 39,
						value: 53,
					},
					TileOp::StoreShared {
						tile: 1,
						offset: 40,
						value: 54,
					},
					TileOp::Barrier,
					TileOp::Loop {
						extent: Extent::Expr(1),
						step: 1,
						start: None,
						body: vec![
							TileOp::AssignScalar {
								tile: 4,
								index: 0,
								value: 66,
							},
							TileOp::AssignScalar {
								tile: 4,
								index: 1,
								value: 67,
							},
						],
					},
					TileOp::Barrier,
				],
			},
			TileOp::StoreShared {
				tile: 2,
				offset: 39,
				value: 64,
			},
			TileOp::StoreShared {
				tile: 2,
				offset: 40,
				value: 65,
			},
			TileOp::LaneIf {
				cond: 11,
				body: vec![
					TileOp::LaneIf {
						cond: 43,
						body: vec![TileOp::StoreScalar {
							binding: 3,
							address: 69,
							value: 64,
						}],
					},
					TileOp::LaneIf {
						cond: 45,
						body: vec![TileOp::StoreScalar {
							binding: 3,
							address: 70,
							value: 65,
						}],
					},
				],
			},
			TileOp::LaneLoopRange {
				start: 3,
				step: 256,
				extent: Extent::Expr(71),
				body: vec![TileOp::StoreShared {
					tile: 3,
					offset: 82,
					value: 80,
				}],
			},
			TileOp::Barrier,
			TileOp::ClearTile { tile: 5 },
			TileOp::Loop {
				extent: Extent::Push(3),
				step: 1,
				start: None,
				body: vec![
					TileOp::AssignScalar {
						tile: 5,
						index: 0,
						value: 91,
					},
					TileOp::AssignScalar {
						tile: 5,
						index: 1,
						value: 92,
					},
				],
			},
			TileOp::LaneIf {
				cond: 11,
				body: vec![
					TileOp::LaneIf {
						cond: 96,
						body: vec![TileOp::StoreScalar {
							binding: 4,
							address: 94,
							value: 89,
						}],
					},
					TileOp::LaneIf {
						cond: 97,
						body: vec![TileOp::StoreScalar {
							binding: 4,
							address: 95,
							value: 90,
						}],
					},
				],
			},
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
				Binding::Output(1),
			],
			push_scalars: vec![
				String::from("batch"),
				String::from("rows"),
				String::from("inner"),
				String::from("columns"),
				String::from("input_features"),
				String::from("kind"),
			],
			push_f32_scalars: vec![],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 16,
				fragment_columns: 2,
				tile_rows: 16,
				tile_columns: 32,
			},
			tiles,
			lane_exprs: vec![
				/* 0 */ E::ConstU32(0),
				/* 1 */ E::ConstU32(16),
				/* 2 */ E::ConstU32(32),
				/* 3 */ E::Lane,
				/* 4 */ E::ModU32(3, 1),
				/* 5 */ E::DivU32(3, 1),
				/* 6 */ E::GroupZ,
				/* 7 */ E::GroupY,
				/* 8 */ E::MulU32(7, 1),
				/* 9 */ E::AddU32(8, 5),
				/* 10 */ E::Push(1),
				/* 11 */ E::LtU32(9, 10),
				/* 12 */ E::Push(2),
				/* 13 */ E::Push(3),
				/* 14 */ E::Push(4),
				/* 15 */ E::Push(5),
				/* 16 */ E::AddU32(4, 1),
				/* 17 */ E::Induction,
				/* 18 */ E::AddU32(17, 4),
				/* 19 */ E::LtU32(18, 12),
				/* 20 */ E::AndBool(11, 19),
				/* 21 */ E::MulU32(6, 10),
				/* 22 */ E::AddU32(21, 9),
				/* 23 */ E::MulU32(22, 12),
				/* 24 */ E::AddU32(23, 18),
				/* 25 */ E::MulU32(6, 12),
				/* 26 */ E::AddU32(25, 18),
				/* 27 */ E::MulU32(26, 10),
				/* 28 */ E::AddU32(27, 9),
				/* 29 */ E::EqU32(15, 0),
				/* 30 */ E::SelectU32 {
					cond: 29,
					if_true: 24,
					if_false: 28,
				},
				/* 31 */ E::LoadF32 {
					binding: 0,
					address: 30,
				},
				/* 32 */ E::ConstF32(0),
				/* 33 */ E::SelectF32 {
					cond: 20,
					if_true: 31,
					if_false: 32,
				},
				/* 34 */ E::ConstU32(17),
				/* 35 */ E::MulU32(5, 34),
				/* 36 */ E::AddU32(35, 4),
				/* 37 */ E::ConstU32(33),
				/* 38 */ E::MulU32(5, 37),
				/* 39 */ E::AddU32(38, 4),
				/* 40 */ E::AddU32(39, 1),
				/* 41 */ E::AddU32(17, 5),
				/* 42 */ E::LtU32(41, 12),
				/* 43 */ E::LtU32(4, 13),
				/* 44 */ E::AndBool(42, 43),
				/* 45 */ E::LtU32(16, 13),
				/* 46 */ E::AndBool(42, 45),
				/* 47 */ E::AddU32(25, 41),
				/* 48 */ E::MulU32(47, 13),
				/* 49 */ E::AddU32(48, 4),
				/* 50 */ E::AddU32(48, 16),
				/* 51 */ E::LoadF32 {
					binding: 1,
					address: 49,
				},
				/* 52 */ E::LoadF32 {
					binding: 1,
					address: 50,
				},
				/* 53 */ E::SelectF32 {
					cond: 44,
					if_true: 51,
					if_false: 32,
				},
				/* 54 */ E::SelectF32 {
					cond: 46,
					if_true: 52,
					if_false: 32,
				},
				/* 55 */ E::AddU32(35, 17),
				/* 56 */ E::SharedF32 {
					tile: 0,
					offset: 55,
				},
				/* 57 */ E::MulU32(17, 37),
				/* 58 */ E::AddU32(57, 4),
				/* 59 */ E::SharedF32 {
					tile: 1,
					offset: 58,
				},
				/* 60 */ E::AddU32(58, 1),
				/* 61 */ E::SharedF32 {
					tile: 1,
					offset: 60,
				},
				/* 62 */ E::MulF32(56, 59),
				/* 63 */ E::MulF32(56, 61),
				/* 64 */ E::RegisterF32 { tile: 4, index: 0 },
				/* 65 */ E::RegisterF32 { tile: 4, index: 1 },
				/* 66 */ E::AddF32(64, 62),
				/* 67 */ E::AddF32(65, 63),
				/* 68 */ E::MulU32(22, 13),
				/* 69 */ E::AddU32(68, 4),
				/* 70 */ E::AddU32(68, 16),
				/* 71 */ E::ConstU32(1024),
				/* 72 */ E::DivU32(17, 2),
				/* 73 */ E::ModU32(17, 2),
				/* 74 */ E::LtU32(72, 13),
				/* 75 */ E::LtU32(73, 14),
				/* 76 */ E::AndBool(74, 75),
				/* 77 */ E::MulU32(72, 14),
				/* 78 */ E::AddU32(77, 73),
				/* 79 */ E::LoadF32 {
					binding: 2,
					address: 78,
				},
				/* 80 */ E::SelectF32 {
					cond: 76,
					if_true: 79,
					if_false: 32,
				},
				/* 81 */ E::MulU32(72, 37),
				/* 82 */ E::AddU32(81, 73),
				/* 83 */ E::AddU32(38, 17),
				/* 84 */ E::SharedF32 {
					tile: 2,
					offset: 83,
				},
				/* 85 */ E::SharedF32 {
					tile: 3,
					offset: 58,
				},
				/* 86 */ E::SharedF32 {
					tile: 3,
					offset: 60,
				},
				/* 87 */ E::MulF32(84, 85),
				/* 88 */ E::MulF32(84, 86),
				/* 89 */ E::RegisterF32 { tile: 5, index: 0 },
				/* 90 */ E::RegisterF32 { tile: 5, index: 1 },
				/* 91 */ E::AddF32(89, 87),
				/* 92 */ E::AddF32(90, 88),
				/* 93 */ E::MulU32(22, 14),
				/* 94 */ E::AddU32(93, 4),
				/* 95 */ E::AddU32(93, 16),
				/* 96 */ E::LtU32(4, 14),
				/* 97 */ E::LtU32(16, 14),
			],
			ops,
		}
	}

	/// dY[row][n] * X[row][k]` with `dB[e][n] = Σ dY[row][n]` on the
	/// gated `thread_input == 0 && input_base == 0` lanes.
	pub(in crate::runtime) fn grouped_backward_prologue() -> Result<Self> {
		// The fused producer + data adjoint: one workgroup per 32-row
		// tile walks expert segments (`loop1`), 64-column groups
		// (`loop2`) and 16-wide reduction steps (`loop3`). `dy` is
		// evaluated and materialized inside the gradient staging loop;
		// the weight tile is staged transposed with the running expert
		// origin on the reduction axis.
		let shape = BlockShape {
			lanes: 64,
			thread_columns: 8,
			fragment_rows: 4,
			fragment_columns: 4,
			tile_rows: 32,
			tile_columns: 32,
		};
		let tiles = vec![
			TileDecl::Shared {
				rows: 32,
				columns: 16,
				stride: 17,
			},
			TileDecl::Shared {
				rows: 64,
				columns: 16,
				stride: 17,
			},
			TileDecl::Register {
				rows: 8,
				columns: 4,
			},
			TileDecl::Register {
				rows: 1,
				columns: 1,
			},
			TileDecl::Register {
				rows: 1,
				columns: 1,
			},
		];
		let (gradient, weight, acc, scan_first, scan_last) = (0usize, 1, 2, 3, 4);
		let mut lane_exprs = vec![
			LaneExpr::GroupX,       // 0
			LaneExpr::ConstU32(32), // 1
			LaneExpr::MulU32(0, 1), // 2: row_base
			LaneExpr::Push(0),      // 3: rows
			LaneExpr::GeU32(2, 3),  // 4: row_base >= rows
			LaneExpr::AddU32(2, 1), // 5: row_base + 32
			LaneExpr::MinU32(5, 3), // 6: row_end
			LaneExpr::ConstU32(1),  // 7
			LaneExpr::SubU32(6, 7), // 8: row_end - 1
			LaneExpr::RegisterF32 {
				tile: scan_first,
				index: 0,
			}, // 9
			LaneExpr::U32FromF32(9), // 10: first_expert
			LaneExpr::RegisterF32 {
				tile: scan_last,
				index: 0,
			}, // 11
			LaneExpr::U32FromF32(11), // 12: last_expert
			LaneExpr::AddU32(12, 7), // 13: last_expert + 1
			LaneExpr::LoopVar(2),   // 14: expert
			LaneExpr::LoopVar(3),   // 15: column_base
			LaneExpr::LoopVar(4),   // 16: reduction_base
			LaneExpr::Push(1),      // 17: output_features
			LaneExpr::MulU32(14, 17), // 18: expert * outputs
			LaneExpr::ConstU32(0),  // 19
			LaneExpr::EqU32(14, 10), // 20: expert == first
			LaneExpr::EqU32(15, 19), // 21: column_base == 0
			LaneExpr::AndBool(20, 21), // 22: dy store gate
			LaneExpr::AddU32(14, 7), // 23: expert + 1
		];
		// The offsets table is a `[experts + 1]` boundary array with
		// `offsets[0] == 0` and `offsets[experts] == rows`, so the donor's
		// `expert == 0` / `expert + 1 < experts` branches fold into
		// `max`/`min` over the same loads.
		let segment_lo = lane_exprs.len();
		lane_exprs.push(LaneExpr::MaxU32(
			2,
			lane_exprs.len() + 1, // BufferU32 pushed next
		));
		lane_exprs.push(LaneExpr::BufferU32 {
			binding: 4,
			index: 14,
		});
		let segment_hi = lane_exprs.len();
		lane_exprs.push(LaneExpr::MinU32(6, lane_exprs.len() + 1));
		lane_exprs.push(LaneExpr::BufferU32 {
			binding: 4,
			index: 23,
		});
		let ops = vec![
			TileOp::GuardIf { cond: 4 },
			TileOp::ExpertScan {
				offsets: 4,
				experts: Extent::Push(3),
				row: 2,
				acc: scan_first,
			},
			TileOp::ExpertScan {
				offsets: 4,
				experts: Extent::Push(3),
				row: 8,
				acc: scan_last,
			},
			TileOp::Loop {
				extent: Extent::Expr(13),
				step: 1,
				start: Some(Extent::Expr(10)),
				body: vec![TileOp::Loop {
					extent: Extent::Push(2),
					step: 64,
					start: None,
					body: vec![
						TileOp::ClearTile { tile: acc },
						TileOp::Loop {
							extent: Extent::Push(1),
							step: 16,
							start: None,
							body: vec![
								TileOp::LoadSharedPrologue {
									tile: gradient,
									upstream: 0,
									saved: 1,
									dy: 2,
									row_origin: Extent::Expr(2),
									column_origin: Extent::Expr(16),
									row_extent: Extent::Push(0),
									column_extent: Extent::Push(1),
									gate: 22,
								},
								TileOp::LoadShared {
									tile: weight,
									z: false,
									binding: 3,
									row_origin: Extent::Expr(15),
									column_origin: Extent::Expr(16),
									row_extent: Extent::Push(2),
									column_extent: Extent::Push(1),
									row_offset: None,
									mask_origin: None,
									column_offset: Some(Extent::Expr(18)),
									transposed: true,
									vec: 1,
								},
								TileOp::Barrier,
								TileOp::DotSharedWide {
									acc,
									lhs: gradient,
									rhs: weight,
									column_tiles: 2,
								},
								TileOp::Barrier,
							],
						},
						TileOp::StoreTileSegment {
							acc,
							binding: 5,
							row_origin: Extent::Expr(2),
							row_lo: Extent::Expr(segment_lo),
							row_hi: Extent::Expr(segment_hi),
							column_origin: Extent::Expr(15),
							column_extent: Extent::Push(2),
							column_tiles: 2,
						},
					],
				}],
			},
		];
		Ok(Self {
			// [upstream, saved, dy, weight, offsets, input_gradient]
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Output(0),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(1),
			],
			push_scalars: vec![
				String::from("rows"),
				String::from("output_features"),
				String::from("input_features"),
				String::from("experts"),
				String::from("prologue_kind"),
			],
			push_f32_scalars: vec![String::from("alpha")],
			shape,
			tiles,
			lane_exprs,
			ops,
		})
	}

	pub(in crate::runtime) fn grouped_linear_m_parameter_backward() -> Result<Self> {
		Self::grouped_parameter_body(true)
	}

	/// `grouped_gemm_m_parameter_backward` — the same schedule with no
	/// bias accumulator or store.
	pub(in crate::runtime) fn grouped_gemm_m_parameter_backward() -> Result<Self> {
		Self::grouped_parameter_body(false)
	}

	/// Shared grouped parameter-adjoint body — 32x32x16 tiles, 64
	/// lanes, 4x4 fragments, `expert = group_id.z`. The reduction range
	/// `[begin, end)` comes from the offsets buffer, so both shared
	/// tiles transpose in load (feature-major shared layout) and the
	/// weight store carries the expert offset on its row coordinate
	/// while masking the bare feature. `has_bias` adds the gated
	/// `dB` accumulator and its scalar-column store.
	fn grouped_parameter_body(has_bias: bool) -> Result<Self> {
		let shape = BlockShape {
			lanes: 64,
			thread_columns: 8,
			fragment_rows: 4,
			fragment_columns: 4,
			tile_rows: 32,
			tile_columns: 32,
		};
		let tiles = vec![
			TileDecl::Shared {
				rows: 32,
				columns: 16,
				stride: 17,
			},
			TileDecl::Shared {
				rows: 32,
				columns: 16,
				stride: 17,
			},
			TileDecl::Register {
				rows: 4,
				columns: 4,
			},
			TileDecl::Register {
				rows: 4,
				columns: 1,
			},
			TileDecl::Register {
				rows: 1,
				columns: 1,
			},
			TileDecl::Register {
				rows: 1,
				columns: 1,
			},
		];
		let lane_exprs = vec![
			LaneExpr::GroupZ,         // 0: expert
			LaneExpr::Push(3),        // 1: experts
			LaneExpr::GeU32(0, 1),    // 2: expert >= experts
			LaneExpr::GroupX,         // 3
			LaneExpr::ConstU32(32),   // 4
			LaneExpr::MulU32(3, 4),   // 5: output_base
			LaneExpr::Push(1),        // 6: outputs
			LaneExpr::GeU32(5, 6),    // 7
			LaneExpr::GroupY,         // 8
			LaneExpr::MulU32(8, 4),   // 9: input_base
			LaneExpr::Push(2),        // 10: inputs
			LaneExpr::GeU32(9, 10),   // 11
			LaneExpr::OrBool(2, 7),   // 12
			LaneExpr::OrBool(12, 11), // 13: guard
			LaneExpr::BufferU32 {
				binding: 2,
				index: 0,
			}, // 14: begin
			LaneExpr::ConstU32(1),    // 15
			LaneExpr::AddU32(0, 15),  // 16: expert + 1
			LaneExpr::BufferU32 {
				binding: 2,
				index: 16,
			}, // 17: end
			LaneExpr::F32FromU32(14), // 18
			LaneExpr::F32FromU32(17), // 19
			LaneExpr::Lane,           // 20
			LaneExpr::ConstU32(8),    // 21
			LaneExpr::ModU32(20, 21), // 22: thread_input
			LaneExpr::ConstU32(0),    // 23
			LaneExpr::EqU32(22, 23),  // 24: thread_input == 0
			LaneExpr::EqU32(8, 23),   // 25: input_base == 0
			LaneExpr::AndBool(24, 25), // 26: bias gate
			LaneExpr::MulU32(0, 6),   // 27: expert * outputs
			LaneExpr::AddU32(27, 5),  // 28: expert row + output_base
			LaneExpr::RegisterF32 { tile: 4, index: 0 }, // 29
			LaneExpr::U32FromF32(29), // 30: begin (hoisted)
			LaneExpr::RegisterF32 { tile: 5, index: 0 }, // 31
			LaneExpr::U32FromF32(31), // 32: end (hoisted)
		];
		let mut ops = vec![
			TileOp::GuardIf { cond: 13 },
			TileOp::AssignScalar {
				tile: 4,
				index: 0,
				value: 18,
			},
			TileOp::AssignScalar {
				tile: 5,
				index: 0,
				value: 19,
			},
			TileOp::ClearTile { tile: 2 },
		];
		if has_bias {
			ops.push(TileOp::ClearTile { tile: 3 });
		}
		let dot = if has_bias {
			TileOp::DotSharedParam {
				acc: 2,
				lhs: 0,
				rhs: 1,
				bias_acc: 3,
				bias_gate: 26,
			}
		} else {
			TileOp::DotShared {
				acc: 2,
				lhs: 0,
				rhs: 1,
			}
		};
		ops.push(TileOp::Loop {
			extent: Extent::Expr(32),
			step: 16,
			start: Some(Extent::Expr(30)),
			body: vec![
				// shared_gradient[feature][row] — transpose-in-load of
				// `dY[row * outputs + feature]`, masked `row < end`.
				TileOp::LoadShared {
					tile: 0,
					binding: 0,
					row_origin: Extent::GroupX(32),
					column_origin: Extent::Induction,
					row_extent: Extent::Push(1),
					column_extent: Extent::Expr(32),
					vec: 1,
					row_offset: None,
					mask_origin: None,
					z: false,
					column_offset: None,
					transposed: true,
				},
				// shared_input[feature][row] — transpose-in-load of
				// `X[row * inputs + feature]`.
				TileOp::LoadShared {
					tile: 1,
					binding: 1,
					row_origin: Extent::GroupY(32),
					column_origin: Extent::Induction,
					row_extent: Extent::Push(2),
					column_extent: Extent::Expr(32),
					vec: 1,
					row_offset: None,
					mask_origin: None,
					z: false,
					column_offset: None,
					transposed: true,
				},
				TileOp::Barrier,
				dot,
				TileOp::Barrier,
			],
		});
		// `dW[(expert * outputs + feature) * inputs + k]` — the store
		// coordinate carries the expert row while the mask keeps the
		// bare output feature.
		ops.push(TileOp::StoreTile {
			acc: 2,
			binding: 3,
			row_origin: Extent::Expr(28),
			column_origin: Extent::GroupY(32),
			row_extent: Extent::Push(1),
			column_extent: Extent::Push(2),
			mask_origin: Some(Extent::GroupX(32)),
		});
		if has_bias {
			ops.push(TileOp::StoreTileBiasParam {
				acc: 3,
				binding: 4,
				origin: Extent::Expr(27),
				row_origin: Extent::GroupX(32),
				row_extent: Extent::Push(1),
				gate: 26,
			});
		}
		let mut bindings = vec![
			Binding::Input(0),
			Binding::Input(1),
			Binding::Input(2),
			Binding::Output(0),
		];
		if has_bias {
			bindings.push(Binding::Output(1));
		}
		Ok(Self {
			bindings,
			push_scalars: vec![
				"rows".into(),
				"output_features".into(),
				"input_features".into(),
				"experts".into(),
			],
			push_f32_scalars: Vec::new(),
			shape,
			tiles,
			lane_exprs,
			ops,
		})
	}

	/// The `mat_mul_nt_gemv` donor re-expression: one workgroup per output
	/// element (`group_id.x` row, `group_id.y` column), every lane strides
	/// the K extent into a private accumulator, and a shared-memory tree
	/// sums the lanes for lane 0's scalar store. `lanes` is the workgroup
	/// width — the donor ships 256.
	pub(in crate::runtime) fn mat_mul_nt_gemv(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"gemv lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX, // row
			/* 1 */ E::GroupY, // column
			/* 2 */ E::Push(0), // m
			/* 3 */ E::Push(1), // n
			/* 4 */ E::Push(2), // k
			/* 5 */ E::LtU32(0, 2), // row < m
			/* 6 */ E::LtU32(1, 3), // column < n
			/* 7 */ E::AndBool(5, 6), // in-bounds
			/* 8 */ E::Induction, // inner
			/* 9 */ E::MulU32(0, 4), // row * k
			/* 10 */ E::AddU32(9, 8), // left element address
			/* 11 */ E::LoadF32 {
				binding: 0,
				address: 10,
			},
			/* 12 */ E::MulU32(1, 4), // column * k
			/* 13 */ E::AddU32(12, 8), // right element address
			/* 14 */ E::LoadF32 {
				binding: 1,
				address: 13,
			},
			/* 15 */ E::MulF32(11, 14),
			/* 16 */ E::ConstF32(0), // 0.0f
			/* 17 */ E::RegisterF32 { tile: 1, index: 0 }, // accumulator
			/* 18 */ E::Lane,
			/* 19 */ E::ConstU32(0),
			/* 20 */ E::SharedF32 {
				tile: 0,
				offset: 19,
			}, // partials[0]
			/* 21 */ E::MulU32(0, 3), // row * n
			/* 22 */ E::AddU32(21, 1), // output element address
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec!["m".into(), "n".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				// Donor ordering: the accumulator is declared before the
				// bounds guard so lane-out-of-range rows still stage a
				// zero for the tree.
				TileOp::AssignScalar {
					tile: 1,
					index: 0,
					value: 16,
				},
				TileOp::LaneIf {
					cond: 7,
					body: vec![TileOp::LaneFold {
						acc: 1,
						init: 16,
						extent: Extent::Push(2),
						op: LaneReduce::Sum,
						element: 15,
						serial: false,
					}],
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 18,
					value: 17,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::LaneGuard {
					lane: 19,
					body: vec![TileOp::LaneIf {
						cond: 7,
						body: vec![TileOp::StoreScalar {
							binding: 2,
							address: 22,
							value: 20,
						}],
					}],
				},
			],
		})
	}
}

#[cfg(test)]
#[path = "../../../../../../test/rs/runtime/ir_tile_unit.rs"]
mod tests;

/// The shared cooperative staging/accumulate body of the GEMM-family
/// templates: clear the fragment accumulator, loop over the K extent in
/// `tile_k` steps staging both operands through padded shared tiles,
/// barrier-separated per-lane FMA. `k_push`, `row_push` and
/// `column_push` are the push-scalar indices holding the K, M and N
/// extents for this template's ABI.
fn gemm_tile_body(
	geometry: (u32, u32, u32, u32, u32, u32),
	pushes: (usize, usize, usize),
) -> Result<(BlockShape, Vec<TileDecl>, Vec<TileOp>)> {
	gemm_tile_body_multi(geometry, pushes, &[1])
}

/// Shared GEMM staging/accumulation for one input tile and `weights`
/// weight tiles — tiles are declared `[shared input, shared w0..,
/// register acc0..]` and each weight pair gets its own `DotShared`.
/// `qkv_projection_bias` (three weights) and `gate_up_swiglu_bias` (two)
/// build on this.
fn gemm_tile_body_multi(
	geometry: (u32, u32, u32, u32, u32, u32),
	pushes: (usize, usize, usize),
	weights: &[usize],
) -> Result<(BlockShape, Vec<TileDecl>, Vec<TileOp>)> {
	let (tile_rows, tile_columns, tile_k, fragment_rows, fragment_columns, vec) = geometry;
	let (k_push, row_push, column_push) = pushes;
	if weights.is_empty() {
		return Err(Error::invalid_argument(
			"multi-weight GEMM body needs at least one weight binding",
		));
	}
	if vec == 0 || vec > 4 || !tile_k.is_multiple_of(vec) {
		return Err(Error::invalid_argument(
			"tile_k must be a positive multiple of the staging width (≤4)",
		));
	}
	if !tile_rows.is_multiple_of(fragment_rows) || !tile_columns.is_multiple_of(fragment_columns) {
		return Err(Error::invalid_argument(
			"output tile must be an exact multiple of the fragment tile",
		));
	}
	let thread_columns = tile_columns / fragment_columns;
	let lanes = (tile_rows / fragment_rows) * thread_columns;
	if lanes > 1024 {
		return Err(Error::invalid_argument(
			"fragment grid exceeds the Vulkan workgroup floor",
		));
	}
	let shape = BlockShape {
		lanes,
		thread_columns,
		fragment_rows,
		fragment_columns,
		tile_rows,
		tile_columns,
	};
	let count = weights.len();
	let mut tiles = vec![TileDecl::Shared {
		rows: tile_rows,
		columns: tile_k,
		stride: tile_k + 1,
	}];
	for _ in weights {
		tiles.push(TileDecl::Shared {
			rows: tile_columns,
			columns: tile_k,
			stride: tile_k + 1,
		});
	}
	for _ in weights {
		tiles.push(TileDecl::Register {
			rows: fragment_rows,
			columns: fragment_columns,
		});
	}
	let mut ops = Vec::new();
	for acc in 0..count {
		ops.push(TileOp::ClearTile {
			tile: count + 1 + acc,
		});
	}
	let mut body = vec![TileOp::LoadShared {
		tile: 0,
		binding: 0,
		row_origin: Extent::GroupX(tile_rows),
		column_origin: Extent::Induction,
		row_extent: Extent::Push(row_push),
		column_extent: Extent::Push(k_push),
		vec,
		row_offset: None,
		mask_origin: None,
		z: false,
		column_offset: None,
		transposed: false,
	}];
	for (index, binding) in weights.iter().enumerate() {
		body.push(TileOp::LoadShared {
			tile: 1 + index,
			binding: *binding,
			row_origin: Extent::GroupY(tile_columns),
			column_origin: Extent::Induction,
			row_extent: Extent::Push(column_push),
			column_extent: Extent::Push(k_push),
			vec,
			row_offset: None,
			mask_origin: None,
			z: false,
			column_offset: None,
			transposed: false,
		});
	}
	body.push(TileOp::Barrier);
	for index in 0..count {
		body.push(TileOp::DotShared {
			acc: count + 1 + index,
			lhs: 0,
			rhs: 1 + index,
		});
	}
	body.push(TileOp::Barrier);
	ops.push(TileOp::Loop {
		extent: Extent::Push(k_push),
		step: tile_k,
		start: None,
		body,
	});
	Ok((shape, tiles, ops))
}

/// Shared `MlLinearBackward*` prelude — the donor's 32×32×16 tiled
/// geometry: 64 lanes of 4×4 fragments over a 32×32 output tile.
/// `tiles` is `[dy, weight, acc]`; the norm variants insert stats
/// scratch tiles before `acc`.
fn linear_backward_prelude() -> (BlockShape, Vec<TileDecl>, Vec<TileOp>) {
	(
		BlockShape {
			lanes: 64,
			thread_columns: 8,
			fragment_rows: 4,
			fragment_columns: 4,
			tile_rows: 32,
			tile_columns: 32,
		},
		vec![
			TileDecl::Shared {
				rows: 32,
				columns: 16,
				stride: 17,
			},
			TileDecl::Shared {
				rows: 32,
				columns: 16,
				stride: 17,
			},
			TileDecl::Register {
				rows: 4,
				columns: 4,
			},
		],
		Vec::new(),
	)
}

/// The reduction loop and epilogue shared by the `MlLinearBackward*`
/// family: `lhs` stages the upstream/`dy` tile (plain load, fused
/// prologue, or fused norm adjoint), the weight is staged transposed
/// out of binding 0, and `acc` stores to `dx_binding`.
fn linear_backward_ops(lhs: TileOp, acc: usize, dx_binding: usize) -> Vec<TileOp> {
	vec![
		TileOp::ClearTile { tile: acc },
		TileOp::Loop {
			extent: Extent::Push(2),
			step: 16,
			start: None,
			body: vec![
				lhs,
				TileOp::LoadShared {
					tile: 1,
					z: false,
					binding: 0,
					row_origin: Extent::GroupY(32),
					column_origin: Extent::Induction,
					row_extent: Extent::Push(1),
					column_extent: Extent::Push(2),
					row_offset: None,
					mask_origin: None,
					column_offset: None,
					transposed: true,
					vec: 1,
				},
				TileOp::Barrier,
				TileOp::DotShared {
					acc,
					lhs: 0,
					rhs: 1,
				},
				TileOp::Barrier,
			],
		},
		TileOp::StoreTile {
			acc,
			binding: dx_binding,
			row_origin: Extent::GroupX(32),
			column_origin: Extent::GroupY(32),
			row_extent: Extent::Push(0),
			column_extent: Extent::Push(1),
			mask_origin: None,
		},
	]
}

/// `gemm_tile_body` variant whose operand-`B` shared load selects its
/// descriptor by `group_id.z` — `linear_triple` keeps the input staged
/// once while the weight tile comes from `buffer1 + z`.
fn gemm_tile_body_z(
	geometry: (u32, u32, u32, u32, u32, u32),
	pushes: (usize, usize, usize),
) -> Result<(BlockShape, Vec<TileDecl>, Vec<TileOp>)> {
	let (shape, tiles, mut ops) = gemm_tile_body(geometry, pushes)?;
	if let TileOp::Loop { body, .. } = &mut ops[1]
		&& let TileOp::LoadShared { z, .. } = &mut body[1]
	{
		*z = true;
	}
	Ok((shape, tiles, ops))
}

impl BlockProgram {
	/// `moe_router_select` — the donor's cooperative 256-lane
	/// row-per-workgroup schedule: lane-strided max and sum-of-exp folds
	/// into shared tree reductions (softmax over `logits`), a strided
	/// probabilities materialization, then a serial top-K pass whose
	/// per-rank lane scan skips `taken[]` entries and whose paired
	/// value+index tree reduction keeps the donor's
	/// `(v2 > v1) || (v2 == v1 && i2 < i1)` tie-break, and a final
	/// strided `mask` pass. `[logits, selection, probabilities, values,
	/// indices, mask]` + `[rows, experts, k]` ABI over a `[rows,1,1]`
	/// grid.
	pub(in crate::runtime) fn moe_router_select() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // experts
			/* 2 */ E::Push(2), // k
			/* 3 */ E::GroupX, // row
			/* 4 */ E::MulU32(3, 1), // base = row * experts
			/* 5 */ E::Lane,
			/* 6 */ E::Induction, // strided index / inner serial var
			/* 7 */ E::AddU32(4, 6), // element = base + index
			/* 8 */ E::LoadF32 {
				binding: 0,
				address: 7,
			}, // logit
			/* 9 */ E::ConstF32(0xff7fffff), // -FLT_MAX
			/* 10 */ E::RegisterF32 { tile: 3, index: 0 }, // local_maximum
			/* 11 */ E::ConstU32(0),
			/* 12 */ E::SharedF32 {
				tile: 0,
				offset: 11,
			}, // shared_values[0]
			/* 13 */ E::RegisterF32 { tile: 4, index: 0 }, // row_maximum
			/* 14 */ E::SubF32(8, 13), // logit - row_maximum
			/* 15 */ E::ExpF32(14),
			/* 16 */ E::ConstF32(0), // 0.0f
			/* 17 */ E::RegisterF32 { tile: 5, index: 0 }, // local_sum
			/* 18 */ E::ConstF32(0x3f800000), // 1.0f
			/* 19 */ E::DivF32(18, 12), // 1 / shared[0]
			/* 20 */ E::RegisterF32 { tile: 6, index: 0 }, // inverse_sum
			/* 21 */ E::MulF32(15, 20), // probability
			/* 22 */ E::LoopVar(2), // rank (top-K) / index (mask pass)
			/* 23 */ E::LoopVar(3), // index (top-K) / rank (mask pass)
			/* 24 */ E::LoopVar(4), // previous
			/* 25 */ E::SharedU32 {
				tile: 2,
				offset: 24,
			}, // taken[previous]
			/* 26 */ E::EqU32(25, 23),
			/* 27 */ E::ConstU32(1),
			/* 28 */ E::SelectU32 {
				cond: 26,
				if_true: 27,
				if_false: 11,
			},
			/* 29 */ E::RegisterU32 { tile: 9, index: 0 }, // is_taken / selected
			/* 30 */ E::AddU32(29, 28),
			/* 31 */ E::EqU32(29, 11), // is_taken == 0
			/* 32 */ E::AddU32(4, 23), // element (top-K)
			/* 33 */ E::LoadF32 {
				binding: 1,
				address: 32,
			}, // selection value
			/* 34 */ E::RegisterF32 { tile: 7, index: 0 }, // best_value
			/* 35 */ E::GtF32(33, 34),
			/* 36 */ E::EqF32(33, 34),
			/* 37 */ E::RegisterU32 { tile: 8, index: 0 }, // best_index
			/* 38 */ E::LtU32(23, 37), // index < best_index
			/* 39 */ E::AndBool(36, 38),
			/* 40 */ E::OrBool(35, 39), // better pair
			/* 41 */ E::AndBool(31, 40), // !is_taken && better
			/* 42 */ E::MulU32(3, 2), // row * k
			/* 43 */ E::AddU32(42, 22), // row*k + rank
			/* 44 */ E::SharedU32 {
				tile: 1,
				offset: 11,
			}, // shared_indices[0]
			/* 45 */ E::SharedU32 {
				tile: 2,
				offset: 23,
			}, // taken[rank] (mask)
			/* 46 */ E::EqU32(45, 22), // taken[rank] == index
			/* 47 */ E::SelectU32 {
				cond: 46,
				if_true: 27,
				if_false: 11,
			},
			/* 48 */ E::AddU32(29, 47), // selected count
			/* 49 */ E::AddU32(4, 22), // mask element
			/* 50 */ E::MinU32(29, 27), // min(selected, 1)
			/* 51 */ E::F32FromU32(50), // float(selected)
			/* 52 */ E::ConstU32(u32::MAX), // 0xffffffff
		];
		Self {
			bindings: vec![
				Binding::Input(0),  // logits
				Binding::Input(1),  // selection
				Binding::Output(0), // probabilities
				Binding::Output(1), // values
				Binding::Output(2), // indices
				Binding::Output(3), // mask
			],
			push_scalars: vec!["rows".into(), "experts".into(), "k".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: 256,
					stride: 256,
				}, // shared_values
				TileDecl::SharedU32 { elements: 256 },  // shared_indices
				TileDecl::SharedU32 { elements: 1024 }, // taken
				TileDecl::Register {
					rows: 1,
					columns: 1,
				}, // local_maximum
				TileDecl::Register {
					rows: 1,
					columns: 1,
				}, // row_maximum
				TileDecl::Register {
					rows: 1,
					columns: 1,
				}, // local_sum
				TileDecl::Register {
					rows: 1,
					columns: 1,
				}, // inverse_sum
				TileDecl::Register {
					rows: 1,
					columns: 1,
				}, // best_value
				TileDecl::RegisterU32 { elements: 1 },  // best_index
				TileDecl::RegisterU32 { elements: 1 },  // is_taken / selected
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				// Declare `tile9` at function scope so the mask pass can
				// reuse the is_taken/selected register outside the top-K
				// loop's lexical block.
				TileOp::AssignScalarU32 {
					tile: 9,
					index: 0,
					value: 11,
				},
				TileOp::LaneFold {
					acc: 3,
					init: 9,
					extent: Extent::Push(1),
					op: LaneReduce::Max,
					element: 8,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 5,
					value: 10,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Max,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 12,
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 5,
					init: 16,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 15,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 5,
					value: 17,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 6,
					index: 0,
					value: 19,
				},
				TileOp::Barrier,
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 7,
						value: 21,
					}],
				},
				TileOp::Barrier,
				TileOp::Loop {
					extent: Extent::Push(2),
					step: 1,
					start: None,
					body: vec![
						TileOp::AssignScalar {
							tile: 7,
							index: 0,
							value: 9,
						},
						TileOp::AssignScalarU32 {
							tile: 8,
							index: 0,
							value: 52,
						},
						TileOp::LaneLoop {
							extent: Extent::Push(1),
							body: vec![
								TileOp::AssignScalarU32 {
									tile: 9,
									index: 0,
									value: 11,
								},
								TileOp::Loop {
									extent: Extent::Expr(22),
									step: 1,
									start: None,
									body: vec![TileOp::AssignScalarU32 {
										tile: 9,
										index: 0,
										value: 30,
									}],
								},
								TileOp::LaneIf {
									cond: 41,
									body: vec![
										TileOp::AssignScalar {
											tile: 7,
											index: 0,
											value: 33,
										},
										TileOp::AssignScalarU32 {
											tile: 8,
											index: 0,
											value: 23,
										},
									],
								},
							],
						},
						TileOp::StoreShared {
							tile: 0,
							offset: 5,
							value: 34,
						},
						TileOp::StoreSharedU32 {
							tile: 1,
							offset: 5,
							value: 37,
						},
						TileOp::Barrier,
						TileOp::TreeReduceArgmax {
							value_tile: 0,
							index_tile: 1,
						},
						TileOp::LaneGuard {
							lane: 11,
							body: vec![
								TileOp::StoreSharedU32 {
									tile: 2,
									offset: 22,
									value: 44,
								},
								TileOp::StoreScalar {
									binding: 3,
									address: 43,
									value: 12,
								},
								TileOp::StoreScalarU32 {
									binding: 4,
									address: 43,
									value: 44,
								},
							],
						},
						TileOp::Barrier,
					],
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 9,
							index: 0,
							value: 11,
						},
						TileOp::Loop {
							extent: Extent::Push(2),
							step: 1,
							start: None,
							body: vec![TileOp::AssignScalarU32 {
								tile: 9,
								index: 0,
								value: 48,
							}],
						},
						TileOp::StoreScalar {
							binding: 5,
							address: 49,
							value: 51,
						},
					],
				},
			],
		}
	}

	/// `moe_gather` — `output[i] = input[token[i / row_width] *
	/// row_width + i % row_width]` over `routes * row_width` flat
	/// elements. `[input, packed_token, output]` + `[routes, row_width]`
	/// ABI; `token` is a per-lane u32 table load.
	pub(in crate::runtime) fn moe_gather() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // flat index
			/* 5 */ E::Push(0), // routes
			/* 6 */ E::Push(1), // row_width
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7), // out-of-bounds
			/* 9 */ E::DivU32(4, 6), // packed_row
			/* 10 */ E::ModU32(4, 6), // column
			/* 11 */ E::BufferU32 {
				binding: 1,
				index: 9,
			}, // token
			/* 12 */ E::MulU32(11, 6), // token * row_width
			/* 13 */ E::AddU32(12, 10), // input element
			/* 14 */ E::LoadF32 {
				binding: 0,
				address: 13,
			},
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec!["routes".into(), "row_width".into()],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 14,
				},
			],
		)
	}

	/// `moe_combine` — `output[t, c] = Σ_slot inverse[t*K+slot] < routes
	/// ? packed[inverse[..] * row_width + c] * gate[t*K+slot] : 0` over
	/// `tokens * row_width` flat elements. `[packed, route_gate,
	/// inverse, output]` + `[tokens, routes_per_token, row_width,
	/// routes]` ABI.
	pub(in crate::runtime) fn moe_combine() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // flat index
			/* 5 */ E::Push(0), // tokens
			/* 6 */ E::Push(2), // row_width
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7),
			/* 9 */ E::DivU32(4, 6), // token
			/* 10 */ E::ModU32(4, 6), // column
			/* 11 */ E::Push(1), // routes_per_token
			/* 12 */ E::MulU32(9, 11), // token * K
			/* 13 */ E::Induction, // route_slot
			/* 14 */ E::AddU32(12, 13), // slot
			/* 15 */ E::BufferU32 {
				binding: 2,
				index: 14,
			}, // packed
			/* 16 */ E::Push(3), // routes
			/* 17 */ E::LtU32(15, 16), // in-range
			/* 18 */ E::MulU32(15, 6), // packed * row_width
			/* 19 */ E::AddU32(18, 10), // packed element
			/* 20 */ E::LoadF32 {
				binding: 0,
				address: 19,
			},
			/* 21 */ E::LoadF32 {
				binding: 1,
				address: 14,
			}, // gate
			/* 22 */ E::MulF32(20, 21),
			/* 23 */ E::ConstF32(0), // 0.0f
			/* 24 */ E::RegisterF32 { tile: 0, index: 0 }, // sum
			/* 25 */ E::AddF32(24, 22), // sum + product
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			vec![
				"tokens".into(),
				"routes_per_token".into(),
				"row_width".into(),
				"routes".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 23,
				},
				TileOp::Loop {
					extent: Extent::Push(1),
					step: 1,
					start: None,
					body: vec![TileOp::LaneIf {
						cond: 17,
						body: vec![TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: 25,
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 3,
					address: 4,
					value: 24,
				},
			],
		)
	}

	/// `moe_gather_backward` — `output[i] = Σ_slot inverse[token*K+slot]
	/// < routes ? source[inverse[..] * row_width + column] : 0` over
	/// `tokens * row_width`. `[source, inverse, output]` + `[routes,
	/// row_width, tokens, routes_per_token]` ABI.
	pub(in crate::runtime) fn moe_gather_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3),
			/* 5 */ E::Push(2), // tokens
			/* 6 */ E::Push(1), // row_width
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7),
			/* 9 */ E::DivU32(4, 6), // token
			/* 10 */ E::ModU32(4, 6), // column
			/* 11 */ E::Push(3), // routes_per_token
			/* 12 */ E::MulU32(9, 11), // token * K
			/* 13 */ E::Induction,
			/* 14 */ E::AddU32(12, 13), // slot
			/* 15 */ E::BufferU32 {
				binding: 1,
				index: 14,
			}, // packed
			/* 16 */ E::Push(0), // routes
			/* 17 */ E::LtU32(15, 16),
			/* 18 */ E::MulU32(15, 6),
			/* 19 */ E::AddU32(18, 10),
			/* 20 */ E::LoadF32 {
				binding: 0,
				address: 19,
			},
			/* 21 */ E::ConstF32(0),
			/* 22 */ E::RegisterF32 { tile: 0, index: 0 },
			/* 23 */ E::AddF32(22, 20),
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"routes".into(),
				"row_width".into(),
				"tokens".into(),
				"routes_per_token".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 21,
				},
				TileOp::Loop {
					extent: Extent::Push(3),
					step: 1,
					start: None,
					body: vec![TileOp::LaneIf {
						cond: 17,
						body: vec![TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: 23,
						}],
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 22,
				},
			],
		)
	}

	/// `moe_combine_backward` — two flat programs in one dispatch: for
	/// `index < routes * row_width`, `packed_grad[index] =
	/// output_grad[token * row_width + column] * gate[slot]` with
	/// `slot = packed_slot[packed_row]`, `token = slot / K`; for `index <
	/// routes`, `gate_grad[index] = Σ_c output_grad[token * row_width +
	/// c] * packed[packed_row * row_width + c]` where `token =
	/// index / K`, `packed_row = inverse[index]`. Seven bindings +
	/// `[tokens, routes_per_token, row_width, routes]` ABI.
	pub(in crate::runtime) fn moe_combine_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // flat index
			/* 5 */ E::Push(3), // routes
			/* 6 */ E::Push(2), // row_width
			/* 7 */ E::MulU32(5, 6), // routes * row_width
			/* 8 */ E::LtU32(4, 7), // region one live
			/* 9 */ E::DivU32(4, 6), // packed_row
			/* 10 */ E::ModU32(4, 6), // column
			/* 11 */ E::BufferU32 {
				binding: 4,
				index: 9,
			}, // slot
			/* 12 */ E::Push(1), // routes_per_token
			/* 13 */ E::DivU32(11, 12), // token
			/* 14 */ E::MulU32(13, 6),
			/* 15 */ E::AddU32(14, 10), // og element
			/* 16 */ E::LoadF32 {
				binding: 0,
				address: 15,
			},
			/* 17 */ E::LoadF32 {
				binding: 2,
				address: 11,
			}, // gate
			/* 18 */ E::MulF32(16, 17),
			/* 19 */ E::LtU32(4, 5), // region two live
			/* 20 */ E::DivU32(4, 12), // token2
			/* 21 */ E::BufferU32 {
				binding: 3,
				index: 4,
			}, // packed
			/* 22 */ E::Induction, // column loop
			/* 23 */ E::MulU32(20, 6),
			/* 24 */ E::AddU32(23, 22), // og2 element
			/* 25 */ E::LoadF32 {
				binding: 0,
				address: 24,
			},
			/* 26 */ E::MulU32(21, 6),
			/* 27 */ E::AddU32(26, 22), // packed element
			/* 28 */ E::LoadF32 {
				binding: 1,
				address: 27,
			},
			/* 29 */ E::MulF32(25, 28),
			/* 30 */ E::ConstF32(0),
			/* 31 */ E::RegisterF32 { tile: 0, index: 0 }, // sum
			/* 32 */ E::AddF32(31, 29),
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Output(0),
				Binding::Output(1),
			],
			vec![
				"tokens".into(),
				"routes_per_token".into(),
				"row_width".into(),
				"routes".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::LaneIf {
					cond: 8,
					body: vec![TileOp::StoreScalar {
						binding: 5,
						address: 4,
						value: 18,
					}],
				},
				TileOp::LaneIf {
					cond: 19,
					body: vec![
						TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: 30,
						},
						TileOp::Loop {
							extent: Extent::Push(2),
							step: 1,
							start: None,
							body: vec![TileOp::AssignScalar {
								tile: 0,
								index: 0,
								value: 32,
							}],
						},
						TileOp::StoreScalar {
							binding: 6,
							address: 4,
							value: 31,
						},
					],
				},
			],
		)
	}

	/// `moe_route_weights` — normalize route weights per token:
	/// `denominator = Σ_slot probabilities[token * experts +
	/// expert[slot]]` over in-range experts, then
	/// `output[token*K+slot] = prob * (denom > 0 ? 1/denom : 0)` or 0 for
	/// out-of-range experts. `[probabilities, expert_indices, output]` +
	/// `[tokens, experts, routes_per_token]` ABI.
	pub(in crate::runtime) fn moe_route_weights() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // token
			/* 5 */ E::Push(0), // tokens
			/* 6 */ E::GeU32(4, 5),
			/* 7 */ E::Push(2), // routes_per_token
			/* 8 */ E::MulU32(4, 7), // token * K
			/* 9 */ E::Induction, // route_slot
			/* 10 */ E::AddU32(8, 9), // route
			/* 11 */ E::BufferU32 {
				binding: 1,
				index: 10,
			}, // expert
			/* 12 */ E::Push(1), // experts
			/* 13 */ E::LtU32(11, 12), // in-range
			/* 14 */ E::MulU32(4, 12), // token * experts
			/* 15 */ E::AddU32(14, 11), // prob element
			/* 16 */ E::LoadF32 {
				binding: 0,
				address: 15,
			},
			/* 17 */ E::ConstF32(0), // 0.0f
			/* 18 */ E::RegisterF32 { tile: 0, index: 0 }, // denominator
			/* 19 */ E::AddF32(18, 16), // denom + prob
			/* 20 */ E::GtF32(18, 17), // denom > 0
			/* 21 */ E::ConstF32(0x3f800000), // 1.0f
			/* 22 */ E::DivF32(21, 18),
			/* 23 */ E::SelectF32 {
				cond: 20,
				if_true: 22,
				if_false: 17,
			},
			/* 24 */ E::RegisterF32 { tile: 1, index: 0 }, // inverse_denom
			/* 25 */ E::MulF32(16, 24),
			/* 26 */ E::SelectF32 {
				cond: 13,
				if_true: 25,
				if_false: 17,
			},
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec!["tokens".into(), "experts".into(), "routes_per_token".into()],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 6 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 17,
				},
				TileOp::Loop {
					extent: Extent::Push(2),
					step: 1,
					start: None,
					body: vec![TileOp::LaneIf {
						cond: 13,
						body: vec![TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: 19,
						}],
					}],
				},
				TileOp::AssignScalar {
					tile: 1,
					index: 0,
					value: 23,
				},
				TileOp::Loop {
					extent: Extent::Push(2),
					step: 1,
					start: None,
					body: vec![TileOp::StoreScalar {
						binding: 2,
						address: 10,
						value: 26,
					}],
				},
			],
		)
	}

	/// `moe_route_weights_backward` — per `index` in `tokens * experts`:
	/// `denominator`/`dot` accumulate over `routes_per_token` in-range
	/// slots, `selected_slot` records the slot whose expert equals
	/// `index % experts`; gradient is `(og[token*K+slot] - dot) /
	/// denominator` or 0. `[output_gradient, probabilities,
	/// expert_indices, route_weights, probability_gradient]` +
	/// `[tokens, experts, routes_per_token]` ABI.
	pub(in crate::runtime) fn moe_route_weights_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // index
			/* 5 */ E::Push(0), // tokens
			/* 6 */ E::Push(1), // experts
			/* 7 */ E::MulU32(5, 6), // element_count
			/* 8 */ E::GeU32(4, 7),
			/* 9 */ E::DivU32(4, 6), // token
			/* 10 */ E::ModU32(4, 6), // expert_index
			/* 11 */ E::Push(2), // routes_per_token
			/* 12 */ E::MulU32(9, 11), // token * K
			/* 13 */ E::Induction, // route_slot
			/* 14 */ E::AddU32(12, 13), // route
			/* 15 */ E::BufferU32 {
				binding: 2,
				index: 14,
			}, // expert
			/* 16 */ E::LtU32(15, 6), // in-range
			/* 17 */ E::MulU32(9, 6), // token * experts
			/* 18 */ E::AddU32(17, 15), // prob element
			/* 19 */ E::LoadF32 {
				binding: 1,
				address: 18,
			},
			/* 20 */ E::RegisterF32 { tile: 0, index: 0 }, // denominator
			/* 21 */ E::AddF32(20, 19),
			/* 22 */ E::LoadF32 {
				binding: 0,
				address: 14,
			}, // og[route]
			/* 23 */ E::LoadF32 {
				binding: 3,
				address: 14,
			}, // rw[route]
			/* 24 */ E::MulF32(22, 23),
			/* 25 */ E::RegisterF32 { tile: 1, index: 0 }, // dot
			/* 26 */ E::AddF32(25, 24),
			/* 27 */ E::EqU32(15, 10), // expert == expert_index
			/* 28 */ E::RegisterU32 { tile: 2, index: 0 }, // selected_slot
			/* 29 */ E::ConstU32(u32::MAX), // -1 as u32
			/* 30 */ E::NeqU32(28, 29), // selected >= 0
			/* 31 */ E::ConstF32(0), // 0.0f
			/* 32 */ E::GtF32(20, 31), // denom > 0
			/* 33 */ E::AndBool(30, 32),
			/* 34 */ E::AddU32(12, 28), // token*K + selected
			/* 35 */ E::LoadF32 {
				binding: 0,
				address: 34,
			},
			/* 36 */ E::SubF32(35, 25),
			/* 37 */ E::DivF32(36, 20),
			/* 38 */ E::SelectF32 {
				cond: 33,
				if_true: 37,
				if_false: 31,
			},
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Output(0),
			],
			vec!["tokens".into(), "experts".into(), "routes_per_token".into()],
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
			],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 31,
				},
				TileOp::AssignScalar {
					tile: 1,
					index: 0,
					value: 31,
				},
				TileOp::AssignScalarU32 {
					tile: 2,
					index: 0,
					value: 29,
				},
				TileOp::Loop {
					extent: Extent::Push(2),
					step: 1,
					start: None,
					body: vec![TileOp::LaneIf {
						cond: 16,
						body: vec![
							TileOp::AssignScalar {
								tile: 0,
								index: 0,
								value: 21,
							},
							TileOp::AssignScalar {
								tile: 1,
								index: 0,
								value: 26,
							},
							TileOp::LaneIf {
								cond: 27,
								body: vec![TileOp::AssignScalarU32 {
									tile: 2,
									index: 0,
									value: 13,
								}],
							},
						],
					}],
				},
				TileOp::StoreScalar {
					binding: 4,
					address: 4,
					value: 38,
				},
			],
		)
	}

	/// `cross_entropy` / `masked_cross_entropy` forward — the
	/// row-cooperative loss donors: the 2D-flattened
	/// `row = group_id.x + group_id.y * 65535` grid, strided max and
	/// `exp(x - row_max)` sum folds through `partial[]`, and the
	/// exclusive lane-0 `log(sum) + row_max - logits[target]` store
	/// guarded by `target_dtype == 1 && target < classes` (an invalid
	/// target leaves `0.0/0.0` NaN). `masked` adds the mask binding and
	/// the `mask == 0` early-return zero path. ABI: three or four
	/// buffers `[logits, targets, (mask), output]`, `[rows, classes,
	/// target_dtype]` push.
	pub(in crate::runtime) fn cross_entropy(lanes: u32, masked: bool) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"cross_entropy lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let mut lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // classes
			/* 2 */ E::GroupX,
			/* 3 */ E::GroupY,
			/* 4 */ E::ConstU32(65535),
			/* 5 */ E::MulU32(3, 4),
			/* 6 */ E::AddU32(2, 5), // row
			/* 7 */ E::GeU32(6, 0), // row >= rows
			/* 8 */ E::MulU32(6, 1), // row base
			/* 9 */ E::Induction, // column
			/* 10 */ E::AddU32(8, 9), // element address
			/* 11 */ E::LoadF32 {
				binding: 0,
				address: 10,
			}, // logit
			/* 12 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 13 */ E::ConstF32(0x0000_0000), // 0.0
			/* 14 */ E::Lane,
			/* 15 */ E::ConstU32(0),
			/* 16 */ E::SharedF32 {
				tile: 0,
				offset: 15,
			}, // partial[0]
			/* 17 */ E::RegisterF32 { tile: 1, index: 0 }, // max acc
			/* 18 */ E::RegisterF32 { tile: 2, index: 0 }, // sum acc
			/* 19 */ E::RegisterF32 { tile: 3, index: 0 }, // row max
			/* 20 */ E::SubF32(11, 19), // x - row_max
			/* 21 */ E::ExpF32(20),
			/* 22 */ E::BufferU32 {
				binding: 1,
				index: 6,
			}, // target
			/* 23 */ E::AddU32(8, 22), // base + target
			/* 24 */ E::LoadF32 {
				binding: 0,
				address: 23,
			}, // logits[target]
			/* 25 */ E::LogF32(16), // log(sum)
			/* 26 */ E::AddF32(25, 19), // + row_max
			/* 27 */ E::SubF32(26, 24), // loss
			/* 28 */ E::Push(2), // target_dtype
			/* 29 */ E::ConstU32(1),
			/* 30 */ E::EqU32(28, 29),
			/* 31 */ E::LtU32(22, 1), // target < classes
			/* 32 */ E::AndBool(30, 31), // valid
			/* 33 */ E::RegisterF32 { tile: 4, index: 0 }, // loss reg
			/* 34 */ E::ConstF32(0xffc0_0000), // -NaN: the donor's `0.0f/0.0f` sign
		];
		let mut ops = vec![TileOp::GuardIf { cond: 7 }];
		if masked {
			lane_exprs.extend([
				/* 35 */ E::LoadF32 {
					binding: 2,
					address: 6,
				}, // mask[row]
				/* 36 */ E::EqF32(35, 13), // mask == 0.0
			]);
			ops.push(TileOp::LaneIf {
				cond: 36,
				body: vec![
					TileOp::LaneGuard {
						lane: 15,
						body: vec![TileOp::StoreScalar {
							binding: 3,
							address: 6,
							value: 13,
						}],
					},
					TileOp::Return,
				],
			});
		}
		ops.extend([
			TileOp::LaneFold {
				acc: 1,
				init: 12,
				extent: Extent::Push(1),
				op: LaneReduce::Max,
				element: 11,
				serial: false,
			},
			TileOp::StoreShared {
				tile: 0,
				offset: 14,
				value: 17,
			},
			TileOp::Barrier,
			TileOp::TreeReduce {
				tile: 0,
				op: LaneReduce::Max,
			},
			TileOp::AssignScalar {
				tile: 3,
				index: 0,
				value: 16,
			},
			TileOp::Barrier,
			TileOp::LaneFold {
				acc: 2,
				init: 13,
				extent: Extent::Push(1),
				op: LaneReduce::Sum,
				element: 21,
				serial: false,
			},
			TileOp::StoreShared {
				tile: 0,
				offset: 14,
				value: 18,
			},
			TileOp::Barrier,
			TileOp::TreeReduce {
				tile: 0,
				op: LaneReduce::Sum,
			},
			TileOp::LaneGuard {
				lane: 15,
				body: vec![
					TileOp::AssignScalar {
						tile: 4,
						index: 0,
						value: 34,
					},
					TileOp::LaneIf {
						cond: 32,
						body: vec![TileOp::AssignScalar {
							tile: 4,
							index: 0,
							value: 27,
						}],
					},
					TileOp::StoreScalar {
						binding: if masked { 3 } else { 2 },
						address: 6,
						value: 33,
					},
				],
			},
		]);
		let mut bindings = vec![Binding::Input(0), Binding::Input(1)];
		if masked {
			bindings.push(Binding::Input(2));
		}
		bindings.push(Binding::Output(0));
		Ok(Self {
			bindings,
			push_scalars: vec!["rows".into(), "classes".into(), "target_dtype".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops,
		})
	}

	/// `cross_entropy_backward` / `masked_cross_entropy_backward` —
	/// the row-cooperative logits adjoints: the same 2D-flattened row
	/// grid and max/`exp` folds, then the strided
	/// `(exp(x - row_max) * inverse_sum - (column == target)) *
	/// inverse_count` epilogue under the `target_dtype`/`target`
	/// validity select, with invalid rows storing NaN like the donor.
	/// `masked` adds the mask binding, the zero-fill early return and
	/// `inverse_valid_count` (`1/float(push.valid_count)`) in place of
	/// `inverse_rows`. ABI: three or four buffers `[logits, targets,
	/// (mask), logits_gradient]`, `[rows, classes, target_dtype,
	/// (valid_count)]` push.
	pub(in crate::runtime) fn cross_entropy_backward(lanes: u32, masked: bool) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"cross_entropy_backward lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let count_push = if masked { 3 } else { 0 };
		let mut lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // classes
			/* 2 */ E::GroupX,
			/* 3 */ E::GroupY,
			/* 4 */ E::ConstU32(65535),
			/* 5 */ E::MulU32(3, 4),
			/* 6 */ E::AddU32(2, 5), // row
			/* 7 */ E::GeU32(6, 0), // row >= rows
			/* 8 */ E::MulU32(6, 1), // row base
			/* 9 */ E::Induction, // column
			/* 10 */ E::AddU32(8, 9), // element address
			/* 11 */ E::LoadF32 {
				binding: 0,
				address: 10,
			}, // logit
			/* 12 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 13 */ E::ConstF32(0x0000_0000), // 0.0
			/* 14 */ E::Lane,
			/* 15 */ E::ConstU32(0),
			/* 16 */ E::SharedF32 {
				tile: 0,
				offset: 15,
			}, // partial[0]
			/* 17 */ E::BufferU32 {
				binding: 1,
				index: 6,
			}, // target
			/* 18 */ E::RegisterF32 { tile: 1, index: 0 }, // max acc
			/* 19 */ E::RegisterF32 { tile: 2, index: 0 }, // sum acc
			/* 20 */ E::RegisterF32 { tile: 3, index: 0 }, // row max
			/* 21 */ E::SubF32(11, 20), // x - row_max
			/* 22 */ E::ExpF32(21),
			/* 23 */ E::RegisterF32 { tile: 4, index: 0 }, // inverse_sum
			/* 24 */ E::RegisterF32 { tile: 5, index: 0 }, // inverse_count
			/* 25 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 26 */ E::DivF32(25, 16), // 1 / sum
			/* 27 */ E::Push(count_push), // rows or valid_count
			/* 28 */ E::F32FromU32(27), // float(count)
			/* 29 */ E::DivF32(25, 28), // inverse_count
			/* 30 */ E::ConstU32(1),
			/* 31 */ E::EqU32(9, 17), // column == target
			/* 32 */
			E::SelectF32 {
				cond: 31,
				if_true: 25,
				if_false: 13,
			}, // 1.0 : 0.0
			/* 33 */ E::MulF32(22, 23), // exp * inverse_sum
			/* 34 */ E::SubF32(33, 32), // - indicator
			/* 35 */ E::MulF32(34, 24), // * inverse_count
			/* 36 */ E::Push(2), // target_dtype
			/* 37 */ E::EqU32(36, 30),
			/* 38 */ E::LtU32(17, 1), // target < classes
			/* 39 */ E::AndBool(37, 38), // valid_target
			/* 40 */ E::ConstF32(0xffc0_0000), // -NaN: the donor's `0.0f/0.0f` sign
			/* 41 */
			E::SelectF32 {
				cond: 39,
				if_true: 35,
				if_false: 40,
			}, // gradient
		];
		let mut ops = vec![TileOp::GuardIf { cond: 7 }];
		if masked {
			lane_exprs.extend([
				/* 42 */ E::LoadF32 {
					binding: 2,
					address: 6,
				}, // mask[row]
				/* 43 */ E::EqF32(42, 13), // mask == 0.0
			]);
			ops.push(TileOp::LaneIf {
				cond: 43,
				body: vec![
					TileOp::LaneLoop {
						extent: Extent::Push(1),
						body: vec![TileOp::StoreScalar {
							binding: 3,
							address: 10,
							value: 13,
						}],
					},
					TileOp::Return,
				],
			});
		}
		ops.extend([
			TileOp::LaneFold {
				acc: 1,
				init: 12,
				extent: Extent::Push(1),
				op: LaneReduce::Max,
				element: 11,
				serial: false,
			},
			TileOp::StoreShared {
				tile: 0,
				offset: 14,
				value: 18,
			},
			TileOp::Barrier,
			TileOp::TreeReduce {
				tile: 0,
				op: LaneReduce::Max,
			},
			TileOp::AssignScalar {
				tile: 3,
				index: 0,
				value: 16,
			},
			TileOp::Barrier,
			TileOp::LaneFold {
				acc: 2,
				init: 13,
				extent: Extent::Push(1),
				op: LaneReduce::Sum,
				element: 22,
				serial: false,
			},
			TileOp::StoreShared {
				tile: 0,
				offset: 14,
				value: 19,
			},
			TileOp::Barrier,
			TileOp::TreeReduce {
				tile: 0,
				op: LaneReduce::Sum,
			},
			TileOp::AssignScalar {
				tile: 4,
				index: 0,
				value: 26,
			},
			TileOp::AssignScalar {
				tile: 5,
				index: 0,
				value: 29,
			},
			TileOp::LaneLoop {
				extent: Extent::Push(1),
				body: vec![TileOp::StoreScalar {
					binding: if masked { 3 } else { 2 },
					address: 10,
					value: 41,
				}],
			},
		]);
		let mut bindings = vec![Binding::Input(0), Binding::Input(1)];
		if masked {
			bindings.push(Binding::Input(2));
		}
		bindings.push(Binding::Output(0));
		let mut push_scalars = vec!["rows".into(), "classes".into(), "target_dtype".into()];
		if masked {
			push_scalars.push("valid_count".into());
		}
		Ok(Self {
			bindings,
			push_scalars,
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops,
		})
	}

	/// `layer_norm_backward_sum` — the fused add + LayerNorm data
	/// adjoint: identical to `layer_norm_backward` except the output
	/// gradient is `upstream = addend_a + addend_b` continued by
	/// `+ addend_c` under `addend_count == 3` — spelled
	/// `count == 3 ? (a+b)+c : a+b` so both branches evaluate in-bounds
	/// (the fused node always binds all three addend slots, with
	/// `addend_b` duplicated for two addends) and the two-addend value
	/// is exactly the donor's. The epilogue additionally republishes
	/// `upstream` to the `dy` binding. ABI: nine buffers
	/// `[normalized, inverse_stddev, weight, addend_a, addend_b,
	/// addend_c, input_gradient, weight_contribution, dy]`, `[rows,
	/// columns, addend_count]` push, `[rows, 1, 1]` grid.
	pub(in crate::runtime) fn layer_norm_backward_sum(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"layer_norm_backward_sum lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // rows
			/* 1 */ E::Push(1), // columns
			/* 2 */ E::GroupX, // row
			/* 3 */ E::MulU32(2, 1), // row base
			/* 4 */ E::Induction, // column index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 3,
				address: 5,
			}, // addend_a
			/* 7 */ E::LoadF32 {
				binding: 4,
				address: 5,
			}, // addend_b
			/* 8 */ E::AddF32(6, 7), // a + b
			/* 9 */ E::LoadF32 {
				binding: 5,
				address: 5,
			}, // addend_c
			/* 10 */ E::AddF32(8, 9), // (a + b) + c
			/* 11 */ E::Push(2), // addend_count
			/* 12 */ E::ConstU32(3),
			/* 13 */ E::EqU32(11, 12), // count == 3
			/* 14 */
			E::SelectF32 {
				cond: 13,
				if_true: 10,
				if_false: 8,
			}, // upstream
			/* 15 */ E::LoadF32 {
				binding: 2,
				address: 4,
			}, // weight[column]
			/* 16 */ E::MulF32(14, 15), // scaled_gradient
			/* 17 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // normalized
			/* 18 */ E::MulF32(16, 17), // scaled_gradient * normalized
			/* 19 */ E::ConstF32(0x0000_0000), // 0.0
			/* 20 */ E::Lane, // lane
			/* 21 */ E::ConstU32(0),
			/* 22 */ E::RegisterF32 { tile: 1, index: 0 }, // sg acc
			/* 23 */ E::RegisterF32 { tile: 2, index: 0 }, // sgn acc
			/* 24 */ E::SharedF32 {
				tile: 0,
				offset: 21,
			}, // partial[0]
			/* 25 */ E::RegisterF32 { tile: 3, index: 0 }, // sg total
			/* 26 */ E::RegisterF32 { tile: 4, index: 0 }, // sgn total
			/* 27 */ E::LoadF32 {
				binding: 1,
				address: 2,
			}, // inverse_stddev[row]
			/* 28 */ E::RegisterF32 { tile: 5, index: 0 }, // istddev reg
			/* 29 */ E::F32FromU32(1), // float(columns)
			/* 30 */ E::ConstF32(0x3f80_0000), // 1.0
			/* 31 */ E::DivF32(30, 29), // inverse_columns
			/* 32 */ E::RegisterF32 { tile: 10, index: 0 }, // inv_cols reg
			/* 33 */ E::RegisterF32 { tile: 6, index: 0 }, // scaled reg
			/* 34 */ E::MulF32(33, 17), // scaled * normalized
			/* 35 */ E::AddF32(22, 33), // sg + scaled
			/* 36 */ E::AddF32(23, 34), // sgn + scaled*n
			/* 37 */ E::RegisterF32 { tile: 7, index: 0 }, // output_gradient reg
			/* 38 */ E::RegisterF32 { tile: 8, index: 0 }, // weight reg
			/* 39 */ E::RegisterF32 { tile: 9, index: 0 }, // normalized reg
			/* 40 */ E::MulF32(37, 38), // output_gradient * weight
			/* 41 */ E::MulF32(25, 32), // sg * inverse_columns
			/* 42 */ E::MulF32(39, 26), // normalized * sgn
			/* 43 */ E::MulF32(42, 32), // * inverse_columns
			/* 44 */ E::SubF32(40, 41), // og*w - sg*inv_cols
			/* 45 */ E::SubF32(44, 43), // - n*sgn*inv_cols
			/* 46 */ E::MulF32(28, 45), // input_gradient
			/* 47 */ E::MulF32(37, 39), // weight_contribution
		];
		Ok(Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Input(3),
				Binding::Input(4),
				Binding::Input(5),
				Binding::Output(0),
				Binding::Output(1),
				Binding::Output(2),
			],
			push_scalars: vec!["rows".into(), "columns".into(), "addend_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::AssignScalar {
					tile: 1,
					index: 0,
					value: 19,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 19,
				},
				// The donor names `scaled_gradient` once per iteration so
				// both accumulators consume the same SSA value; materializing
				// it as a per-iteration register preserves that contraction
				// shape through backend lowering.
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::AssignScalar {
							tile: 6,
							index: 0,
							value: 16,
						},
						TileOp::AssignScalar {
							tile: 1,
							index: 0,
							value: 35,
						},
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 36,
						},
					],
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 20,
					value: 22,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 24,
				},
				TileOp::Barrier,
				TileOp::StoreShared {
					tile: 0,
					offset: 20,
					value: 23,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 24,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: 27,
				},
				TileOp::AssignScalar {
					tile: 10,
					index: 0,
					value: 31,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![
						TileOp::AssignScalar {
							tile: 7,
							index: 0,
							value: 14,
						},
						TileOp::AssignScalar {
							tile: 8,
							index: 0,
							value: 15,
						},
						TileOp::AssignScalar {
							tile: 9,
							index: 0,
							value: 17,
						},
						TileOp::StoreScalar {
							binding: 6,
							address: 5,
							value: 46,
						},
						TileOp::StoreScalar {
							binding: 7,
							address: 5,
							value: 47,
						},
						TileOp::StoreScalar {
							binding: 8,
							address: 5,
							value: 37,
						},
					],
				},
			],
		})
	}

	/// `embedding_backward_sum` — the fused two-addend Embedding
	/// scatter-add adjoint: one flat thread per table scalar scans all
	/// positions and accumulates `addend_a[pos * dim + feature] +
	/// addend_b[...]` where `indices[pos] == row`. The position-gated
	/// accumulate is spelled `index == row ? gradient + ab : gradient`
	/// so non-matching positions leave the register untouched like the
	/// donor's `if` (a `+ 0.0` merge would flip a `-0.0` accumulator).
	/// ABI: four buffers `[indices, addend_a, addend_b,
	/// weight_gradient]`, `[index_count, num_embeddings,
	/// embedding_dim]` push.
	pub(in crate::runtime) fn embedding_backward_sum() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // weight_index
			/* 5 */ E::Push(1), // num_embeddings
			/* 6 */ E::Push(2), // embedding_dim
			/* 7 */ E::MulU32(5, 6), // weight_count
			/* 8 */ E::GeU32(4, 7), // dead lane
			/* 9 */ E::DivU32(4, 6), // row
			/* 10 */ E::ModU32(4, 6), // feature
			/* 11 */ E::ConstF32(0x0000_0000), // 0.0
			/* 12 */ E::LoopVar(2), // position
			/* 13 */ E::BufferU32 {
				binding: 0,
				index: 12,
			}, // indices[position]
			/* 14 */ E::EqU32(13, 9), // index == row
			/* 15 */ E::MulU32(12, 6),
			/* 16 */ E::AddU32(15, 10), // element
			/* 17 */ E::LoadF32 {
				binding: 1,
				address: 16,
			}, // addend_a[element]
			/* 18 */ E::LoadF32 {
				binding: 2,
				address: 16,
			}, // addend_b[element]
			/* 19 */ E::AddF32(17, 18), // combined
			/* 20 */ E::RegisterF32 { tile: 0, index: 0 }, // gradient
			/* 21 */ E::AddF32(20, 19), // gradient + ab
			/* 22 */
			E::SelectF32 {
				cond: 14,
				if_true: 21,
				if_false: 20,
			}, // position-gated accumulate
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			vec![
				"index_count".into(),
				"num_embeddings".into(),
				"embedding_dim".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 11,
				},
				TileOp::Loop {
					extent: Extent::Push(0),
					step: 1,
					start: None,
					body: vec![TileOp::AssignScalar {
						tile: 0,
						index: 0,
						value: 22,
					}],
				},
				TileOp::StoreScalar {
					binding: 3,
					address: 4,
					value: 20,
				},
			],
		)
	}

	/// `MlEmbeddingBackwardF32` — the deterministic scatter adjoint:
	/// one invocation per table scalar folds the `index_count` position
	/// loop in dispatch order.
	pub(in crate::runtime) fn embedding_backward() -> Self {
		Self::embedding_backward_program(false)
	}

	/// `MlEmbeddingBackwardU8F32` — the packed-u8 index-table variant.
	pub(in crate::runtime) fn embedding_backward_u8() -> Self {
		Self::embedding_backward_program(true)
	}

	/// Shared `embedding_backward` builder — `packed` selects the
	/// `load_u8` byte extraction over the plain u32 index read.
	fn embedding_backward_program(packed: bool) -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // weight_index
			/* 5 */ E::Push(1), // num_embeddings
			/* 6 */ E::Push(2), // embedding_dim
			/* 7 */ E::MulU32(5, 6), // weight_count
			/* 8 */ E::GeU32(4, 7), // dead lane
			/* 9 */ E::DivU32(4, 6), // row
			/* 10 */ E::ModU32(4, 6), // feature
			/* 11 */ E::ConstF32(0x0000_0000), // 0.0
			/* 12 */ E::LoopVar(2), // position
			/* 13 */
			if packed {
				E::BufferU8 {
					binding: 0,
					index: 12,
				}
			} else {
				E::BufferU32 {
					binding: 0,
					index: 12,
				}
			}, // indices[position]
			/* 14 */ E::EqU32(13, 9), // index == row
			/* 15 */ E::MulU32(12, 6),
			/* 16 */ E::AddU32(15, 10), // element
			/* 17 */
			E::LoadF32 {
				binding: 1,
				address: 16,
			}, // output_gradient[element]
			/* 18 */ E::RegisterF32 { tile: 0, index: 0 }, // gradient
			/* 19 */ E::AddF32(18, 17), // gradient + dy
			/* 20 */
			E::SelectF32 {
				cond: 14,
				if_true: 19,
				if_false: 18,
			}, // position-gated accumulate
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"index_count".into(),
				"num_embeddings".into(),
				"embedding_dim".into(),
			],
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 8 },
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: 11,
				},
				TileOp::Loop {
					extent: Extent::Push(0),
					step: 1,
					start: None,
					body: vec![TileOp::AssignScalar {
						tile: 0,
						index: 0,
						value: 20,
					}],
				},
				TileOp::StoreScalar {
					binding: 2,
					address: 4,
					value: 18,
				},
			],
		)
	}

	/// `MatrixSumBackwardF32` — the broadcast adjoint: every output
	/// element copies its `(outer, inner)` gradient slot.
	pub(in crate::runtime) fn sum_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // input_index
			/* 5 */ E::Push(0), // outer_size
			/* 6 */ E::Push(1), // dim_size
			/* 7 */ E::Push(2), // inner_size
			/* 8 */ E::MulU32(6, 7), // dim_size * inner_size
			/* 9 */ E::MulU32(5, 8), // input_count
			/* 10 */ E::GeU32(4, 9), // dead lane
			/* 11 */ E::DivU32(4, 8), // outer
			/* 12 */ E::ModU32(4, 7), // inner
			/* 13 */ E::MulU32(11, 7),
			/* 14 */ E::AddU32(13, 12), // gradient_index
			/* 15 */ E::LoadF32 {
				binding: 0,
				address: 14,
			},
		];
		Self::flat_program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec!["outer_size".into(), "dim_size".into(), "inner_size".into()],
			vec![],
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 10 },
				TileOp::StoreScalar {
					binding: 1,
					address: 4,
					value: 15,
				},
			],
		)
	}

	/// `max` — the full-tensor single-workgroup Max reduction: a
	/// `-FLT_MAX` lane-strided fold, one value per lane into shared
	/// memory, then the max tree and the lane-zero scalar store.
	/// ABI: `[input, output]` + `[element_count]` push, `[1, 1, 1]`
	/// grid.
	pub(in crate::runtime) fn max(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"max lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			},
			/* 3 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 4 */ E::RegisterF32 { tile: 1, index: 0 },
			/* 5 */ E::Lane,
			/* 6 */ E::ConstU32(0),
			/* 7 */ E::SharedF32 { tile: 0, offset: 6 },
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::LaneFold {
					acc: 1,
					init: 3,
					extent: Extent::Push(0),
					op: LaneReduce::Max,
					element: 2,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 5,
					value: 4,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Max,
				},
				TileOp::LaneGuard {
					lane: 6,
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 6,
						value: 7,
					}],
				},
			],
		})
	}

	/// `max_wave` — the subgroup candidate for the full-tensor Max
	/// reduction: the same lane-strided `-FLT_MAX` fold, then
	/// `WaveActiveMax` reduces within each subgroup, leader lanes
	/// stage one partial per subgroup, and the first subgroup folds
	/// the partials with a second `WaveActiveMax` under the
	/// `lane < WaveGetLaneCount()` gate (uniform within that
	/// subgroup). Max is order-insensitive, so the result is
	/// bit-identical to the shared-tree candidate while the schedule
	/// replaces `log2(lanes)` shared round-trips with two subgroup
	/// reduces. Requires the queried subgroup arithmetic capability.
	/// ABI matches `max`: `[input, output]` + `[element_count]`,
	/// `[1, 1, 1]` grid.
	pub(in crate::runtime) fn max_wave(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"max_wave lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			},
			/* 3 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 4 */ E::RegisterF32 { tile: 1, index: 0 }, // best
			/* 5 */ E::Lane,
			/* 6 */ E::WaveMax(4), // subgroup max
			/* 7 */ E::WaveLaneCount, // S
			/* 8 */ E::ModU32(5, 7), // lane % S
			/* 9 */ E::ConstU32(0),
			/* 10 */ E::EqU32(8, 9), // subgroup leader
			/* 11 */ E::DivU32(5, 7), // lane / S — partial slot
			/* 12 */ E::ConstU32(lanes),
			/* 13 */ E::DivU32(12, 7), // W = lanes / S
			/* 14 */ E::LtU32(5, 13), // lane < W — live partial
			/* 15 */ E::SharedF32 { tile: 0, offset: 5 }, // shared[lane]
			/* 16 */
			E::SelectF32 {
				cond: 14,
				if_true: 15,
				if_false: 3,
			}, // lane < W ? shared[lane] : -FLT_MAX
			/* 17 */ E::WaveMax(16), // total
			/* 18 */ E::LtU32(5, 7), // lane < S — first subgroup
			/* 19 */ E::RegisterF32 { tile: 2, index: 0 }, // wave
			/* 20 */ E::RegisterF32 { tile: 3, index: 0 }, // total
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::LaneFold {
					acc: 1,
					init: 3,
					extent: Extent::Push(0),
					op: LaneReduce::Max,
					element: 2,
					serial: false,
				},
				// The wave op must be evaluated by every lane of the
				// subgroup — a leader-gated spelling would reduce over
				// one active lane only.
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 6,
				},
				TileOp::LaneIf {
					cond: 10,
					body: vec![TileOp::StoreShared {
						tile: 0,
						offset: 11,
						value: 19,
					}],
				},
				TileOp::Barrier,
				TileOp::LaneIf {
					cond: 18,
					body: vec![
						// The whole first subgroup evaluates the final
						// reduce; only lane zero's store is gated.
						TileOp::AssignScalar {
							tile: 3,
							index: 0,
							value: 17,
						},
						TileOp::LaneGuard {
							lane: 9,
							body: vec![TileOp::StoreScalar {
								binding: 1,
								address: 9,
								value: 20,
							}],
						},
					],
				},
			],
		})
	}

	/// `sum` subgroup variant — the same lane-strided fold and
	/// leader-staged partials as `max_wave`, with `WaveActiveSum` for
	/// both reduces and `0.0f` as the fold init and dead-partial
	/// sentinel. Sum is order-sensitive: this candidate changes the
	/// reduction order versus the shared tree, so it is admitted only
	/// behind `OA_TILE_SUM_WAVE` and qualifies against a tolerance
	/// oracle, not incumbent bits. ABI: `[input, output]` +
	/// `[element_count]` push, `[1, 1, 1]` grid.
	pub(in crate::runtime) fn sum_wave(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"sum_wave lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			},
			/* 3 */ E::ConstF32(0), // 0.0f
			/* 4 */ E::RegisterF32 { tile: 1, index: 0 }, // acc
			/* 5 */ E::Lane,
			/* 6 */ E::WaveSum(4), // subgroup sum
			/* 7 */ E::WaveLaneCount, // S
			/* 8 */ E::ModU32(5, 7), // lane % S
			/* 9 */ E::ConstU32(0),
			/* 10 */ E::EqU32(8, 9), // subgroup leader
			/* 11 */ E::DivU32(5, 7), // lane / S — partial slot
			/* 12 */ E::ConstU32(lanes),
			/* 13 */ E::DivU32(12, 7), // W = lanes / S
			/* 14 */ E::LtU32(5, 13), // lane < W — live partial
			/* 15 */ E::SharedF32 { tile: 0, offset: 5 }, // shared[lane]
			/* 16 */
			E::SelectF32 {
				cond: 14,
				if_true: 15,
				if_false: 3,
			}, // lane < W ? shared[lane] : 0.0f
			/* 17 */ E::WaveSum(16), // total
			/* 18 */ E::LtU32(5, 7), // lane < S — first subgroup
			/* 19 */ E::RegisterF32 { tile: 2, index: 0 }, // wave
			/* 20 */ E::RegisterF32 { tile: 3, index: 0 }, // total
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::LaneFold {
					acc: 1,
					init: 3,
					extent: Extent::Push(0),
					op: LaneReduce::Sum,
					element: 2,
					serial: false,
				},
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 6,
				},
				TileOp::LaneIf {
					cond: 10,
					body: vec![TileOp::StoreShared {
						tile: 0,
						offset: 11,
						value: 19,
					}],
				},
				TileOp::Barrier,
				TileOp::LaneIf {
					cond: 18,
					body: vec![
						TileOp::AssignScalar {
							tile: 3,
							index: 0,
							value: 17,
						},
						TileOp::LaneGuard {
							lane: 9,
							body: vec![TileOp::StoreScalar {
								binding: 1,
								address: 9,
								value: 20,
							}],
						},
					],
				},
			],
		})
	}

	/// `argmax` — the full-tensor paired argmax reduction: each lane
	/// keeps `(best_value, best_index)` through the strided fold, the
	/// pair lands in shared f32/u32 tiles, and the paired tree reduce
	/// applies the donor's `other > best || (equal && lower index)`
	/// tie-break before lane zero stores the u32 index. The two
	/// register updates read the pre-update `best` in donor order —
	/// the index select assigns before the value select so both
	/// comparisons see the same stale maximum. ABI: `[input, output]`
	/// + `[element_count]` push, `[1, 1, 1]` grid.
	pub(in crate::runtime) fn argmax(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"argmax lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			}, // value
			/* 3 */ E::RegisterF32 { tile: 2, index: 0 }, // best_value
			/* 4 */ E::RegisterU32 { tile: 3, index: 0 }, // best_index
			/* 5 */ E::GtF32(2, 3), // value > best
			/* 6 */
			E::SelectF32 {
				cond: 5,
				if_true: 2,
				if_false: 3,
			}, // value > best ? value : best
			/* 7 */
			E::SelectU32 {
				cond: 5,
				if_true: 1,
				if_false: 4,
			}, // value > best ? index : best_index
			/* 8 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 9 */ E::ConstU32(0),
			/* 10 */ E::Lane,
			/* 11 */ E::SharedU32 { tile: 1, offset: 9 }, // shared_indices[0]
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::SharedU32 { elements: lanes },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
			],
			lane_exprs,
			ops: vec![
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 8,
				},
				TileOp::AssignScalarU32 {
					tile: 3,
					index: 0,
					value: 9,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(0),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 3,
							index: 0,
							value: 7,
						},
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 6,
						},
					],
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 10,
					value: 3,
				},
				TileOp::StoreSharedU32 {
					tile: 1,
					offset: 10,
					value: 4,
				},
				TileOp::Barrier,
				TileOp::TreeReduceArgmax {
					value_tile: 0,
					index_tile: 1,
				},
				TileOp::LaneGuard {
					lane: 9,
					body: vec![TileOp::StoreScalarU32 {
						binding: 1,
						address: 9,
						value: 11,
					}],
				},
			],
		})
	}

	/// `argmax` subgroup variant — the same lane-strided `(value,
	/// index)` fold as `argmax`, then a two-reduce subgroup pass:
	/// `WaveActiveMax` finds the subgroup maximum and
	/// `WaveActiveMin` over `value == wave_max ? index : u32::MAX`
	/// finds the lowest index holding it — the donor's tie-break
	/// exactly. Leaders stage both into shared tiles and the first
	/// subgroup repeats the pair reduce for the global result. Both
	/// reductions are order-insensitive, so the route stays
	/// bit-exact with the tree candidate. ABI: `[input, output]` +
	/// `[element_count]` push, `[1, 1, 1]` grid.
	pub(in crate::runtime) fn argmax_wave(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"argmax_wave lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // element_count
			/* 1 */ E::Induction, // strided element index
			/* 2 */ E::LoadF32 {
				binding: 0,
				address: 1,
			}, // value
			/* 3 */ E::RegisterF32 { tile: 2, index: 0 }, // best_value
			/* 4 */ E::RegisterU32 { tile: 3, index: 0 }, // best_index
			/* 5 */ E::GtF32(2, 3), // value > best
			/* 6 */
			E::SelectF32 {
				cond: 5,
				if_true: 2,
				if_false: 3,
			},
			/* 7 */
			E::SelectU32 {
				cond: 5,
				if_true: 1,
				if_false: 4,
			},
			/* 8 */ E::ConstF32(0xff7f_ffff), // -FLT_MAX
			/* 9 */ E::ConstU32(0),
			/* 10 */ E::Lane,
			/* 11 */ E::WaveMax(3), // subgroup max
			/* 12 */ E::RegisterF32 { tile: 4, index: 0 }, // wave_value
			/* 13 */ E::EqF32(3, 12), // best == wave_value
			/* 14 */ E::ConstU32(0xffff_ffff), // u32::MAX sentinel
			/* 15 */
			E::SelectU32 {
				cond: 13,
				if_true: 4,
				if_false: 14,
			}, // index contribution
			/* 16 */ E::WaveMinU32(15), // lowest index at the max
			/* 17 */ E::WaveLaneCount, // S
			/* 18 */ E::ModU32(10, 17), // lane % S
			/* 19 */ E::EqU32(18, 9), // subgroup leader
			/* 20 */ E::DivU32(10, 17), // lane / S — partial slot
			/* 21 */ E::ConstU32(lanes),
			/* 22 */ E::DivU32(21, 17), // W = lanes / S
			/* 23 */ E::LtU32(10, 22), // lane < W — live partial
			/* 24 */ E::SharedF32 {
				tile: 0,
				offset: 10,
			}, // shared_v[lane]
			/* 25 */ E::SharedU32 {
				tile: 1,
				offset: 10,
			}, // shared_i[lane]
			/* 26 */
			E::SelectF32 {
				cond: 23,
				if_true: 24,
				if_false: 8,
			}, // staged value
			/* 27 */
			E::SelectU32 {
				cond: 23,
				if_true: 25,
				if_false: 14,
			}, // staged index
			/* 28 */ E::LtU32(10, 17), // lane < S — first subgroup
			/* 29 */ E::RegisterF32 { tile: 6, index: 0 }, // staged v
			/* 30 */ E::WaveMax(29), // total_v
			/* 31 */ E::RegisterF32 { tile: 8, index: 0 }, // total_v reg
			/* 32 */ E::EqF32(29, 31), // v == total_v
			/* 33 */ E::RegisterU32 { tile: 7, index: 0 }, // staged i
			/* 34 */
			E::SelectU32 {
				cond: 32,
				if_true: 33,
				if_false: 14,
			}, // index contribution
			/* 35 */ E::WaveMinU32(34), // total_i
			/* 36 */ E::RegisterU32 { tile: 9, index: 0 }, // total_i reg
			/* 37 */ E::RegisterU32 { tile: 5, index: 0 }, // wave_index
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::SharedU32 { elements: lanes },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::RegisterU32 { elements: 1 },
			],
			lane_exprs,
			ops: vec![
				TileOp::AssignScalar {
					tile: 2,
					index: 0,
					value: 8,
				},
				TileOp::AssignScalarU32 {
					tile: 3,
					index: 0,
					value: 9,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(0),
					body: vec![
						TileOp::AssignScalarU32 {
							tile: 3,
							index: 0,
							value: 7,
						},
						TileOp::AssignScalar {
							tile: 2,
							index: 0,
							value: 6,
						},
					],
				},
				// The wave reduces must run on every lane of the
				// subgroup — only the leader store is gated.
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 11,
				},
				TileOp::AssignScalarU32 {
					tile: 5,
					index: 0,
					value: 16,
				},
				TileOp::LaneIf {
					cond: 19,
					body: vec![
						TileOp::StoreShared {
							tile: 0,
							offset: 20,
							value: 12,
						},
						TileOp::StoreSharedU32 {
							tile: 1,
							offset: 20,
							value: 37,
						},
					],
				},
				TileOp::Barrier,
				TileOp::LaneIf {
					cond: 28,
					body: vec![
						TileOp::AssignScalar {
							tile: 6,
							index: 0,
							value: 26,
						},
						TileOp::AssignScalarU32 {
							tile: 7,
							index: 0,
							value: 27,
						},
						TileOp::AssignScalar {
							tile: 8,
							index: 0,
							value: 30,
						},
						TileOp::AssignScalarU32 {
							tile: 9,
							index: 0,
							value: 35,
						},
						TileOp::LaneGuard {
							lane: 9,
							body: vec![TileOp::StoreScalarU32 {
								binding: 1,
								address: 9,
								value: 36,
							}],
						},
					],
				},
			],
		})
	}

	/// `max_backward` — the flat per-thread Max adjoint: one lane per
	/// element writes `(value == maximum) ? output_gradient : 0.0f`.
	/// ABI: `[input, maximum, output_gradient, input_gradient]` +
	/// `[element_count]` push.
	pub(in crate::runtime) fn max_backward() -> Self {
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::GroupX,
			/* 1 */ E::ConstU32(256),
			/* 2 */ E::MulU32(0, 1),
			/* 3 */ E::Lane,
			/* 4 */ E::AddU32(2, 3), // index
			/* 5 */ E::Push(0), // element_count
			/* 6 */ E::GeU32(4, 5), // dead lane
			/* 7 */ E::LoadF32 {
				binding: 0,
				address: 4,
			}, // value
			/* 8 */ E::ConstU32(0),
			/* 9 */ E::LoadF32 {
				binding: 1,
				address: 8,
			}, // maximum
			/* 10 */ E::LoadF32 {
				binding: 2,
				address: 8,
			}, // output_gradient
			/* 11 */ E::EqF32(7, 9), // value == maximum
			/* 12 */ E::ConstF32(0x0000_0000), // 0.0
			/* 13 */ E::SelectF32 {
				cond: 11,
				if_true: 10,
				if_false: 12,
			},
		];
		Self::flat_program(
			vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			vec!["element_count".into()],
			Vec::new(),
			lane_exprs,
			vec![
				TileOp::GuardIf { cond: 6 },
				TileOp::StoreScalar {
					binding: 3,
					address: 4,
					value: 13,
				},
			],
		)
	}

	/// `channel_normalize` — the `mel_normalize` donor re-expressed as a
	/// row-cooperative block program (OA-CC-32). One workgroup per
	/// channel, `lanes` threads striding the channel's contiguous
	/// `values_per_channel` f32 run: two shared-tree reductions produce
	/// the channel mean and variance, then the strided epilogue writes
	/// `(x - mean) * rsqrt(variance / n + 1e-5)` — identical math and
	/// precision to the donor. ABI: two buffers `[input, output]`,
	/// `[channels, values_per_channel]` push scalars, donor grid
	/// `[channels, 1, 1]` (`workgroups(channels, 1)`).
	pub(in crate::runtime) fn channel_normalize(lanes: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 {
			return Err(Error::invalid_argument(
				"channel_normalize lanes must be a power of two within the workgroup floor",
			));
		}
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(0), // channels
			/* 1 */ E::Push(1), // values_per_channel
			/* 2 */ E::GroupX, // channel
			/* 3 */ E::MulU32(2, 1), // channel base
			/* 4 */ E::Induction, // element index
			/* 5 */ E::AddU32(3, 4), // element address
			/* 6 */ E::LoadF32 {
				binding: 0,
				address: 5,
			}, // x
			/* 7 */ E::ConstF32(0x0000_0000), // 0.0
			/* 8 */ E::Lane, // lane
			/* 9 */ E::RegisterF32 { tile: 2, index: 0 }, // fold acc
			/* 10 */ E::ConstU32(0),
			/* 11 */ E::SharedF32 {
				tile: 0,
				offset: 10,
			}, // partial_sum[0]
			/* 12 */ E::F32FromU32(1), // float(values)
			/* 13 */ E::DivF32(11, 12), // mean
			/* 14 */ E::RegisterF32 { tile: 3, index: 0 }, // mean
			/* 15 */ E::SubF32(6, 14), // centered
			/* 16 */ E::MulF32(15, 15), // centered^2
			/* 17 */ E::SharedF32 {
				tile: 1,
				offset: 10,
			}, // partial_sq[0]
			/* 18 */ E::DivF32(17, 12), // variance
			/* 19 */ E::ConstF32(0x3727_c5ac), // 1e-5
			/* 20 */ E::AddF32(18, 19), // variance + eps
			/* 21 */ E::RsqrtF32(20), // inverse_stddev
			/* 22 */ E::RegisterF32 { tile: 4, index: 0 }, // inv reg
			/* 23 */ E::MulF32(15, 22), // normalized
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["channels".into(), "values_per_channel".into()],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneFold {
					acc: 2,
					init: 7,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 6,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 0,
					offset: 8,
					value: 9,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 0,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 3,
					index: 0,
					value: 13,
				},
				TileOp::LaneFold {
					acc: 2,
					init: 7,
					extent: Extent::Push(1),
					op: LaneReduce::Sum,
					element: 16,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 1,
					offset: 8,
					value: 9,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 4,
					index: 0,
					value: 21,
				},
				TileOp::LaneLoop {
					extent: Extent::Push(1),
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: 5,
						value: 23,
					}],
				},
			],
		})
	}

	/// `image_resize` — the `resize_{nearest,bilinear}` donors re-expressed
	/// as a tile-IR gather program (OA-CC-33): a 16x16 workgroup per
	/// output tile, `group_id.z` the packed batch-channel axis, source
	/// coordinates composed from lane expressions, and the typed
	/// `BilinearF32` fetch over the input plane. The nearest variant
	/// shares the coordinate/index vocabulary and lowers to a single
	/// clamped load. ABI: two buffers `[input, output]`, donor push
	/// `[batch, channels, in_h, in_w, out_h, out_w]`, donor grid
	/// `[ceil(out_w/16), ceil(out_h/16), batch*channels]`.
	pub(in crate::runtime) fn image_resize(nearest: bool) -> Result<Self> {
		let mut lane_exprs: Vec<LaneExpr> = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		// Push-scalar indices in donor order.
		const P_BATCH: usize = 0;
		const P_CHANNELS: usize = 1;
		const P_IN_H: usize = 2;
		const P_IN_W: usize = 3;
		const P_OUT_H: usize = 4;
		const P_OUT_W: usize = 5;

		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let batch_channel = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let half = expr!(LaneExpr::ConstF32(0.5f32.to_bits()));
		let one = expr!(LaneExpr::ConstU32(1));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(P_BATCH));
		let channels = expr!(LaneExpr::Push(P_CHANNELS));
		let in_height = expr!(LaneExpr::Push(P_IN_H));
		let in_width = expr!(LaneExpr::Push(P_IN_W));
		let out_height = expr!(LaneExpr::Push(P_OUT_H));
		let out_width = expr!(LaneExpr::Push(P_OUT_W));
		let batch_channels = expr!(LaneExpr::MulU32(batch, channels));
		let x_over = expr!(LaneExpr::GeU32(out_x, out_width));
		let y_over = expr!(LaneExpr::GeU32(out_y, out_height));
		let z_over = expr!(LaneExpr::GeU32(batch_channel, batch_channels));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let out_x_f = expr!(LaneExpr::F32FromU32(out_x));
		let out_y_f = expr!(LaneExpr::F32FromU32(out_y));
		let in_width_f = expr!(LaneExpr::F32FromU32(in_width));
		let in_height_f = expr!(LaneExpr::F32FromU32(in_height));
		let out_width_f = expr!(LaneExpr::F32FromU32(out_width));
		let out_height_f = expr!(LaneExpr::F32FromU32(out_height));
		let in_plane = expr!(LaneExpr::MulU32(in_height, in_width));
		let plane_base = expr!(LaneExpr::MulU32(batch_channel, in_plane));
		let out_rows = expr!(LaneExpr::MulU32(batch_channel, out_height));
		let out_row = expr!(LaneExpr::AddU32(out_rows, out_y));
		let out_line = expr!(LaneExpr::MulU32(out_row, out_width));
		let out_index = expr!(LaneExpr::AddU32(out_line, out_x));
		let value = if nearest {
			// floor(out * in / out), clamped to the last source texel.
			let x_scale = expr!(LaneExpr::MulF32(out_x_f, in_width_f));
			let x_raw = expr!(LaneExpr::DivF32(x_scale, out_width_f));
			let x_trunc = expr!(LaneExpr::U32FromF32(x_raw));
			let x_last = expr!(LaneExpr::SubU32(in_width, one));
			let in_x = expr!(LaneExpr::MinU32(x_trunc, x_last));
			let y_scale = expr!(LaneExpr::MulF32(out_y_f, in_height_f));
			let y_raw = expr!(LaneExpr::DivF32(y_scale, out_height_f));
			let y_trunc = expr!(LaneExpr::U32FromF32(y_raw));
			let y_last = expr!(LaneExpr::SubU32(in_height, one));
			let in_y = expr!(LaneExpr::MinU32(y_trunc, y_last));
			let row = expr!(LaneExpr::MulU32(in_y, in_width));
			let offset = expr!(LaneExpr::AddU32(row, in_x));
			let address = expr!(LaneExpr::AddU32(plane_base, offset));
			expr!(LaneExpr::LoadF32 {
				binding: 0,
				address,
			})
		} else {
			// (out + 0.5) * in / out - 0.5, then the typed gather.
			let x_centered = expr!(LaneExpr::AddF32(out_x_f, half));
			let x_scaled = expr!(LaneExpr::MulF32(x_centered, in_width_f));
			let x_ratio = expr!(LaneExpr::DivF32(x_scaled, out_width_f));
			let in_x_f = expr!(LaneExpr::SubF32(x_ratio, half));
			let y_centered = expr!(LaneExpr::AddF32(out_y_f, half));
			let y_scaled = expr!(LaneExpr::MulF32(y_centered, in_height_f));
			let y_ratio = expr!(LaneExpr::DivF32(y_scaled, out_height_f));
			let in_y_f = expr!(LaneExpr::SubF32(y_ratio, half));
			expr!(LaneExpr::BilinearF32 {
				binding: 0,
				base: plane_base,
				xf: in_x_f,
				yf: in_y_f,
				width: in_width,
				height: in_height,
			})
		};
		let ops = vec![
			TileOp::GuardIf {
				cond: out_of_bounds,
			},
			TileOp::StoreScalar {
				binding: 1,
				address: out_index,
				value,
			},
		];
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops,
		})
	}

	/// `image_flip` — the `flip` donor re-expressed on the gather
	/// vocabulary (OA-CC-33): a 16x16 workgroup per output tile,
	/// `group_id.z` the packed batch-channel axis, horizontal/vertical
	/// index reversal as `SelectU32` over the push flags. ABI: two
	/// buffers `[input, output]`, donor push
	/// `[batch, channels, height, width, horizontal, vertical]`, donor
	/// grid `[ceil(width/16), ceil(height/16), batch*channels]`.
	pub(in crate::runtime) fn image_flip() -> Self {
		let mut lane_exprs: Vec<LaneExpr> = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		// Push-scalar indices in donor order.
		const P_BATCH: usize = 0;
		const P_CHANNELS: usize = 1;
		const P_HEIGHT: usize = 2;
		const P_WIDTH: usize = 3;
		const P_HORIZONTAL: usize = 4;
		const P_VERTICAL: usize = 5;

		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let plane = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let zero = expr!(LaneExpr::ConstU32(0));
		let one = expr!(LaneExpr::ConstU32(1));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(P_BATCH));
		let channels = expr!(LaneExpr::Push(P_CHANNELS));
		let height = expr!(LaneExpr::Push(P_HEIGHT));
		let width = expr!(LaneExpr::Push(P_WIDTH));
		let horizontal = expr!(LaneExpr::Push(P_HORIZONTAL));
		let vertical = expr!(LaneExpr::Push(P_VERTICAL));
		let batch_channels = expr!(LaneExpr::MulU32(batch, channels));
		let x_over = expr!(LaneExpr::GeU32(out_x, width));
		let y_over = expr!(LaneExpr::GeU32(out_y, height));
		let z_over = expr!(LaneExpr::GeU32(plane, batch_channels));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let flip_x = expr!(LaneExpr::NeqU32(horizontal, zero));
		let width_last = expr!(LaneExpr::SubU32(width, one));
		let mirrored_x = expr!(LaneExpr::SubU32(width_last, out_x));
		let source_x = expr!(LaneExpr::SelectU32 {
			cond: flip_x,
			if_true: mirrored_x,
			if_false: out_x,
		});
		let flip_y = expr!(LaneExpr::NeqU32(vertical, zero));
		let height_last = expr!(LaneExpr::SubU32(height, one));
		let mirrored_y = expr!(LaneExpr::SubU32(height_last, out_y));
		let source_y = expr!(LaneExpr::SelectU32 {
			cond: flip_y,
			if_true: mirrored_y,
			if_false: out_y,
		});
		let plane_base = expr!(LaneExpr::MulU32(plane, height));
		let source_row = expr!(LaneExpr::AddU32(plane_base, source_y));
		let source_line = expr!(LaneExpr::MulU32(source_row, width));
		let source_index = expr!(LaneExpr::AddU32(source_line, source_x));
		let out_row = expr!(LaneExpr::AddU32(plane_base, out_y));
		let out_line = expr!(LaneExpr::MulU32(out_row, width));
		let out_index = expr!(LaneExpr::AddU32(out_line, out_x));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: source_index,
		});
		let ops = vec![
			TileOp::GuardIf {
				cond: out_of_bounds,
			},
			TileOp::StoreScalar {
				binding: 1,
				address: out_index,
				value,
			},
		];
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
				"horizontal".into(),
				"vertical".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops,
		}
	}

	/// `image_pad` — the `pad` donor on the bordered-gather vocabulary
	/// (OA-CC-33). Donor ABI: two buffers `[input, output]`, push
	/// `[batch, channels, in_h, in_w, out_h, out_w, pad_top, pad_left,
	/// border_mode]` + f32 `border_value`, donor grid.
	pub(in crate::runtime) fn image_pad() -> Self {
		let mut lane_exprs = Vec::new();
		let frame = ImageGatherFrame::new(&mut lane_exprs);
		let pad_top = lane(&mut lane_exprs, LaneExpr::Push(6));
		let pad_left = lane(&mut lane_exprs, LaneExpr::Push(7));
		let border_mode = lane(&mut lane_exprs, LaneExpr::Push(8));
		let border_value = lane(&mut lane_exprs, LaneExpr::PushF32(0));
		let value = lane(
			&mut lane_exprs,
			LaneExpr::PadFetchF32 {
				binding: 0,
				base: frame.plane_base,
				x: frame.out_x,
				y: frame.out_y,
				pad_left,
				pad_top,
				width: frame.in_width,
				height: frame.in_height,
				mode: border_mode,
				value: border_value,
			},
		);
		frame.program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"pad_top".into(),
				"pad_left".into(),
				"border_mode".into(),
			],
			value,
			lane_exprs,
		)
	}

	/// `image_warp` — the `warp_affine`/`warp_perspective` donors on the
	/// bordered-gather vocabulary (OA-CC-33). Donor ABI: three buffers
	/// `[input, transform, output]`, push `[batch, channels, in_h, in_w,
	/// out_h, out_w, interp, border_mode]` + f32 `border_value`, donor
	/// grid.
	pub(in crate::runtime) fn image_warp(perspective: bool) -> Self {
		let mut lane_exprs = Vec::new();
		let frame = ImageGatherFrame::new(&mut lane_exprs);
		let interp = lane(&mut lane_exprs, LaneExpr::Push(6));
		let border_mode = lane(&mut lane_exprs, LaneExpr::Push(7));
		let border_value = lane(&mut lane_exprs, LaneExpr::PushF32(0));
		let value = lane(
			&mut lane_exprs,
			LaneExpr::WarpF32 {
				input: 0,
				transform: 1,
				base: frame.plane_base,
				x: frame.out_x,
				y: frame.out_y,
				width: frame.in_width,
				height: frame.in_height,
				interp,
				mode: border_mode,
				value: border_value,
				perspective,
			},
		);
		frame.program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"interpolation".into(),
				"border_mode".into(),
			],
			value,
			lane_exprs,
		)
	}

	/// `image_remap` — the `remap` donor on the bordered-gather
	/// vocabulary (OA-CC-33). Donor ABI: three buffers `[input, map,
	/// output]`, push `[batch, channels, in_h, in_w, out_h, out_w,
	/// interp, border_mode, map_batch]` + f32 `border_value`, donor
	/// grid.
	pub(in crate::runtime) fn image_remap() -> Self {
		let mut lane_exprs = Vec::new();
		let frame = ImageGatherFrame::new(&mut lane_exprs);
		let interp = lane(&mut lane_exprs, LaneExpr::Push(6));
		let border_mode = lane(&mut lane_exprs, LaneExpr::Push(7));
		let map_batch = lane(&mut lane_exprs, LaneExpr::Push(8));
		let border_value = lane(&mut lane_exprs, LaneExpr::PushF32(0));
		let batch = lane(
			&mut lane_exprs,
			LaneExpr::DivU32(frame.plane, frame.channels),
		);
		let value = lane(
			&mut lane_exprs,
			LaneExpr::RemapF32 {
				input: 0,
				map: 1,
				base: frame.plane_base,
				x: frame.out_x,
				y: frame.out_y,
				batch,
				width: frame.in_width,
				height: frame.in_height,
				out_width: frame.out_width,
				out_height: frame.out_height,
				interp,
				mode: border_mode,
				map_batch,
				value: border_value,
			},
		);
		frame.program(
			vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"interpolation".into(),
				"border_mode".into(),
				"map_batch".into(),
			],
			value,
			lane_exprs,
		)
	}

	/// `image_crop` — the `crop`/`center_crop` donors as a pure index
	/// remap (OA-CC-33): `source = (plane*in_h + crop_y + y)*in_w +
	/// crop_x + x`. Donor ABI: two buffers, push `[batch, channels,
	/// in_h, in_w, crop_x, crop_y, out_h, out_w]`, donor grid.
	pub(in crate::runtime) fn image_crop() -> Self {
		let mut lane_exprs = Vec::new();
		let frame = ImageGatherFrame::new_out_at(&mut lane_exprs, 6, 7);
		let crop_x = lane(&mut lane_exprs, LaneExpr::Push(4));
		let crop_y = lane(&mut lane_exprs, LaneExpr::Push(5));
		let source_x = lane(&mut lane_exprs, LaneExpr::AddU32(crop_x, frame.out_x));
		let source_y = lane(&mut lane_exprs, LaneExpr::AddU32(crop_y, frame.out_y));
		let source_line = lane(&mut lane_exprs, LaneExpr::MulU32(source_y, frame.in_width));
		let source_row = lane(
			&mut lane_exprs,
			LaneExpr::AddU32(frame.plane_base, source_line),
		);
		let source_index = lane(&mut lane_exprs, LaneExpr::AddU32(source_row, source_x));
		let value = lane(
			&mut lane_exprs,
			LaneExpr::LoadF32 {
				binding: 0,
				address: source_index,
			},
		);
		let mut program = frame.program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"crop_x".into(),
				"crop_y".into(),
				"output_height".into(),
				"output_width".into(),
			],
			value,
			lane_exprs,
		);
		program.push_f32_scalars.clear();
		program
	}

	/// `image_rotate` — the `rotate` donor as a pure index remap
	/// (OA-CC-33): 90/180/270 selects over the output coordinates.
	/// Donor ABI: two buffers, push `[batch, channels, in_h, in_w,
	/// out_h, out_w, degrees]`, donor grid.
	pub(in crate::runtime) fn image_rotate() -> Self {
		let mut lane_exprs = Vec::new();
		let frame = ImageGatherFrame::new(&mut lane_exprs);
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let degrees = expr!(LaneExpr::Push(6));
		let one = expr!(LaneExpr::ConstU32(1));
		let ninety = expr!(LaneExpr::ConstU32(90));
		let one_eighty = expr!(LaneExpr::ConstU32(180));
		let two_seventy = expr!(LaneExpr::ConstU32(270));
		let w_last = expr!(LaneExpr::SubU32(frame.in_width, one));
		let h_last = expr!(LaneExpr::SubU32(frame.in_height, one));
		let flip_x = expr!(LaneExpr::SubU32(w_last, frame.out_x));
		let flip_y = expr!(LaneExpr::SubU32(h_last, frame.out_y));
		let mirror_y_on_x = expr!(LaneExpr::SubU32(h_last, frame.out_x));
		let mirror_x_on_y = expr!(LaneExpr::SubU32(w_last, frame.out_y));
		let d90 = expr!(LaneExpr::EqU32(degrees, ninety));
		let d180 = expr!(LaneExpr::EqU32(degrees, one_eighty));
		let d270 = expr!(LaneExpr::EqU32(degrees, two_seventy));
		let sx_180_270_0 = expr!(LaneExpr::SelectU32 {
			cond: d270,
			if_true: mirror_x_on_y,
			if_false: frame.out_x,
		});
		let sx_180 = expr!(LaneExpr::SelectU32 {
			cond: d180,
			if_true: flip_x,
			if_false: sx_180_270_0,
		});
		let source_x = expr!(LaneExpr::SelectU32 {
			cond: d90,
			if_true: frame.out_y,
			if_false: sx_180,
		});
		let sy_180_270_0 = expr!(LaneExpr::SelectU32 {
			cond: d270,
			if_true: frame.out_x,
			if_false: frame.out_y,
		});
		let sy_180 = expr!(LaneExpr::SelectU32 {
			cond: d180,
			if_true: flip_y,
			if_false: sy_180_270_0,
		});
		let source_y = expr!(LaneExpr::SelectU32 {
			cond: d90,
			if_true: mirror_y_on_x,
			if_false: sy_180,
		});
		let source_line = expr!(LaneExpr::MulU32(source_y, frame.in_width));
		let source_row = expr!(LaneExpr::AddU32(frame.plane_base, source_line));
		let source_index = expr!(LaneExpr::AddU32(source_row, source_x));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: source_index,
		});
		let mut program = frame.program(
			vec![Binding::Input(0), Binding::Output(0)],
			vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
				"degrees".into(),
			],
			value,
			lane_exprs,
		);
		program.push_f32_scalars.clear();
		program
	}

	/// Coordinate/fetch prelude shared by the `filter.slang`
	/// bordered-window donors (OA-CC-33): the 16×16 output tile, plane
	/// offset, bounds guard, and border-mode/kernel-extent pushes. The
	/// filter push layout is `[batch, channels, height, width,
	/// operation, border_mode, kernel_width, kernel_height,
	/// kernel_size, dx, dy]` + `p0..p3, border_value`.
	fn filter_prelude(lane_exprs: &mut Vec<LaneExpr>) -> FilterPrelude {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let plane = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let two = expr!(LaneExpr::ConstU32(2));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(0));
		let channels = expr!(LaneExpr::Push(1));
		let height = expr!(LaneExpr::Push(2));
		let width = expr!(LaneExpr::Push(3));
		let mode = expr!(LaneExpr::Push(5));
		let kernel_width = expr!(LaneExpr::Push(6));
		let kernel_height = expr!(LaneExpr::Push(7));
		let size = expr!(LaneExpr::Push(8));
		let border_value = expr!(LaneExpr::PushF32(4));
		let batch_channels = expr!(LaneExpr::MulU32(batch, channels));
		let x_over = expr!(LaneExpr::GeU32(out_x, width));
		let y_over = expr!(LaneExpr::GeU32(out_y, height));
		let z_over = expr!(LaneExpr::GeU32(plane, batch_channels));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let plane_rows = expr!(LaneExpr::MulU32(plane, height));
		let plane_base = expr!(LaneExpr::MulU32(plane_rows, width));
		let radius_x = expr!(LaneExpr::DivU32(kernel_width, two));
		let radius_y = expr!(LaneExpr::DivU32(kernel_height, two));
		let kernel_extent = expr!(LaneExpr::MulU32(kernel_width, kernel_height));
		let size_radius = expr!(LaneExpr::DivU32(size, two));
		let size_extent = expr!(LaneExpr::MulU32(size, size));
		let out_row = expr!(LaneExpr::MulU32(out_y, width));
		let out_line = expr!(LaneExpr::AddU32(plane_base, out_row));
		let out_index = expr!(LaneExpr::AddU32(out_line, out_x));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let zero = expr!(LaneExpr::ConstF32(0));
		let flt_max = expr!(LaneExpr::ConstF32(0x7F7F_FFFF));
		let neg_flt_max = expr!(LaneExpr::ConstF32(0xFF7F_FFFF));
		FilterPrelude {
			out_of_bounds,
			zero_u32,
			plane,
			plane_base,
			out_x,
			out_y,
			out_index,
			height,
			width,
			mode,
			kernel_width,
			kernel_extent,
			size,
			size_radius,
			size_extent,
			radius_x,
			radius_y,
			border_value,
			zero,
			flt_max,
			neg_flt_max,
		}
	}

	/// The `filter.slang` donor push layout.
	const FILTER_PUSH_SCALARS: &[&str] = &[
		"batch_size",
		"channels",
		"height",
		"width",
		"operation",
		"border_mode",
		"kernel_width",
		"kernel_height",
		"kernel_size",
		"dx",
		"dy",
	];

	/// `convolve_2d` — the filter donor's runtime-K bordered window
	/// conv re-emitted as a serial per-lane fold (OA-CC-33 filter
	/// anchor): each output texel accumulates `image_sample_bordered`
	/// taps weighted by the kernel buffer over `kernel_height ×
	/// kernel_width` inductions (`ky = i / kw`, `kx = i % kw`, the
	/// donor's loop order). Tap coordinates are u32 expressions whose
	/// wraparound reproduces the donor's signed offsets. Donor ABI:
	/// `[input, kernel_x, kernel_y, output]` bindings, 11 u32 +
	/// `p0..p3, border_value` f32 pushes, donor grid.
	pub(in crate::runtime) fn image_convolve() -> Self {
		const P_BATCH: usize = 0;
		const P_CHANNELS: usize = 1;
		const P_HEIGHT: usize = 2;
		const P_WIDTH: usize = 3;
		const P_BORDER_MODE: usize = 5;
		const P_KERNEL_WIDTH: usize = 6;
		const P_KERNEL_HEIGHT: usize = 7;
		const P_BORDER_VALUE: usize = 4;
		use LaneExpr as E;
		let lane_exprs = vec![
			/* 0 */ E::Push(P_BATCH),
			/* 1 */ E::Push(P_CHANNELS),
			/* 2 */ E::Push(P_HEIGHT),
			/* 3 */ E::Push(P_WIDTH),
			/* 4 */ E::Push(P_BORDER_MODE),
			/* 5 */ E::Push(P_KERNEL_WIDTH),
			/* 6 */ E::Push(P_KERNEL_HEIGHT),
			/* 7 */ E::PushF32(P_BORDER_VALUE),
			/* 8 */ E::GroupX,
			/* 9 */ E::GroupY,
			/* 10 */ E::GroupZ,
			/* 11 */ E::Lane,
			/* 12 */ E::ConstU32(16),
			/* 13 */ E::DivU32(11, 12), // thread_row
			/* 14 */ E::ModU32(11, 12), // thread_column
			/* 15 */ E::MulU32(8, 12), // tile_x
			/* 16 */ E::MulU32(9, 12), // tile_y
			/* 17 */ E::AddU32(15, 14), // out_x
			/* 18 */ E::AddU32(16, 13), // out_y
			/* 19 */ E::MulU32(0, 1), // batch*channels
			/* 20 */ E::GeU32(17, 3), // x_over
			/* 21 */ E::GeU32(18, 2), // y_over
			/* 22 */ E::GeU32(10, 19), // z_over
			/* 23 */ E::OrBool(20, 21),
			/* 24 */ E::OrBool(23, 22), // out_of_bounds
			/* 25 */ E::MulU32(10, 2), // plane * height
			/* 26 */ E::MulU32(25, 3), // plane_base
			/* 27 */ E::ConstU32(2),
			/* 28 */ E::DivU32(5, 27), // radius_x
			/* 29 */ E::DivU32(6, 27), // radius_y
			/* 30 */ E::Induction, // flat tap index
			/* 31 */ E::DivU32(30, 5), // ky
			/* 32 */ E::ModU32(30, 5), // kx
			/* 33 */ E::AddU32(17, 32), // out_x + kx
			/* 34 */ E::SubU32(33, 28), // tap_x (wraps negative)
			/* 35 */ E::AddU32(18, 31), // out_y + ky
			/* 36 */ E::SubU32(35, 29), // tap_y
			/* 37 */
			E::BorderedFetchF32 {
				binding: 0,
				base: 26,
				x: 34,
				y: 36,
				width: 3,
				height: 2,
				mode: 4,
				value: 7,
			}, // tap
			/* 38 */ E::LoadF32 {
				binding: 1,
				address: 30,
			}, // coefficient
			/* 39 */ E::MulF32(37, 38), // weighted tap
			/* 40 */ E::ConstF32(0), // 0.0
			/* 41 */ E::MulU32(5, 6), // kernel_extent
			/* 42 */ E::MulU32(18, 3), // out_y * width
			/* 43 */ E::AddU32(26, 42), // plane_base + out_y * width
			/* 44 */ E::AddU32(43, 17), // out_index
			/* 45 */ E::RegisterF32 { tile: 0, index: 0 }, // conv acc
		];
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
				"operation".into(),
				"border_mode".into(),
				"kernel_width".into(),
				"kernel_height".into(),
				"kernel_size".into(),
				"dx".into(),
				"dy".into(),
			],
			push_f32_scalars: vec![
				"p0".into(),
				"p1".into(),
				"p2".into(),
				"p3".into(),
				"border_value".into(),
			],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			ops: vec![
				TileOp::GuardIf { cond: 24 },
				TileOp::LaneFold {
					acc: 0,
					init: 40,
					extent: Extent::Expr(41),
					op: LaneReduce::Sum,
					element: 39,
					serial: true,
				},
				TileOp::StoreScalar {
					binding: 3,
					address: 44,
					value: 45,
				},
			],
		}
	}

	/// `average_blur` — the filter donor's uniform bordered sum over
	/// `kernel_height × kernel_width` taps, divided by the extent
	/// (donor operation 2). Donor ABI: `[input, kernel_x, kernel_y,
	/// output]` (kernel bindings unused), filter push layout.
	pub(in crate::runtime) fn image_average_blur() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let i = expr!(LaneExpr::Induction);
		let ky = expr!(LaneExpr::DivU32(i, frame.kernel_width));
		let kx = expr!(LaneExpr::ModU32(i, frame.kernel_width));
		let tap_xp = expr!(LaneExpr::AddU32(frame.out_x, kx));
		let tap_x = expr!(LaneExpr::SubU32(tap_xp, frame.radius_x));
		let tap_yp = expr!(LaneExpr::AddU32(frame.out_y, ky));
		let tap_y = expr!(LaneExpr::SubU32(tap_yp, frame.radius_y));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: tap_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let count = expr!(LaneExpr::F32FromU32(frame.kernel_extent));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let result = expr!(LaneExpr::DivF32(acc, count));
		let mut program = Self::filter_program(
			lane_exprs,
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			vec![TileOp::LaneFold {
				acc: 0,
				init: frame.zero,
				extent: Extent::Expr(frame.kernel_extent),
				op: LaneReduce::Sum,
				element: tap,
				serial: true,
			}],
			frame.out_of_bounds,
			frame.out_index,
			result,
		);
		program.bindings[1] = Binding::Input(1);
		program.bindings[2] = Binding::Input(2);
		program
	}

	/// `erode`/`dilate` — the `morphology_once` donor: a Min/Max fold
	/// over bordered taps with ±FLT_MAX init (operations 6/7).
	pub(in crate::runtime) fn image_morphology_once(dilate: bool) -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let i = expr!(LaneExpr::Induction);
		let ky = expr!(LaneExpr::DivU32(i, frame.kernel_width));
		let kx = expr!(LaneExpr::ModU32(i, frame.kernel_width));
		let tap_xp = expr!(LaneExpr::AddU32(frame.out_x, kx));
		let tap_x = expr!(LaneExpr::SubU32(tap_xp, frame.radius_x));
		let tap_yp = expr!(LaneExpr::AddU32(frame.out_y, ky));
		let tap_y = expr!(LaneExpr::SubU32(tap_yp, frame.radius_y));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: tap_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		Self::filter_program(
			lane_exprs,
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			vec![TileOp::LaneFold {
				acc: 0,
				init: if dilate {
					frame.neg_flt_max
				} else {
					frame.flt_max
				},
				extent: Extent::Expr(frame.kernel_extent),
				op: if dilate {
					LaneReduce::Max
				} else {
					LaneReduce::Min
				},
				element: tap,
				serial: true,
			}],
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// `morphology_gradient` — `dilate − erode` as two serial folds
	/// (operation 10).
	pub(in crate::runtime) fn image_morphology_gradient() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let i = expr!(LaneExpr::Induction);
		let ky = expr!(LaneExpr::DivU32(i, frame.kernel_width));
		let kx = expr!(LaneExpr::ModU32(i, frame.kernel_width));
		let tap_xp = expr!(LaneExpr::AddU32(frame.out_x, kx));
		let tap_x = expr!(LaneExpr::SubU32(tap_xp, frame.radius_x));
		let tap_yp = expr!(LaneExpr::AddU32(frame.out_y, ky));
		let tap_y = expr!(LaneExpr::SubU32(tap_yp, frame.radius_y));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: tap_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let dilated = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let eroded = expr!(LaneExpr::RegisterF32 { tile: 1, index: 0 });
		let result = expr!(LaneExpr::SubF32(dilated, eroded));
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			vec![
				TileOp::LaneFold {
					acc: 0,
					init: frame.neg_flt_max,
					extent: Extent::Expr(frame.kernel_extent),
					op: LaneReduce::Max,
					element: tap,
					serial: true,
				},
				TileOp::LaneFold {
					acc: 1,
					init: frame.flt_max,
					extent: Extent::Expr(frame.kernel_extent),
					op: LaneReduce::Min,
					element: tap,
					serial: true,
				},
			],
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// `separable_convolve_2d` — the donor's nested fold (operation
	/// 1): an outer `k` loop folds `border_coordinate(y + k − ry)`
	/// through the border mode; each surviving row runs an inner
	/// horizontal weighted fold against `kernel_x`, then the term
	/// accumulates `intermediate × kernel_y[k]` into the output.
	pub(in crate::runtime) fn image_separable_convolve() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let k = expr!(LaneExpr::LoopVar(2));
		let source_y = expr!(LaneExpr::AddU32(frame.out_y, k));
		let shifted_y = expr!(LaneExpr::SubU32(source_y, frame.radius_y));
		let mapped_y = expr!(LaneExpr::BorderCoordU32 {
			coordinate: shifted_y,
			extent: frame.height,
			mode: frame.mode,
		});
		let neg_threshold = expr!(LaneExpr::ConstU32(0x8000_0000));
		let mapped_neg = expr!(LaneExpr::GeU32(mapped_y, neg_threshold));
		let i = expr!(LaneExpr::Induction);
		let tap_xp = expr!(LaneExpr::AddU32(frame.out_x, i));
		let tap_x = expr!(LaneExpr::SubU32(tap_xp, frame.radius_x));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: mapped_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let coeff_x = expr!(LaneExpr::LoadF32 {
			binding: 1,
			address: i,
		});
		let inner_term = expr!(LaneExpr::MulF32(tap, coeff_x));
		let inner_acc = expr!(LaneExpr::RegisterF32 { tile: 1, index: 0 });
		let intermediate = expr!(LaneExpr::SelectF32 {
			cond: mapped_neg,
			if_true: frame.border_value,
			if_false: inner_acc,
		});
		let coeff_y = expr!(LaneExpr::LoadF32 {
			binding: 2,
			address: k,
		});
		let term = expr!(LaneExpr::MulF32(intermediate, coeff_y));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let updated = expr!(LaneExpr::AddF32(acc, term));
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			vec![
				TileOp::AssignScalar {
					tile: 0,
					index: 0,
					value: frame.zero,
				},
				TileOp::LaneLoopRange {
					start: frame.zero_u32,
					step: 1,
					extent: Extent::Push(7),
					body: vec![
						TileOp::LaneFold {
							acc: 1,
							init: frame.zero,
							extent: Extent::Push(6),
							op: LaneReduce::Sum,
							element: inner_term,
							serial: true,
						},
						TileOp::AssignScalar {
							tile: 0,
							index: 0,
							value: updated,
						},
					],
				},
			],
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// The `morphology_twice` body ops shared by open/close and the
	/// hats: emits the mapped-coordinate and inner-fold expressions
	/// into `lane_exprs` and returns `[init, loop]` ops leaving the
	/// composed value in register tile 0.
	fn morphology_twice_body(
		lane_exprs: &mut Vec<LaneExpr>,
		frame: &FilterPrelude,
		first_dilate: bool,
	) -> Vec<TileOp> {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let outer = expr!(LaneExpr::LoopVar(2));
		let ky = expr!(LaneExpr::DivU32(outer, frame.kernel_width));
		let kx = expr!(LaneExpr::ModU32(outer, frame.kernel_width));
		let source_x = expr!(LaneExpr::AddU32(frame.out_x, kx));
		let shifted_x = expr!(LaneExpr::SubU32(source_x, frame.radius_x));
		let source_y = expr!(LaneExpr::AddU32(frame.out_y, ky));
		let shifted_y = expr!(LaneExpr::SubU32(source_y, frame.radius_y));
		let mapped_x = expr!(LaneExpr::BorderCoordU32 {
			coordinate: shifted_x,
			extent: frame.width,
			mode: frame.mode,
		});
		let mapped_y = expr!(LaneExpr::BorderCoordU32 {
			coordinate: shifted_y,
			extent: frame.height,
			mode: frame.mode,
		});
		let neg_threshold = expr!(LaneExpr::ConstU32(0x8000_0000));
		let neg_x = expr!(LaneExpr::GeU32(mapped_x, neg_threshold));
		let neg_y = expr!(LaneExpr::GeU32(mapped_y, neg_threshold));
		let mapped_neg = expr!(LaneExpr::OrBool(neg_x, neg_y));
		let i = expr!(LaneExpr::Induction);
		let inner_ky = expr!(LaneExpr::DivU32(i, frame.kernel_width));
		let inner_kx = expr!(LaneExpr::ModU32(i, frame.kernel_width));
		let inner_xp = expr!(LaneExpr::AddU32(mapped_x, inner_kx));
		let inner_x = expr!(LaneExpr::SubU32(inner_xp, frame.radius_x));
		let inner_yp = expr!(LaneExpr::AddU32(mapped_y, inner_ky));
		let inner_y = expr!(LaneExpr::SubU32(inner_yp, frame.radius_y));
		let inner_tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: inner_x,
			y: inner_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let inner_acc = expr!(LaneExpr::RegisterF32 { tile: 1, index: 0 });
		let value = expr!(LaneExpr::SelectF32 {
			cond: mapped_neg,
			if_true: frame.border_value,
			if_false: inner_acc,
		});
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let second_dilate = !first_dilate;
		let updated = if second_dilate {
			expr!(LaneExpr::MaxF32(acc, value))
		} else {
			expr!(LaneExpr::MinF32(acc, value))
		};
		vec![
			TileOp::AssignScalar {
				tile: 0,
				index: 0,
				value: if second_dilate {
					frame.neg_flt_max
				} else {
					frame.flt_max
				},
			},
			TileOp::LaneLoopRange {
				start: frame.zero_u32,
				step: 1,
				extent: Extent::Expr(frame.kernel_extent),
				body: vec![
					TileOp::LaneFold {
						acc: 1,
						init: if first_dilate {
							frame.neg_flt_max
						} else {
							frame.flt_max
						},
						extent: Extent::Expr(frame.kernel_extent),
						op: if first_dilate {
							LaneReduce::Max
						} else {
							LaneReduce::Min
						},
						element: inner_tap,
						serial: true,
					},
					TileOp::AssignScalar {
						tile: 0,
						index: 0,
						value: updated,
					},
				],
			},
		]
	}

	/// `morphology_open`/`morphology_close` — the `morphology_twice`
	/// donor (operations 8/9): an outer Min/Max fold over per-tap
	/// `morphology_once` results, with constant-mode coordinate misses
	/// substituting the border value.
	pub(in crate::runtime) fn image_morphology_twice(first_dilate: bool) -> Self {
		let mut lane_exprs = Vec::new();
		let frame = Self::filter_prelude(&mut lane_exprs);
		let body = Self::morphology_twice_body(&mut lane_exprs, &frame, first_dilate);
		let acc = lane(&mut lane_exprs, LaneExpr::RegisterF32 { tile: 0, index: 0 });
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			body,
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// `morphology_top_hat`/`morphology_black_hat` — `center − open`
	/// or `close − center` (operations 16/17), reusing the
	/// `morphology_twice` fold structure.
	pub(in crate::runtime) fn image_morphology_hat(top: bool) -> Self {
		let mut lane_exprs = Vec::new();
		let frame = Self::filter_prelude(&mut lane_exprs);
		let body = Self::morphology_twice_body(&mut lane_exprs, &frame, !top);
		let center = lane(
			&mut lane_exprs,
			LaneExpr::BorderedFetchF32 {
				binding: 0,
				base: frame.plane_base,
				x: frame.out_x,
				y: frame.out_y,
				width: frame.width,
				height: frame.height,
				mode: frame.mode,
				value: frame.border_value,
			},
		);
		let twice = lane(&mut lane_exprs, LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let result = if top {
			lane(&mut lane_exprs, LaneExpr::SubF32(center, twice))
		} else {
			lane(&mut lane_exprs, LaneExpr::SubF32(twice, center))
		};
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			body,
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// The `fixed_linear` donor's 3×3 coefficient geometry: the serial
	/// fold index decomposed as `im3 = i % 3` (kx + 1) and `id3 = i / 3`
	/// (ky + 1), the bordered tap at `out + (im3 − 1, id3 − 1)`, and
	/// the nine-element extent. Returns `(tap, im3, id3, nine)`.
	fn fixed_linear_tap(
		lane_exprs: &mut Vec<LaneExpr>,
		frame: &FilterPrelude,
	) -> (usize, usize, usize, usize) {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let i = expr!(LaneExpr::Induction);
		let three = expr!(LaneExpr::ConstU32(3));
		let one = expr!(LaneExpr::ConstU32(1));
		let id3 = expr!(LaneExpr::DivU32(i, three));
		let im3 = expr!(LaneExpr::ModU32(i, three));
		let tap_xp = expr!(LaneExpr::AddU32(frame.out_x, im3));
		let tap_x = expr!(LaneExpr::SubU32(tap_xp, one));
		let tap_yp = expr!(LaneExpr::AddU32(frame.out_y, id3));
		let tap_y = expr!(LaneExpr::SubU32(tap_yp, one));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: tap_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let nine = expr!(LaneExpr::ConstU32(9));
		(tap, im3, id3, nine)
	}

	/// A `SelectF32` sign lookup over one ternary axis index `axis`
	/// (`0 → -1`, `2 → +1`, otherwise `0`), matching the donor's
	/// `(k == -1 ? -1 : k == 1 ? 1 : 0)` chain.
	fn signed_axis(lane_exprs: &mut Vec<LaneExpr>, axis: usize) -> usize {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let two_u32 = expr!(LaneExpr::ConstU32(2));
		let neg_one = expr!(LaneExpr::ConstF32(0xBF80_0000));
		let pos_one = expr!(LaneExpr::ConstF32(0x3F80_0000));
		let zero = expr!(LaneExpr::ConstF32(0));
		let is_neg = expr!(LaneExpr::EqU32(axis, zero_u32));
		let is_pos = expr!(LaneExpr::EqU32(axis, two_u32));
		let pos_or_zero = expr!(LaneExpr::SelectF32 {
			cond: is_pos,
			if_true: pos_one,
			if_false: zero,
		});
		expr!(LaneExpr::SelectF32 {
			cond: is_neg,
			if_true: neg_one,
			if_false: pos_or_zero,
		})
	}

	/// A `SelectF32` weight lookup over one ternary axis index:
	/// `axis == 1 → center_weight`, otherwise `edge_weight`.
	fn centered_axis(
		lane_exprs: &mut Vec<LaneExpr>,
		axis: usize,
		center_weight: usize,
		edge_weight: usize,
	) -> usize {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let one_u32 = expr!(LaneExpr::ConstU32(1));
		let centered = expr!(LaneExpr::EqU32(axis, one_u32));
		expr!(LaneExpr::SelectF32 {
			cond: centered,
			if_true: center_weight,
			if_false: edge_weight,
		})
	}

	/// `sobel`/`scharr` — `fixed_linear` operations 3/4: a serial 3×3
	/// fold whose coefficient is `sign × weight` with the sign axis
	/// selected by `push.dx` and the magnitudes `(2, 1)` or `(10, 3)`.
	pub(in crate::runtime) fn image_directional_linear(center: u32, edge: u32) -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, im3, id3, nine) = Self::fixed_linear_tap(&mut lane_exprs, &frame);
		let center_w = expr!(LaneExpr::ConstF32(center));
		let edge_w = expr!(LaneExpr::ConstF32(edge));
		let sign_kx = Self::signed_axis(&mut lane_exprs, im3);
		let sign_ky = Self::signed_axis(&mut lane_exprs, id3);
		let w_ky = Self::centered_axis(&mut lane_exprs, id3, center_w, edge_w);
		let w_kx = Self::centered_axis(&mut lane_exprs, im3, center_w, edge_w);
		let one_u32 = expr!(LaneExpr::ConstU32(1));
		let dx = expr!(LaneExpr::Push(9));
		let is_dx = expr!(LaneExpr::EqU32(dx, one_u32));
		let x_coeff = expr!(LaneExpr::MulF32(sign_kx, w_ky));
		let y_coeff = expr!(LaneExpr::MulF32(sign_ky, w_kx));
		let coeff = expr!(LaneExpr::SelectF32 {
			cond: is_dx,
			if_true: x_coeff,
			if_false: y_coeff,
		});
		let element = expr!(LaneExpr::MulF32(tap, coeff));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		Self::filter_program(
			lane_exprs,
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			vec![TileOp::LaneFold {
				acc: 0,
				init: frame.zero,
				extent: Extent::Expr(nine),
				op: LaneReduce::Sum,
				element,
				serial: true,
			}],
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// `laplacian` — `fixed_linear` operation 5: `−4` at the center
	/// tap, `+1` on the four edge neighbors, `0` on corners.
	pub(in crate::runtime) fn image_laplacian() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, im3, id3, nine) = Self::fixed_linear_tap(&mut lane_exprs, &frame);
		let one_u32 = expr!(LaneExpr::ConstU32(1));
		let x_center = expr!(LaneExpr::EqU32(im3, one_u32));
		let y_center = expr!(LaneExpr::EqU32(id3, one_u32));
		let centered = expr!(LaneExpr::AndBool(x_center, y_center));
		let edge = expr!(LaneExpr::OrBool(x_center, y_center));
		let neg_four = expr!(LaneExpr::ConstF32(0xC080_0000));
		let pos_one = expr!(LaneExpr::ConstF32(0x3F80_0000));
		let zero = expr!(LaneExpr::ConstF32(0));
		let edge_or_zero = expr!(LaneExpr::SelectF32 {
			cond: edge,
			if_true: pos_one,
			if_false: zero,
		});
		let coeff = expr!(LaneExpr::SelectF32 {
			cond: centered,
			if_true: neg_four,
			if_false: edge_or_zero,
		});
		let element = expr!(LaneExpr::MulF32(tap, coeff));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		Self::filter_program(
			lane_exprs,
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			vec![TileOp::LaneFold {
				acc: 0,
				init: frame.zero,
				extent: Extent::Expr(nine),
				op: LaneReduce::Sum,
				element,
				serial: true,
			}],
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// `sharpen` — `fixed_linear` operation 12: `1 + 4·p0` at the
	/// center, `−p0` on edge neighbors, `0` on corners.
	pub(in crate::runtime) fn image_sharpen() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, im3, id3, nine) = Self::fixed_linear_tap(&mut lane_exprs, &frame);
		let one_u32 = expr!(LaneExpr::ConstU32(1));
		let x_center = expr!(LaneExpr::EqU32(im3, one_u32));
		let y_center = expr!(LaneExpr::EqU32(id3, one_u32));
		let centered = expr!(LaneExpr::AndBool(x_center, y_center));
		let edge = expr!(LaneExpr::OrBool(x_center, y_center));
		let amount = expr!(LaneExpr::PushF32(0));
		let four = expr!(LaneExpr::ConstF32(0x4080_0000));
		let one = expr!(LaneExpr::ConstF32(0x3F80_0000));
		let scaled = expr!(LaneExpr::MulF32(four, amount));
		let center_w = expr!(LaneExpr::AddF32(one, scaled));
		let neg_amount = expr!(LaneExpr::SubF32(frame.zero, amount));
		let zero = expr!(LaneExpr::ConstF32(0));
		let edge_or_zero = expr!(LaneExpr::SelectF32 {
			cond: edge,
			if_true: neg_amount,
			if_false: zero,
		});
		let coeff = expr!(LaneExpr::SelectF32 {
			cond: centered,
			if_true: center_w,
			if_false: edge_or_zero,
		});
		let element = expr!(LaneExpr::MulF32(tap, coeff));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		Self::filter_program(
			lane_exprs,
			vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			vec![TileOp::LaneFold {
				acc: 0,
				init: frame.zero,
				extent: Extent::Expr(nine),
				op: LaneReduce::Sum,
				element,
				serial: true,
			}],
			frame.out_of_bounds,
			frame.out_index,
			acc,
		)
	}

	/// The `gaussian`/`neighborhood` per-tap geometry: bordered fetch
	/// at `out + (i % size − size/2, i / size − size/2)` and the float
	/// squared offset distance. Wrapped-u32 squaring keeps each
	/// signed offset's square exact below 2³¹, so `float(u32)` of the
	/// sum matches the donor's `float(ox² + oy²)` bit for bit.
	fn neighborhood_tap(lane_exprs: &mut Vec<LaneExpr>, frame: &FilterPrelude) -> (usize, usize) {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let i = expr!(LaneExpr::Induction);
		let ky = expr!(LaneExpr::DivU32(i, frame.size));
		let kx = expr!(LaneExpr::ModU32(i, frame.size));
		let oy = expr!(LaneExpr::SubU32(ky, frame.size_radius));
		let ox = expr!(LaneExpr::SubU32(kx, frame.size_radius));
		let tap_x = expr!(LaneExpr::AddU32(frame.out_x, ox));
		let tap_y = expr!(LaneExpr::AddU32(frame.out_y, oy));
		let tap = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: tap_x,
			y: tap_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let ox_sq = expr!(LaneExpr::MulU32(ox, ox));
		let oy_sq = expr!(LaneExpr::MulU32(oy, oy));
		let dist_u32 = expr!(LaneExpr::AddU32(ox_sq, oy_sq));
		let dist = expr!(LaneExpr::F32FromU32(dist_u32));
		(tap, dist)
	}

	/// The two serial `weighted`/`weight_sum` folds shared by the
	/// `gaussian` and `neighborhood` donors. Splitting the donor's
	/// interleaved accumulators into sequential folds preserves each
	/// accumulator's add sequence, so the results stay bitwise
	/// identical. Returns `(weighted, weight_sum)` lane expressions
	/// reading register tiles 0 and 1.
	fn weighted_folds(
		lane_exprs: &mut Vec<LaneExpr>,
		frame: &FilterPrelude,
		element: usize,
		weight: usize,
	) -> (usize, usize, Vec<TileOp>) {
		let weighted = lane(lane_exprs, LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let weight_sum = lane(lane_exprs, LaneExpr::RegisterF32 { tile: 1, index: 0 });
		let ops = vec![
			TileOp::LaneFold {
				acc: 0,
				init: frame.zero,
				extent: Extent::Expr(frame.size_extent),
				op: LaneReduce::Sum,
				element,
				serial: true,
			},
			TileOp::LaneFold {
				acc: 1,
				init: frame.zero,
				extent: Extent::Expr(frame.size_extent),
				op: LaneReduce::Sum,
				element: weight,
				serial: true,
			},
		];
		(weighted, weight_sum, ops)
	}

	/// `exp(0 − dist / (2·σ²))` over the f32 push `sigma_index`,
	/// matching the donor's `-float(dist) / (2σ²)` operand order.
	fn gaussian_weight(
		lane_exprs: &mut Vec<LaneExpr>,
		frame: &FilterPrelude,
		dist: usize,
		sigma_index: usize,
	) -> usize {
		macro_rules! expr {
			($e:expr) => {
				lane(lane_exprs, $e)
			};
		}
		let two = expr!(LaneExpr::ConstF32(0x4000_0000));
		let sigma = expr!(LaneExpr::PushF32(sigma_index));
		let two_sigma = expr!(LaneExpr::MulF32(two, sigma));
		let denom = expr!(LaneExpr::MulF32(two_sigma, sigma));
		let ratio = expr!(LaneExpr::DivF32(dist, denom));
		let neg = expr!(LaneExpr::SubF32(frame.zero, ratio));
		expr!(LaneExpr::ExpF32(neg))
	}

	/// `gaussian_blur` — operation 11: weighted serial fold over
	/// `kernel_size²` taps normalized by `max(weight_sum, 1e-20)`.
	pub(in crate::runtime) fn image_gaussian() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, dist) = Self::neighborhood_tap(&mut lane_exprs, &frame);
		let weight = Self::gaussian_weight(&mut lane_exprs, &frame, dist, 0);
		let element = expr!(LaneExpr::MulF32(tap, weight));
		let (weighted, weight_sum, ops) =
			Self::weighted_folds(&mut lane_exprs, &frame, element, weight);
		let eps = expr!(LaneExpr::ConstF32(0x3586_37BD));
		let denom = expr!(LaneExpr::MaxF32(weight_sum, eps));
		let result = expr!(LaneExpr::DivF32(weighted, denom));
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			ops,
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// `bilateral_filter` — `neighborhood` operation 14: the space
	/// weight uses `p1`, the color term `p0` against the center tap.
	pub(in crate::runtime) fn image_bilateral() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, dist) = Self::neighborhood_tap(&mut lane_exprs, &frame);
		let center = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: frame.out_x,
			y: frame.out_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let color = expr!(LaneExpr::SubF32(tap, center));
		let color_sq = expr!(LaneExpr::MulF32(color, color));
		let two = expr!(LaneExpr::ConstF32(0x4000_0000));
		let sigma_space = expr!(LaneExpr::PushF32(1));
		let two_space = expr!(LaneExpr::MulF32(two, sigma_space));
		let space_denom = expr!(LaneExpr::MulF32(two_space, sigma_space));
		let space_ratio = expr!(LaneExpr::DivF32(dist, space_denom));
		let sigma_color = expr!(LaneExpr::PushF32(0));
		let two_color = expr!(LaneExpr::MulF32(two, sigma_color));
		let color_denom = expr!(LaneExpr::MulF32(two_color, sigma_color));
		let color_ratio = expr!(LaneExpr::DivF32(color_sq, color_denom));
		let neg_space = expr!(LaneExpr::SubF32(frame.zero, space_ratio));
		let exponent = expr!(LaneExpr::SubF32(neg_space, color_ratio));
		let weight = expr!(LaneExpr::ExpF32(exponent));
		let element = expr!(LaneExpr::MulF32(tap, weight));
		let (weighted, weight_sum, ops) =
			Self::weighted_folds(&mut lane_exprs, &frame, element, weight);
		let eps = expr!(LaneExpr::ConstF32(0x3586_37BD));
		let denom = expr!(LaneExpr::MaxF32(weight_sum, eps));
		let result = expr!(LaneExpr::DivF32(weighted, denom));
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			ops,
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// `unsharp_mask` — operation 15: `center + amount·(center −
	/// gaussian)` over the same weighted folds as `gaussian_blur`.
	pub(in crate::runtime) fn image_unsharp() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, dist) = Self::neighborhood_tap(&mut lane_exprs, &frame);
		let weight = Self::gaussian_weight(&mut lane_exprs, &frame, dist, 0);
		let element = expr!(LaneExpr::MulF32(tap, weight));
		let (weighted, weight_sum, ops) =
			Self::weighted_folds(&mut lane_exprs, &frame, element, weight);
		let eps = expr!(LaneExpr::ConstF32(0x3586_37BD));
		let denom = expr!(LaneExpr::MaxF32(weight_sum, eps));
		let blurred = expr!(LaneExpr::DivF32(weighted, denom));
		let center = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: frame.out_x,
			y: frame.out_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let amount = expr!(LaneExpr::PushF32(1));
		let diff = expr!(LaneExpr::SubF32(center, blurred));
		let scaled = expr!(LaneExpr::MulF32(amount, diff));
		let result = expr!(LaneExpr::AddF32(center, scaled));
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			ops,
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// `adaptive_threshold_mean`/`adaptive_threshold_gaussian` —
	/// `neighborhood` operations 18/19: a uniform or `p2`-σ weighted
	/// fold, then `center > filtered − c ? max_value : 0`.
	pub(in crate::runtime) fn image_adaptive_threshold(gaussian: bool) -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let (tap, dist) = Self::neighborhood_tap(&mut lane_exprs, &frame);
		let weight = if gaussian {
			Self::gaussian_weight(&mut lane_exprs, &frame, dist, 2)
		} else {
			expr!(LaneExpr::ConstF32(0x3F80_0000))
		};
		let element = expr!(LaneExpr::MulF32(tap, weight));
		let (weighted, weight_sum, ops) =
			Self::weighted_folds(&mut lane_exprs, &frame, element, weight);
		let eps = expr!(LaneExpr::ConstF32(0x3586_37BD));
		let denom = expr!(LaneExpr::MaxF32(weight_sum, eps));
		let filtered = expr!(LaneExpr::DivF32(weighted, denom));
		let center = expr!(LaneExpr::BorderedFetchF32 {
			binding: 0,
			base: frame.plane_base,
			x: frame.out_x,
			y: frame.out_y,
			width: frame.width,
			height: frame.height,
			mode: frame.mode,
			value: frame.border_value,
		});
		let c = expr!(LaneExpr::PushF32(0));
		let bound = expr!(LaneExpr::SubF32(filtered, c));
		let above = expr!(LaneExpr::GtF32(center, bound));
		let max_value = expr!(LaneExpr::PushF32(1));
		let result = expr!(LaneExpr::SelectF32 {
			cond: above,
			if_true: max_value,
			if_false: frame.zero,
		});
		Self::filter_program(
			lane_exprs,
			vec![
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			ops,
			frame.out_of_bounds,
			frame.out_index,
			result,
		)
	}

	/// `pointwise` — the parameterized flat pixel map (thresholds,
	/// clamp, invert, affine, gamma, solarize, posterize): one
	/// elementwise `LoadF32` → select → store over `element_count`
	/// lanes. `operation` selects the donor branch; the emitted
	/// expression reproduces the branch's math exactly.
	pub(in crate::runtime) fn image_pointwise(operation: u32) -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let wide = expr!(LaneExpr::ConstU32(256));
		let tile_base = expr!(LaneExpr::MulU32(group_x, wide));
		let lane_id = expr!(LaneExpr::Lane);
		let index = expr!(LaneExpr::AddU32(tile_base, lane_id));
		let elements = expr!(LaneExpr::Push(0));
		let out_of_bounds = expr!(LaneExpr::GeU32(index, elements));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: index,
		});
		let p0 = expr!(LaneExpr::PushF32(0));
		let p1 = expr!(LaneExpr::PushF32(1));
		let p2 = expr!(LaneExpr::PushF32(2));
		let zero = expr!(LaneExpr::ConstF32(0));
		let result = match operation {
			0 => {
				let above = expr!(LaneExpr::GtF32(value, p0));
				expr!(LaneExpr::SelectF32 {
					cond: above,
					if_true: p1,
					if_false: zero,
				})
			}
			1 => {
				let above = expr!(LaneExpr::GtF32(value, p0));
				expr!(LaneExpr::SelectF32 {
					cond: above,
					if_true: zero,
					if_false: p1,
				})
			}
			2 => expr!(LaneExpr::MinF32(value, p0)),
			3 => {
				let above = expr!(LaneExpr::GtF32(value, p0));
				expr!(LaneExpr::SelectF32 {
					cond: above,
					if_true: value,
					if_false: zero,
				})
			}
			4 => {
				let above = expr!(LaneExpr::GtF32(value, p0));
				expr!(LaneExpr::SelectF32 {
					cond: above,
					if_true: zero,
					if_false: value,
				})
			}
			5 => {
				let lower = expr!(LaneExpr::GeF32(value, p0));
				let upper = expr!(LaneExpr::GeF32(p1, value));
				let inside = expr!(LaneExpr::AndBool(lower, upper));
				expr!(LaneExpr::SelectF32 {
					cond: inside,
					if_true: p2,
					if_false: zero,
				})
			}
			6 => {
				let lo = expr!(LaneExpr::MaxF32(value, p0));
				expr!(LaneExpr::MinF32(lo, p1))
			}
			7 => expr!(LaneExpr::SubF32(p0, value)),
			8 => {
				let scaled = expr!(LaneExpr::MulF32(value, p0));
				expr!(LaneExpr::AddF32(scaled, p1))
			}
			9 => {
				let base = expr!(LaneExpr::MaxF32(value, zero));
				let raised = expr!(LaneExpr::PowF32(base, p0));
				expr!(LaneExpr::MulF32(raised, p1))
			}
			10 => {
				let above = expr!(LaneExpr::GeF32(value, p0));
				let mirrored = expr!(LaneExpr::SubF32(p1, value));
				expr!(LaneExpr::SelectF32 {
					cond: above,
					if_true: mirrored,
					if_false: value,
				})
			}
			11 => {
				let two = expr!(LaneExpr::ConstF32(0x4000_0000));
				let levels = expr!(LaneExpr::MaxF32(p0, two));
				let one = expr!(LaneExpr::ConstF32(0x3F80_0000));
				let half = expr!(LaneExpr::ConstF32(0x3F00_0000));
				let range = expr!(LaneExpr::SubF32(p2, p1));
				let shifted = expr!(LaneExpr::SubF32(value, p1));
				let normalized = expr!(LaneExpr::DivF32(shifted, range));
				let clamp_lo = expr!(LaneExpr::MaxF32(normalized, zero));
				let unit = expr!(LaneExpr::MinF32(clamp_lo, one));
				let last = expr!(LaneExpr::SubF32(levels, one));
				let scaled = expr!(LaneExpr::MulF32(unit, last));
				let rounded = expr!(LaneExpr::AddF32(scaled, half));
				let bucket = expr!(LaneExpr::FloorF32(rounded));
				let fraction = expr!(LaneExpr::DivF32(bucket, last));
				let rescaled = expr!(LaneExpr::MulF32(fraction, range));
				expr!(LaneExpr::AddF32(rescaled, p1))
			}
			_ => value,
		};
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec!["element_count".into(), "operation".into()],
			push_f32_scalars: vec!["p0".into(), "p1".into(), "p2".into(), "p3".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 1,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: index,
					value: result,
				},
			],
		}
	}

	/// `segmentation_overlay` — the label-indexed palette blend: a
	/// valid i32 class label on an RGB channel replaces the pixel with
	/// `value·(1 − α) + palette[label]·α`; everything else passes
	/// through. `uint(label) < class_count` reproduces the donor's
	/// `label >= 0 && label < K` bounds pair in one unsigned compare.
	pub(in crate::runtime) fn image_segmentation_overlay() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let channels = expr!(LaneExpr::Push(1));
		let class_count = expr!(LaneExpr::Push(4));
		let alpha = expr!(LaneExpr::PushF32(0));
		let batch = expr!(LaneExpr::DivU32(frame.plane, channels));
		let channel = expr!(LaneExpr::ModU32(frame.plane, channels));
		let batch_rows = expr!(LaneExpr::MulU32(batch, frame.height));
		let batch_row = expr!(LaneExpr::AddU32(batch_rows, frame.out_y));
		let pixel_row = expr!(LaneExpr::MulU32(batch_row, frame.width));
		let pixel = expr!(LaneExpr::AddU32(pixel_row, frame.out_x));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: frame.out_index,
		});
		let label = expr!(LaneExpr::LoadU32 {
			binding: 1,
			address: pixel,
		});
		let label_ok = expr!(LaneExpr::LtU32(label, class_count));
		let three = expr!(LaneExpr::ConstU32(3));
		let chan_ok = expr!(LaneExpr::LtU32(channel, three));
		let blend_ok = expr!(LaneExpr::AndBool(label_ok, chan_ok));
		let label_rgb = expr!(LaneExpr::MulU32(label, three));
		let palette_addr = expr!(LaneExpr::AddU32(label_rgb, channel));
		let color = expr!(LaneExpr::LoadF32 {
			binding: 2,
			address: palette_addr,
		});
		let one = expr!(LaneExpr::ConstF32(0x3F80_0000));
		let keep = expr!(LaneExpr::SubF32(one, alpha));
		let kept = expr!(LaneExpr::MulF32(value, keep));
		let tinted = expr!(LaneExpr::MulF32(color, alpha));
		let blended = expr!(LaneExpr::AddF32(kept, tinted));
		let result = expr!(LaneExpr::SelectF32 {
			cond: blend_ok,
			if_true: blended,
			if_false: value,
		});
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
				"class_count".into(),
			],
			push_f32_scalars: vec!["alpha".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: frame.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 3,
					address: frame.out_index,
					value: result,
				},
			],
		}
	}

	/// `normalize` — the per-channel `(v − mean_c)/std_c` donor as an
	/// elementwise 2-D tile map: `channel = plane % channels` selects
	/// the mean/std pair through the donor's ternary chain.
	pub(in crate::runtime) fn image_normalize() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let channels = expr!(LaneExpr::Push(1));
		let channel = expr!(LaneExpr::ModU32(frame.plane, channels));
		let one = expr!(LaneExpr::ConstU32(1));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let is_r = expr!(LaneExpr::EqU32(channel, zero_u32));
		let is_g = expr!(LaneExpr::EqU32(channel, one));
		let mean_0 = expr!(LaneExpr::PushF32(0));
		let mean_1 = expr!(LaneExpr::PushF32(1));
		let mean_2 = expr!(LaneExpr::PushF32(2));
		let std_0 = expr!(LaneExpr::PushF32(3));
		let std_1 = expr!(LaneExpr::PushF32(4));
		let std_2 = expr!(LaneExpr::PushF32(5));
		let mean_gb = expr!(LaneExpr::SelectF32 {
			cond: is_g,
			if_true: mean_1,
			if_false: mean_2,
		});
		let mean = expr!(LaneExpr::SelectF32 {
			cond: is_r,
			if_true: mean_0,
			if_false: mean_gb,
		});
		let std_gb = expr!(LaneExpr::SelectF32 {
			cond: is_g,
			if_true: std_1,
			if_false: std_2,
		});
		let stddev = expr!(LaneExpr::SelectF32 {
			cond: is_r,
			if_true: std_0,
			if_false: std_gb,
		});
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: frame.out_index,
		});
		let shifted = expr!(LaneExpr::SubF32(value, mean));
		let result = expr!(LaneExpr::DivF32(shifted, stddev));
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
			],
			push_f32_scalars: vec![
				"mean_0".into(),
				"mean_1".into(),
				"mean_2".into(),
				"std_0".into(),
				"std_1".into(),
				"std_2".into(),
			],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: frame.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: frame.out_index,
					value: result,
				},
			],
		}
	}

	/// `convert_color` — the RGB/BGR channel permutation: three planes
	/// per batch with `2 − channel` swaps on the outer channels when
	/// `swap_red_blue` is set.
	pub(in crate::runtime) fn image_convert_color() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let plane = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let three = expr!(LaneExpr::ConstU32(3));
		let two = expr!(LaneExpr::ConstU32(2));
		let one = expr!(LaneExpr::ConstU32(1));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(0));
		let height = expr!(LaneExpr::Push(1));
		let width = expr!(LaneExpr::Push(2));
		let swap = expr!(LaneExpr::Push(3));
		let batch_planes = expr!(LaneExpr::MulU32(batch, three));
		let x_over = expr!(LaneExpr::GeU32(out_x, width));
		let y_over = expr!(LaneExpr::GeU32(out_y, height));
		let z_over = expr!(LaneExpr::GeU32(plane, batch_planes));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let batch_i = expr!(LaneExpr::DivU32(plane, three));
		let channel = expr!(LaneExpr::ModU32(plane, three));
		let swapping = expr!(LaneExpr::NeqU32(swap, zero_u32));
		let not_green = expr!(LaneExpr::NeqU32(channel, one));
		let swap_this = expr!(LaneExpr::AndBool(swapping, not_green));
		let mirrored = expr!(LaneExpr::SubU32(two, channel));
		let source_channel = expr!(LaneExpr::SelectU32 {
			cond: swap_this,
			if_true: mirrored,
			if_false: channel,
		});
		let plane_elements = expr!(LaneExpr::MulU32(height, width));
		let pixel_row = expr!(LaneExpr::MulU32(out_y, width));
		let pixel = expr!(LaneExpr::AddU32(pixel_row, out_x));
		let batch_base = expr!(LaneExpr::MulU32(batch_i, three));
		let source_plane = expr!(LaneExpr::AddU32(batch_base, source_channel));
		let source_base = expr!(LaneExpr::MulU32(source_plane, plane_elements));
		let source = expr!(LaneExpr::AddU32(source_base, pixel));
		let target_base = expr!(LaneExpr::MulU32(plane, plane_elements));
		let target = expr!(LaneExpr::AddU32(target_base, pixel));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: source,
		});
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"height".into(),
				"width".into(),
				"swap_red_blue".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: target,
					value,
				},
			],
		}
	}

	/// `resize_normalize` — the half-pixel bilinear resize with the
	/// `swap_rb` channel remap and the per-channel `(v − m)/s` tail.
	/// The typed `BilinearF32` gather carries the donor's clamp,
	/// floor, and lerp chain verbatim; the trailing `swap_rb` word
	/// lives in the f32 push tail and reads through `asuint`.
	pub(in crate::runtime) fn image_resize_normalize() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let plane = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let three = expr!(LaneExpr::ConstU32(3));
		let two = expr!(LaneExpr::ConstU32(2));
		let one = expr!(LaneExpr::ConstU32(1));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let half = expr!(LaneExpr::ConstF32(0x3F00_0000));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch = expr!(LaneExpr::Push(0));
		let channels = expr!(LaneExpr::Push(1));
		let in_height = expr!(LaneExpr::Push(2));
		let in_width = expr!(LaneExpr::Push(3));
		let out_height = expr!(LaneExpr::Push(4));
		let out_width = expr!(LaneExpr::Push(5));
		let batch_channels = expr!(LaneExpr::MulU32(batch, channels));
		let x_over = expr!(LaneExpr::GeU32(out_x, out_width));
		let y_over = expr!(LaneExpr::GeU32(out_y, out_height));
		let z_over = expr!(LaneExpr::GeU32(plane, batch_channels));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let channel = expr!(LaneExpr::ModU32(plane, channels));
		let batch_i = expr!(LaneExpr::DivU32(plane, channels));
		let swap = expr!(LaneExpr::PushBitsU32(6));
		let swapping = expr!(LaneExpr::NeqU32(swap, zero_u32));
		let rgb = expr!(LaneExpr::LtU32(channel, three));
		let swap_this = expr!(LaneExpr::AndBool(swapping, rgb));
		let mirrored = expr!(LaneExpr::SubU32(two, channel));
		let source_channel = expr!(LaneExpr::SelectU32 {
			cond: swap_this,
			if_true: mirrored,
			if_false: channel,
		});
		let batch_base = expr!(LaneExpr::MulU32(batch_i, channels));
		let source_plane = expr!(LaneExpr::AddU32(batch_base, source_channel));
		let in_plane = expr!(LaneExpr::MulU32(in_height, in_width));
		let base = expr!(LaneExpr::MulU32(source_plane, in_plane));
		let out_x_f = expr!(LaneExpr::F32FromU32(out_x));
		let out_y_f = expr!(LaneExpr::F32FromU32(out_y));
		let in_width_f = expr!(LaneExpr::F32FromU32(in_width));
		let in_height_f = expr!(LaneExpr::F32FromU32(in_height));
		let out_width_f = expr!(LaneExpr::F32FromU32(out_width));
		let out_height_f = expr!(LaneExpr::F32FromU32(out_height));
		let x_centered = expr!(LaneExpr::AddF32(out_x_f, half));
		let x_scaled = expr!(LaneExpr::MulF32(x_centered, in_width_f));
		let x_ratio = expr!(LaneExpr::DivF32(x_scaled, out_width_f));
		let xf = expr!(LaneExpr::SubF32(x_ratio, half));
		let y_centered = expr!(LaneExpr::AddF32(out_y_f, half));
		let y_scaled = expr!(LaneExpr::MulF32(y_centered, in_height_f));
		let y_ratio = expr!(LaneExpr::DivF32(y_scaled, out_height_f));
		let yf = expr!(LaneExpr::SubF32(y_ratio, half));
		let value = expr!(LaneExpr::BilinearF32 {
			binding: 0,
			base,
			xf,
			yf,
			width: in_width,
			height: in_height,
		});
		let is_r = expr!(LaneExpr::EqU32(channel, zero_u32));
		let is_g = expr!(LaneExpr::EqU32(channel, one));
		let mean_0 = expr!(LaneExpr::PushF32(0));
		let mean_1 = expr!(LaneExpr::PushF32(1));
		let mean_2 = expr!(LaneExpr::PushF32(2));
		let std_0 = expr!(LaneExpr::PushF32(3));
		let std_1 = expr!(LaneExpr::PushF32(4));
		let std_2 = expr!(LaneExpr::PushF32(5));
		let mean_gb = expr!(LaneExpr::SelectF32 {
			cond: is_g,
			if_true: mean_1,
			if_false: mean_2,
		});
		let mean = expr!(LaneExpr::SelectF32 {
			cond: is_r,
			if_true: mean_0,
			if_false: mean_gb,
		});
		let std_gb = expr!(LaneExpr::SelectF32 {
			cond: is_g,
			if_true: std_1,
			if_false: std_2,
		});
		let stddev = expr!(LaneExpr::SelectF32 {
			cond: is_r,
			if_true: std_0,
			if_false: std_gb,
		});
		let shifted = expr!(LaneExpr::SubF32(value, mean));
		let result = expr!(LaneExpr::DivF32(shifted, stddev));
		let out_rows = expr!(LaneExpr::MulU32(plane, out_height));
		let out_row = expr!(LaneExpr::AddU32(out_rows, out_y));
		let out_line = expr!(LaneExpr::MulU32(out_row, out_width));
		let out_index = expr!(LaneExpr::AddU32(out_line, out_x));
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"input_height".into(),
				"input_width".into(),
				"output_height".into(),
				"output_width".into(),
			],
			push_f32_scalars: vec![
				"mean_0".into(),
				"mean_1".into(),
				"mean_2".into(),
				"std_0".into(),
				"std_1".into(),
				"std_2".into(),
				"swap_rb".into(),
			],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: out_index,
					value: result,
				},
			],
		}
	}

	/// `mel_filterbank` — the per-(mel, frame, channel) dot product of
	/// `mel_filterbank.slang` as a serial per-lane fold over
	/// `freq_bins`, with the donor's `log(value + 1e-9)` scale tail.
	/// Donor ABI `[input, output, filterbank]` + five u32 pushes;
	/// grid `[num_mels/32, frames, channels]` at 32 lanes.
	pub(in crate::runtime) fn audio_mel_filterbank() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let frame = expr!(LaneExpr::GroupY);
		let channel = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let thirty_two = expr!(LaneExpr::ConstU32(32));
		let tile_mel = expr!(LaneExpr::MulU32(group_x, thirty_two));
		let mel = expr!(LaneExpr::AddU32(tile_mel, lane_id));
		let channels = expr!(LaneExpr::Push(0));
		let frames = expr!(LaneExpr::Push(1));
		let freq_bins = expr!(LaneExpr::Push(2));
		let num_mels = expr!(LaneExpr::Push(3));
		let log_scale = expr!(LaneExpr::Push(4));
		let mel_over = expr!(LaneExpr::GeU32(mel, num_mels));
		let frame_over = expr!(LaneExpr::GeU32(frame, frames));
		let channel_over = expr!(LaneExpr::GeU32(channel, channels));
		let xy_over = expr!(LaneExpr::OrBool(mel_over, frame_over));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, channel_over));
		let channel_frames = expr!(LaneExpr::MulU32(channel, frames));
		let input_row = expr!(LaneExpr::AddU32(channel_frames, frame));
		let input_base = expr!(LaneExpr::MulU32(input_row, freq_bins));
		let filter_base = expr!(LaneExpr::MulU32(mel, freq_bins));
		let i = expr!(LaneExpr::Induction);
		let input_index = expr!(LaneExpr::AddU32(input_base, i));
		let filter_index = expr!(LaneExpr::AddU32(filter_base, i));
		let sample = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: input_index,
		});
		let weight = expr!(LaneExpr::LoadF32 {
			binding: 2,
			address: filter_index,
		});
		let element = expr!(LaneExpr::MulF32(sample, weight));
		let zero = expr!(LaneExpr::ConstF32(0));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 0, index: 0 });
		let epsilon = expr!(LaneExpr::ConstF32(0x3089_705F));
		let shifted = expr!(LaneExpr::AddF32(acc, epsilon));
		let logged = expr!(LaneExpr::LogF32(shifted));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let scaled = expr!(LaneExpr::NeqU32(log_scale, zero_u32));
		let value = expr!(LaneExpr::SelectF32 {
			cond: scaled,
			if_true: logged,
			if_false: acc,
		});
		let channel_mels = expr!(LaneExpr::MulU32(channel, num_mels));
		let out_row = expr!(LaneExpr::AddU32(channel_mels, mel));
		let out_base = expr!(LaneExpr::MulU32(out_row, frames));
		let out_index = expr!(LaneExpr::AddU32(out_base, frame));
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0), Binding::Input(1)],
			push_scalars: vec![
				"channels".into(),
				"frames".into(),
				"freq_bins".into(),
				"num_mels".into(),
				"log_scale".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 32,
				thread_columns: 32,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: vec![TileDecl::Register {
				rows: 1,
				columns: 1,
			}],
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: out_of_bounds,
				},
				TileOp::LaneFold {
					acc: 0,
					init: zero,
					extent: Extent::Push(2),
					op: LaneReduce::Sum,
					element,
					serial: true,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: out_index,
					value,
				},
			],
		}
	}

	/// `audio_mel_chain` — `mel_filterbank → mel_normalize` as one staged
	/// tile region (OA-CC-32's post-FFT boundary): one workgroup per
	/// channel, lanes striding the `num_mels * frames` channel run.
	/// Phase A folds each element's `freq_bins` dot product and stages
	/// the log-scaled value in a `groupshared` row; phase B runs the
	/// `channel_normalize` shared-tree mean/variance on the staged row;
	/// phase C writes `(x - mean) * rsqrt(var / n + 1e-5)`. Staging
	/// avoids recomputing the serial fold in both reduction passes.
	/// The merge pass owns admission: `capacity` covers the staged row
	/// and `values <= capacity` is proved before substitution. ABI:
	/// three buffers `[spectrum, output, filterbank]` and the
	/// filterbank donor's five push scalars `[channels, frames,
	/// freq_bins, num_mels, log_scale]`; grid `[channels, 1, 1]`.
	pub(in crate::runtime) fn audio_mel_chain(lanes: u32, capacity: u32) -> Result<Self> {
		if !lanes.is_power_of_two() || lanes > 1024 || capacity < lanes {
			return Err(Error::invalid_argument(
				"audio_mel_chain lanes must be a power of two within the workgroup floor and capacity must cover the lanes",
			));
		}
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frames = expr!(LaneExpr::Push(1));
		let freq_bins = expr!(LaneExpr::Push(2));
		let num_mels = expr!(LaneExpr::Push(3));
		let log_scale = expr!(LaneExpr::Push(4));
		let channel = expr!(LaneExpr::GroupX);
		let values = expr!(LaneExpr::MulU32(num_mels, frames));
		// Phase A: the enclosing `LaneLoop` counter is `loop2`; the
		// nested serial fold's counter is `loop3`.
		let e_stage = expr!(LaneExpr::LoopVar(2));
		let mel = expr!(LaneExpr::DivU32(e_stage, frames));
		let frame = expr!(LaneExpr::ModU32(e_stage, frames));
		let channel_frames = expr!(LaneExpr::MulU32(channel, frames));
		let input_row = expr!(LaneExpr::AddU32(channel_frames, frame));
		let input_base = expr!(LaneExpr::MulU32(input_row, freq_bins));
		let filter_base = expr!(LaneExpr::MulU32(mel, freq_bins));
		let k = expr!(LaneExpr::Induction);
		let input_index = expr!(LaneExpr::AddU32(input_base, k));
		let filter_index = expr!(LaneExpr::AddU32(filter_base, k));
		let sample = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: input_index,
		});
		let weight = expr!(LaneExpr::LoadF32 {
			binding: 2,
			address: filter_index,
		});
		let product = expr!(LaneExpr::MulF32(sample, weight));
		let zero_f32 = expr!(LaneExpr::ConstF32(0));
		let acc = expr!(LaneExpr::RegisterF32 { tile: 3, index: 0 });
		let epsilon = expr!(LaneExpr::ConstF32(0x3089_705F));
		let shifted = expr!(LaneExpr::AddF32(acc, epsilon));
		let logged = expr!(LaneExpr::LogF32(shifted));
		let zero_u32 = expr!(LaneExpr::ConstU32(0));
		let do_log = expr!(LaneExpr::NeqU32(log_scale, zero_u32));
		let staged_value = expr!(LaneExpr::SelectF32 {
			cond: do_log,
			if_true: logged,
			if_false: acc,
		});
		// Phase B/C: `Induction` is the lane-strided element index in
		// the reduction folds and the store loop.
		let e = expr!(LaneExpr::Induction);
		let staged = expr!(LaneExpr::SharedF32 { tile: 0, offset: e });
		let reduce_acc = expr!(LaneExpr::RegisterF32 { tile: 4, index: 0 });
		let lane_id = expr!(LaneExpr::Lane);
		let partial = expr!(LaneExpr::SharedF32 {
			tile: 1,
			offset: zero_u32,
		});
		let count_f32 = expr!(LaneExpr::F32FromU32(values));
		let mean = expr!(LaneExpr::DivF32(partial, count_f32));
		let mean_reg = expr!(LaneExpr::RegisterF32 { tile: 5, index: 0 });
		let centered = expr!(LaneExpr::SubF32(staged, mean_reg));
		let squared = expr!(LaneExpr::MulF32(centered, centered));
		let partial_sq = expr!(LaneExpr::SharedF32 {
			tile: 2,
			offset: zero_u32,
		});
		let variance = expr!(LaneExpr::DivF32(partial_sq, count_f32));
		let eps = expr!(LaneExpr::ConstF32(0x3727_c5ac));
		let shifted_variance = expr!(LaneExpr::AddF32(variance, eps));
		let inverse_stddev = expr!(LaneExpr::RsqrtF32(shifted_variance));
		let inverse_reg = expr!(LaneExpr::RegisterF32 { tile: 6, index: 0 });
		let normalized = expr!(LaneExpr::MulF32(centered, inverse_reg));
		let channel_base = expr!(LaneExpr::MulU32(channel, values));
		let out_index = expr!(LaneExpr::AddU32(channel_base, e));
		Ok(Self {
			bindings: vec![Binding::Input(0), Binding::Output(0), Binding::Input(1)],
			push_scalars: vec![
				"channels".into(),
				"frames".into(),
				"freq_bins".into(),
				"num_mels".into(),
				"log_scale".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes,
				thread_columns: lanes,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles: vec![
				TileDecl::Shared {
					rows: 1,
					columns: capacity,
					stride: capacity,
				},
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Shared {
					rows: 1,
					columns: lanes,
					stride: lanes,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
				TileDecl::Register {
					rows: 1,
					columns: 1,
				},
			],
			lane_exprs,
			ops: vec![
				TileOp::Guard { bound: 0 },
				TileOp::LaneLoop {
					extent: Extent::Expr(values),
					body: vec![
						TileOp::LaneFold {
							acc: 3,
							init: zero_f32,
							extent: Extent::Push(2),
							op: LaneReduce::Sum,
							element: product,
							serial: true,
						},
						TileOp::StoreShared {
							tile: 0,
							offset: e_stage,
							value: staged_value,
						},
					],
				},
				TileOp::Barrier,
				TileOp::LaneFold {
					acc: 4,
					init: zero_f32,
					extent: Extent::Expr(values),
					op: LaneReduce::Sum,
					element: staged,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 1,
					offset: lane_id,
					value: reduce_acc,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 1,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 5,
					index: 0,
					value: mean,
				},
				TileOp::LaneFold {
					acc: 4,
					init: zero_f32,
					extent: Extent::Expr(values),
					op: LaneReduce::Sum,
					element: squared,
					serial: false,
				},
				TileOp::StoreShared {
					tile: 2,
					offset: lane_id,
					value: reduce_acc,
				},
				TileOp::Barrier,
				TileOp::TreeReduce {
					tile: 2,
					op: LaneReduce::Sum,
				},
				TileOp::AssignScalar {
					tile: 6,
					index: 0,
					value: inverse_stddev,
				},
				TileOp::LaneLoop {
					extent: Extent::Expr(values),
					body: vec![TileOp::StoreScalar {
						binding: 1,
						address: out_index,
						value: normalized,
					}],
				},
			],
		})
	}

	/// `grayscale` — the Rec.709 luma fold over the three RGB planes:
	/// grid Z is `batch` (single-channel output) and the donor's
	/// `0.2126r + 0.7152g + 0.0722b` accumulator order is preserved.
	pub(in crate::runtime) fn image_grayscale() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let group_x = expr!(LaneExpr::GroupX);
		let group_y = expr!(LaneExpr::GroupY);
		let batch = expr!(LaneExpr::GroupZ);
		let lane_id = expr!(LaneExpr::Lane);
		let sixteen = expr!(LaneExpr::ConstU32(16));
		let two = expr!(LaneExpr::ConstU32(2));
		let thread_row = expr!(LaneExpr::DivU32(lane_id, sixteen));
		let thread_column = expr!(LaneExpr::ModU32(lane_id, sixteen));
		let tile_x = expr!(LaneExpr::MulU32(group_x, sixteen));
		let tile_y = expr!(LaneExpr::MulU32(group_y, sixteen));
		let out_x = expr!(LaneExpr::AddU32(tile_x, thread_column));
		let out_y = expr!(LaneExpr::AddU32(tile_y, thread_row));
		let batch_count = expr!(LaneExpr::Push(0));
		let channels = expr!(LaneExpr::Push(1));
		let height = expr!(LaneExpr::Push(2));
		let width = expr!(LaneExpr::Push(3));
		let x_over = expr!(LaneExpr::GeU32(out_x, width));
		let y_over = expr!(LaneExpr::GeU32(out_y, height));
		let xy_over = expr!(LaneExpr::OrBool(x_over, y_over));
		let z_over = expr!(LaneExpr::GeU32(batch, batch_count));
		let out_of_bounds = expr!(LaneExpr::OrBool(xy_over, z_over));
		let plane = expr!(LaneExpr::MulU32(height, width));
		let row = expr!(LaneExpr::MulU32(out_y, width));
		let pixel = expr!(LaneExpr::AddU32(row, out_x));
		let batch_planes = expr!(LaneExpr::MulU32(batch, channels));
		let batch_base = expr!(LaneExpr::MulU32(batch_planes, plane));
		let base = expr!(LaneExpr::AddU32(batch_base, pixel));
		let green_base = expr!(LaneExpr::AddU32(base, plane));
		let two_plane = expr!(LaneExpr::MulU32(two, plane));
		let blue_base = expr!(LaneExpr::AddU32(base, two_plane));
		let red = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: base,
		});
		let green = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: green_base,
		});
		let blue = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: blue_base,
		});
		let w_r = expr!(LaneExpr::ConstF32(0x3E59_B3D0));
		let w_g = expr!(LaneExpr::ConstF32(0x3F37_1759));
		let w_b = expr!(LaneExpr::ConstF32(0x3D93_DD98));
		let wr = expr!(LaneExpr::MulF32(w_r, red));
		let wg = expr!(LaneExpr::MulF32(w_g, green));
		let wb = expr!(LaneExpr::MulF32(w_b, blue));
		let rg = expr!(LaneExpr::AddF32(wr, wg));
		let luma = expr!(LaneExpr::AddF32(rg, wb));
		let out_plane = expr!(LaneExpr::MulU32(batch, plane));
		let out_index = expr!(LaneExpr::AddU32(out_plane, pixel));
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"input_channels".into(),
				"height".into(),
				"width".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: out_index,
					value: luma,
				},
			],
		}
	}

	/// `channel_reorder` — the per-plane `order[channel]` remap: a
	/// four-entry select chain standing in for the donor's push-array
	/// indexing.
	pub(in crate::runtime) fn image_channel_reorder() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let two = expr!(LaneExpr::ConstU32(2));
		let one = expr!(LaneExpr::ConstU32(1));
		let zero = expr!(LaneExpr::ConstU32(0));
		let channels = expr!(LaneExpr::Push(1));
		let channel = expr!(LaneExpr::ModU32(frame.plane, channels));
		let batch_i = expr!(LaneExpr::DivU32(frame.plane, channels));
		let plane_elements = expr!(LaneExpr::MulU32(frame.height, frame.width));
		let batch_planes = expr!(LaneExpr::MulU32(batch_i, channels));
		let batch_base = expr!(LaneExpr::MulU32(batch_planes, plane_elements));
		let pixel_row = expr!(LaneExpr::MulU32(frame.out_y, frame.width));
		let pixel = expr!(LaneExpr::AddU32(pixel_row, frame.out_x));
		let order_0 = expr!(LaneExpr::Push(4));
		let order_1 = expr!(LaneExpr::Push(5));
		let order_2 = expr!(LaneExpr::Push(6));
		let order_3 = expr!(LaneExpr::Push(7));
		let is_0 = expr!(LaneExpr::EqU32(channel, zero));
		let is_1 = expr!(LaneExpr::EqU32(channel, one));
		let is_2 = expr!(LaneExpr::EqU32(channel, two));
		let tail_23 = expr!(LaneExpr::SelectU32 {
			cond: is_2,
			if_true: order_2,
			if_false: order_3,
		});
		let tail_123 = expr!(LaneExpr::SelectU32 {
			cond: is_1,
			if_true: order_1,
			if_false: tail_23,
		});
		let order = expr!(LaneExpr::SelectU32 {
			cond: is_0,
			if_true: order_0,
			if_false: tail_123,
		});
		let order_offset = expr!(LaneExpr::MulU32(order, plane_elements));
		let source = expr!(LaneExpr::AddU32(batch_base, order_offset));
		let source_index = expr!(LaneExpr::AddU32(source, pixel));
		let value = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: source_index,
		});
		Self {
			bindings: vec![Binding::Input(0), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
				"channel_0".into(),
				"channel_1".into(),
				"channel_2".into(),
				"channel_3".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: frame.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 1,
					address: frame.out_index,
					value,
				},
			],
		}
	}

	/// `color_twist` — the shared `[3, 4]` affine color transform on the
	/// first three channel planes; higher planes pass through.
	pub(in crate::runtime) fn image_color_twist() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let four = expr!(LaneExpr::ConstU32(4));
		let three = expr!(LaneExpr::ConstU32(3));
		let channels = expr!(LaneExpr::Push(1));
		let channel = expr!(LaneExpr::ModU32(frame.plane, channels));
		let batch_i = expr!(LaneExpr::DivU32(frame.plane, channels));
		let plane_elements = expr!(LaneExpr::MulU32(frame.height, frame.width));
		let batch_planes = expr!(LaneExpr::MulU32(batch_i, channels));
		let batch_base = expr!(LaneExpr::MulU32(batch_planes, plane_elements));
		let pixel_row = expr!(LaneExpr::MulU32(frame.out_y, frame.width));
		let pixel = expr!(LaneExpr::AddU32(pixel_row, frame.out_x));
		let value_base = expr!(LaneExpr::AddU32(batch_base, pixel));
		let channel_plane = expr!(LaneExpr::MulU32(channel, plane_elements));
		let target = expr!(LaneExpr::AddU32(value_base, channel_plane));
		let own = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: target,
		});
		let in_rgb = expr!(LaneExpr::LtU32(channel, three));
		let row = expr!(LaneExpr::MulU32(channel, four));
		let mut products = Vec::new();
		for tap in [0_u32, 1, 2] {
			let row_offset = expr!(LaneExpr::ConstU32(tap));
			let coefficient_row = expr!(LaneExpr::AddU32(row, row_offset));
			let coefficient = expr!(LaneExpr::LoadF32 {
				binding: 1,
				address: coefficient_row,
			});
			let plane_offset = expr!(LaneExpr::ConstU32(tap));
			let plane_delta = expr!(LaneExpr::MulU32(plane_offset, plane_elements));
			let sample_index = expr!(LaneExpr::AddU32(value_base, plane_delta));
			let sample = expr!(LaneExpr::LoadF32 {
				binding: 0,
				address: sample_index,
			});
			products.push(expr!(LaneExpr::MulF32(coefficient, sample)));
		}
		let head = expr!(LaneExpr::AddF32(products[0], products[1]));
		let tail = expr!(LaneExpr::AddF32(head, products[2]));
		let bias_row = expr!(LaneExpr::AddU32(row, three));
		let bias = expr!(LaneExpr::LoadF32 {
			binding: 1,
			address: bias_row,
		});
		let accumulate = expr!(LaneExpr::AddF32(tail, bias));
		let result = expr!(LaneExpr::SelectF32 {
			cond: in_rgb,
			if_true: accumulate,
			if_false: own,
		});
		Self {
			bindings: vec![Binding::Input(0), Binding::Input(1), Binding::Output(0)],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
			],
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: frame.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 2,
					address: target,
					value: result,
				},
			],
		}
	}

	/// `composite` — the parameterized `composite.slang` donor:
	/// `operation` 0 = `a(1−α)+bα` alpha blend, 1 = clamped mask blend,
	/// otherwise the unsigned-wrap erase rectangle. `[a, b, mask, out]`.
	pub(in crate::runtime) fn image_composite() -> Self {
		let mut lane_exprs = Vec::new();
		macro_rules! expr {
			($e:expr) => {
				lane(&mut lane_exprs, $e)
			};
		}
		let frame = Self::filter_prelude(&mut lane_exprs);
		let zero = expr!(LaneExpr::ConstU32(0));
		let one = expr!(LaneExpr::ConstU32(1));
		let zero_f = expr!(LaneExpr::ConstF32(0));
		let one_f = expr!(LaneExpr::ConstF32(0x3F80_0000));
		let operation = expr!(LaneExpr::Push(4));
		let mask_channels = expr!(LaneExpr::Push(5));
		let rect_x = expr!(LaneExpr::Push(6));
		let rect_y = expr!(LaneExpr::Push(7));
		let rect_width = expr!(LaneExpr::Push(8));
		let rect_height = expr!(LaneExpr::Push(9));
		let alpha = expr!(LaneExpr::PushF32(0));
		let value_f = expr!(LaneExpr::PushF32(1));
		let a = expr!(LaneExpr::LoadF32 {
			binding: 0,
			address: frame.out_index,
		});
		let b = expr!(LaneExpr::LoadF32 {
			binding: 1,
			address: frame.out_index,
		});
		let one_minus_alpha = expr!(LaneExpr::SubF32(one_f, alpha));
		let a_alpha = expr!(LaneExpr::MulF32(a, one_minus_alpha));
		let b_alpha = expr!(LaneExpr::MulF32(b, alpha));
		let alpha_result = expr!(LaneExpr::AddF32(a_alpha, b_alpha));
		let channels = expr!(LaneExpr::Push(1));
		let channel = expr!(LaneExpr::ModU32(frame.plane, channels));
		let single = expr!(LaneExpr::EqU32(mask_channels, one));
		let mask_channel = expr!(LaneExpr::SelectU32 {
			cond: single,
			if_true: zero,
			if_false: channel,
		});
		let batch_i = expr!(LaneExpr::DivU32(frame.plane, channels));
		let mask_planes = expr!(LaneExpr::MulU32(batch_i, mask_channels));
		let mask_plane = expr!(LaneExpr::AddU32(mask_planes, mask_channel));
		let mask_rows = expr!(LaneExpr::MulU32(mask_plane, frame.height));
		let mask_row = expr!(LaneExpr::AddU32(mask_rows, frame.out_y));
		let mask_line = expr!(LaneExpr::MulU32(mask_row, frame.width));
		let mask_index = expr!(LaneExpr::AddU32(mask_line, frame.out_x));
		let mask_raw = expr!(LaneExpr::LoadF32 {
			binding: 2,
			address: mask_index,
		});
		let mask_lo = expr!(LaneExpr::MaxF32(mask_raw, zero_f));
		let mask = expr!(LaneExpr::MinF32(mask_lo, one_f));
		let one_minus_mask = expr!(LaneExpr::SubF32(one_f, mask));
		let a_mask = expr!(LaneExpr::MulF32(a, one_minus_mask));
		let b_mask = expr!(LaneExpr::MulF32(b, mask));
		let mask_result = expr!(LaneExpr::AddF32(a_mask, b_mask));
		let x_inside = expr!(LaneExpr::GeU32(frame.out_x, rect_x));
		let x_delta = expr!(LaneExpr::SubU32(frame.out_x, rect_x));
		let x_fits = expr!(LaneExpr::LtU32(x_delta, rect_width));
		let x_in = expr!(LaneExpr::AndBool(x_inside, x_fits));
		let y_inside = expr!(LaneExpr::GeU32(frame.out_y, rect_y));
		let y_delta = expr!(LaneExpr::SubU32(frame.out_y, rect_y));
		let y_fits = expr!(LaneExpr::LtU32(y_delta, rect_height));
		let y_in = expr!(LaneExpr::AndBool(y_inside, y_fits));
		let inside = expr!(LaneExpr::AndBool(x_in, y_in));
		let rect_result = expr!(LaneExpr::SelectF32 {
			cond: inside,
			if_true: value_f,
			if_false: a,
		});
		let is_alpha = expr!(LaneExpr::EqU32(operation, zero));
		let is_mask = expr!(LaneExpr::EqU32(operation, one));
		let non_alpha = expr!(LaneExpr::SelectF32 {
			cond: is_mask,
			if_true: mask_result,
			if_false: rect_result,
		});
		let result = expr!(LaneExpr::SelectF32 {
			cond: is_alpha,
			if_true: alpha_result,
			if_false: non_alpha,
		});
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: vec![
				"batch_size".into(),
				"channels".into(),
				"height".into(),
				"width".into(),
				"operation".into(),
				"mask_channels".into(),
				"rect_x".into(),
				"rect_y".into(),
				"rect_width".into(),
				"rect_height".into(),
			],
			push_f32_scalars: vec!["alpha".into(), "value".into()],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles: Vec::new(),
			lane_exprs,
			ops: vec![
				TileOp::GuardIf {
					cond: frame.out_of_bounds,
				},
				TileOp::StoreScalar {
					binding: 3,
					address: frame.out_index,
					value: result,
				},
			],
		}
	}

	/// Assemble a `filter.slang`-ABI program shell: four bindings
	/// `[input, kernel_x, kernel_y, output]`, the shared push layout,
	/// the bounds guard, the body ops, and the output store.
	fn filter_program(
		lane_exprs: Vec<LaneExpr>,
		tiles: Vec<TileDecl>,
		mut body: Vec<TileOp>,
		out_of_bounds: usize,
		out_index: usize,
		result: usize,
	) -> Self {
		let mut ops = Vec::with_capacity(body.len() + 2);
		ops.push(TileOp::GuardIf {
			cond: out_of_bounds,
		});
		ops.append(&mut body);
		ops.push(TileOp::StoreScalar {
			binding: 3,
			address: out_index,
			value: result,
		});
		Self {
			bindings: vec![
				Binding::Input(0),
				Binding::Input(1),
				Binding::Input(2),
				Binding::Output(0),
			],
			push_scalars: Self::FILTER_PUSH_SCALARS
				.iter()
				.map(|name| (*name).to_string())
				.collect(),
			push_f32_scalars: vec![
				"p0".into(),
				"p1".into(),
				"p2".into(),
				"p3".into(),
				"border_value".into(),
			],
			shape: BlockShape {
				lanes: 256,
				thread_columns: 16,
				fragment_rows: 1,
				tile_rows: 1,
				fragment_columns: 1,
				tile_columns: 1,
			},
			tiles,
			lane_exprs,
			ops,
		}
	}

	/// Shared flat per-thread program shell: 256 lanes over the
	/// dispatch index, with per-builder tile declarations.
	fn flat_program(
		bindings: Vec<Binding>,
		push_scalars: Vec<String>,
		tiles: Vec<TileDecl>,
		lane_exprs: Vec<LaneExpr>,
		ops: Vec<TileOp>,
	) -> Self {
		Self {
			bindings,
			push_scalars,
			push_f32_scalars: Vec::new(),
			shape: BlockShape {
				lanes: 256,
				thread_columns: 1,
				fragment_rows: 1,
				fragment_columns: 1,
				tile_rows: 1,
				tile_columns: 1,
			},
			tiles,
			lane_exprs,
			ops,
		}
	}
}
