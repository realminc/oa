use std::sync::Arc;

use std::collections::BTreeMap;

use crate::{Error, Result};

use crate::runtime::{
	Buffer, BufferAccess, BufferBinding, ComputeDispatch, Device, PushConstant, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpId,
	device::{features::ExecutionProfile, physical::DevicePhysical},
	dna::{
		fusion,
		kernel::{KernelId, TrainingReplayRole},
	},
};

use crate::runtime::device::logical::video::{
	DecodedPlaneCopy, EncodePlaneCopy, PlaneCopyPhase, RetainedDecode, RetainedEncode,
};

/// Owned snapshot of concrete work ready for Vulkan command recording.
#[derive(Clone)]
pub(in crate::runtime) struct ExecutableGraph {
	nodes: Vec<GraphNode>,
	barriers: Vec<Vec<BufferHazard>>,
	/// Queue families this plan's node affinities resolved to at capture.
	/// Fixed with the plan — replay does not re-resolve (OA-CC-22).
	families: QueueFamilies,
	node_families: Vec<u32>,
}

/// Conservative per-dispatch GPU floor measured on Xe GT2 (5–9µs per small
/// unfused dispatch in the `discovered_chains` tape sweep, rounded up).
/// Routing policy, not a capability bound; OA-CC-12 route records refine it.
const SUBMISSION_DISPATCH_FLOOR_NS: u64 = 8_000;
/// Memory-bandwidth proxy for node cost in bytes per nanosecond, from the
/// same Xe sweep (~6.6 f32 elements/ns ≈ 26B/ns). Memory-bound first-order
/// model — compute-bound kernels underestimate until OA-CC-12 supplies
/// measured per-kernel rates. This cannot bound watchdog exposure.
const SUBMISSION_BYTES_PER_NS: u64 = 26;
/// Default per-submission budget. Half a second sits below the ~0.67s
/// measured-safe 128-sequence PPO submission on Haswell (the 320-sequence
/// submission exceeded the i915 heartbeat, `VK_ERROR_DEVICE_LOST`
/// 2026-11-09) with margin, and far below typical OS GPU watchdog periods.
/// This is an estimate safety margin, not a guarantee against device loss.
const DEFAULT_SUBMISSION_BUDGET_NS: u64 = 500_000_000;

/// Seed a device's per-submission budget. `OA_SUBMISSION_BUDGET_MS`
/// overrides for laboratory runs; device-class entries arrive only with
/// measured evidence. The policy's source is recorded in the Xe
/// qualification ledger.
pub(in crate::runtime) fn submission_budget_seed(_info: &crate::DeviceInfo) -> u64 {
	if let Ok(value) = std::env::var("OA_SUBMISSION_BUDGET_MS")
		&& let Ok(ms) = value.parse::<u64>()
	{
		return ms.saturating_mul(1_000_000);
	}
	DEFAULT_SUBMISSION_BUDGET_NS
}

/// Queue role used to resolve command recording and resource transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) enum QueueAffinity {
	/// Compute-family command recording.
	#[default]
	Compute,
	/// Transfer-family queue.
	Transfer,
	/// Video-decode-family queue.
	VideoDecode,
	/// Video-encode-family queue.
	VideoEncode,
}

impl QueueAffinity {
	/// Stable diagnostic/sync-report name.
	pub(in crate::runtime) const fn name(self) -> &'static str {
		match self {
			Self::Compute => "compute",
			Self::Transfer => "transfer",
			Self::VideoDecode => "video_decode",
			Self::VideoEncode => "video_encode",
		}
	}
}

/// Queue families a captured plan's node affinities resolve to on its
/// device. Video roles resolve to enabled video families. Transfer work uses
/// the compute family when the engine has no dedicated transfer lane.
#[derive(Clone, Copy)]
pub(in crate::runtime) struct QueueFamilies {
	compute: u32,
	transfer: Option<u32>,
	video_decode: Option<u32>,
	video_encode: Option<u32>,
}

impl QueueFamilies {
	pub(in crate::runtime) fn from_physical(physical: &DevicePhysical) -> Self {
		Self {
			compute: physical.compute_queue_family,
			// A dedicated transfer queue is enabled when the device
			// advertises a transfer family distinct from compute; same-family
			// transfer support resolves to the compute queue instead.
			transfer: physical
				.transfer_queue_family
				.filter(|family| *family != physical.compute_queue_family),
			video_decode: physical.video.decode_queue_family,
			video_encode: physical.video.encode_queue_family,
		}
	}

	/// The queue family a role's commands submit on. Resolving a video
	/// affinity without its enabled family is a capture-time capability
	/// error, never a silent compute fallback.
	pub(in crate::runtime) fn resolve(&self, affinity: QueueAffinity) -> Result<u32> {
		match affinity {
			QueueAffinity::Compute => Ok(self.compute),
			QueueAffinity::Transfer => Ok(self.transfer.unwrap_or(self.compute)),
			QueueAffinity::VideoDecode => self
				.video_decode
				.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue family is enabled")),
			QueueAffinity::VideoEncode => self
				.video_encode
				.ok_or_else(|| Error::missing_capability("no Vulkan Video encode queue family is enabled")),
		}
	}

	pub(in crate::runtime) const fn matrix_copy_affinity(&self) -> QueueAffinity {
		// Matrix buffers are exclusive and can enter or leave a plan on the
		// compute family, including through rebind and host observation.
		// Keep the copy there until plans track boundary ownership.
		if self.transfer.is_some() {
			QueueAffinity::Compute
		} else {
			QueueAffinity::Transfer
		}
	}
}

/// One contiguous run of nodes recorded for a single queue family. Runs
/// split only at resolved-family transitions, and a family may recur in
/// several runs — each run becomes one timeline-chained queue submission
/// (OA-CC-22).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::runtime) struct QueueRun {
	pub(in crate::runtime) family: u32,
	pub(in crate::runtime) nodes: std::ops::Range<usize>,
}

/// One retained operation in executable order. Compute lowering works with
/// dispatch payloads, while capture and replay own the full command sequence.
#[derive(Clone)]
pub(in crate::runtime) enum GraphNode {
	Compute(ComputeNode),
	BufferCopy(BufferCopyNode),
	VideoPlanes(VideoPlaneNode),
	VideoDecode(Arc<RetainedDecode>),
	VideoEncodeInput(VideoEncodeInputNode),
	VideoEncode(Arc<RetainedEncode>),
}

impl GraphNode {
	pub(in crate::runtime) const fn compute(&self) -> Option<&ComputeNode> {
		match self {
			Self::Compute(node) => Some(node),
			Self::BufferCopy(_)
			| Self::VideoPlanes(_)
			| Self::VideoDecode(_)
			| Self::VideoEncodeInput(_)
			| Self::VideoEncode(_) => None,
		}
	}

	fn compute_mut(&mut self) -> Option<&mut ComputeNode> {
		match self {
			Self::Compute(node) => Some(node),
			Self::BufferCopy(_)
			| Self::VideoPlanes(_)
			| Self::VideoDecode(_)
			| Self::VideoEncodeInput(_)
			| Self::VideoEncode(_) => None,
		}
	}

	pub(in crate::runtime) fn semantic(&self) -> (&'static str, &[SemanticOpId], u64) {
		match self {
			Self::Compute(node) => (node.operation, &node.semantic_ops, node.op_contract_hash),
			Self::BufferCopy(node) => (node.operation, &node.semantic_ops, node.op_contract_hash),
			Self::VideoPlanes(node) => (node.phase.name(), &[], 0),
			Self::VideoDecode(node) => (node.name(), &[], 0),
			Self::VideoEncodeInput(node) => (node.copy.name(), &[], 0),
			Self::VideoEncode(node) => (node.name(), &[], 0),
		}
	}

	pub(in crate::runtime) fn buffers(&self) -> &[BufferUse] {
		match self {
			Self::Compute(node) => &node.buffers,
			Self::BufferCopy(node) => &node.buffers,
			Self::VideoPlanes(node) => &node.buffers,
			Self::VideoDecode(_) => &[],
			Self::VideoEncodeInput(node) => &node.buffers,
			Self::VideoEncode(_) => &[],
		}
	}

	fn buffers_mut(&mut self) -> &mut [BufferUse] {
		match self {
			Self::Compute(node) => &mut node.buffers,
			Self::BufferCopy(node) => &mut node.buffers,
			Self::VideoPlanes(node) => &mut node.buffers,
			Self::VideoDecode(_) => &mut [],
			Self::VideoEncodeInput(node) => &mut node.buffers,
			Self::VideoEncode(_) => &mut [],
		}
	}

	fn access(&self, access: BufferAccess) -> AccessState {
		match self {
			Self::Compute(_) => AccessState::from_access(access),
			Self::BufferCopy(_)
			| Self::VideoPlanes(_)
			| Self::VideoDecode(_)
			| Self::VideoEncodeInput(_)
			| Self::VideoEncode(_) => AccessState::from_copy_access(access),
		}
	}

	pub(in crate::runtime) fn indirect(&self) -> Option<&IndirectUse> {
		self.compute().and_then(|node| node.indirect.as_ref())
	}

	pub(in crate::runtime) const fn queue(&self) -> QueueAffinity {
		match self {
			Self::Compute(_) => QueueAffinity::Compute,
			// Copy work carries the transfer role. On routes without a
			// dedicated transfer queue it still resolves to the compute
			// family, so no cross-family edge is derived there.
			Self::BufferCopy(node) => node.queue,
			Self::VideoDecode(_) => QueueAffinity::VideoDecode,
			Self::VideoPlanes(node) => match node.phase {
				PlaneCopyPhase::Copy => QueueAffinity::Compute,
				PlaneCopyPhase::Release | PlaneCopyPhase::Acquire => QueueAffinity::VideoDecode,
			},
			Self::VideoEncode(_) => QueueAffinity::VideoEncode,
			// The input image is CONCURRENT across the compute and encode
			// families, so the staging copy stays on compute and the encode
			// command's internal acquire barrier orders consumption.
			Self::VideoEncodeInput(_) => QueueAffinity::Compute,
		}
	}

	fn estimated_cost_ns(&self) -> Option<u64> {
		match self {
			Self::Compute(node) => node.estimated_cost_ns(),
			Self::VideoDecode(_) | Self::VideoEncode(_) => None,
			Self::BufferCopy(node) => Some(
				SUBMISSION_DISPATCH_FLOOR_NS
					.saturating_add(node.bytes.saturating_mul(2) / SUBMISSION_BYTES_PER_NS),
			),
			Self::VideoPlanes(node) => Some(
				SUBMISSION_DISPATCH_FLOOR_NS.saturating_add(
					node
						.buffers
						.iter()
						.map(|usage| usage.buffer.size())
						.sum::<u64>()
						/ SUBMISSION_BYTES_PER_NS,
				),
			),
			Self::VideoEncodeInput(node) => Some(
				SUBMISSION_DISPATCH_FLOOR_NS.saturating_add(
					node
						.buffers
						.iter()
						.map(|usage| usage.buffer.size())
						.sum::<u64>()
						/ SUBMISSION_BYTES_PER_NS,
				),
			),
		}
	}
}

/// One command of an image ownership round trip. The copy writes ordinary
/// graph buffers so its compute consumers use the usual hazard planner.
#[derive(Clone)]
pub(in crate::runtime) struct VideoPlaneNode {
	pub(in crate::runtime) copy: Arc<DecodedPlaneCopy>,
	pub(in crate::runtime) phase: PlaneCopyPhase,
	buffers: Vec<BufferUse>,
}

impl VideoPlaneNode {
	pub(in crate::runtime) fn validate(&self) -> Result<()> {
		self.copy.validate_ownership()?;
		if self.phase == PlaneCopyPhase::Copy
			&& (self.buffers.len() != self.copy.plane_bytes().len()
				|| self
					.buffers
					.iter()
					.zip(self.copy.plane_bytes())
					.any(|(destination, &bytes)| destination.buffer.byte_len() < bytes)
				|| self.buffers.iter().enumerate().any(|(index, destination)| {
					self.buffers[..index]
						.iter()
						.any(|earlier| earlier.buffer.same_as(&destination.buffer))
				})) {
			return Err(Error::invalid_argument(
				"decoded video planes need disjoint buffers with sufficient capacity",
			));
		}
		Ok(())
	}
}

/// Buffer-to-image staging of one slot's encode input planes. The input
/// image is `CONCURRENT` across the compute and encode families, so this is
/// a single compute-family node — no release/acquire round trip.
#[derive(Clone)]
pub(in crate::runtime) struct VideoEncodeInputNode {
	pub(in crate::runtime) copy: Arc<EncodePlaneCopy>,
	buffers: Vec<BufferUse>,
}

impl VideoEncodeInputNode {
	pub(in crate::runtime) fn validate(&self) -> Result<()> {
		self.copy.validate_ownership()?;
		if self.buffers.len() != self.copy.plane_bytes().len()
			|| self
				.buffers
				.iter()
				.zip(self.copy.plane_bytes())
				.any(|(source, &bytes)| source.buffer.byte_len() < bytes)
			|| self.buffers.iter().enumerate().any(|(index, source)| {
				self.buffers[..index]
					.iter()
					.any(|earlier| earlier.buffer.same_as(&source.buffer))
			}) {
			return Err(Error::invalid_argument(
				"encode input planes need disjoint buffers with sufficient capacity",
			));
		}
		Ok(())
	}
}

/// A whole logical buffer copy with independent source and destination.
#[derive(Clone)]
pub(in crate::runtime) struct BufferCopyNode {
	operation: &'static str,
	semantic_ops: Vec<SemanticOpId>,
	op_contract_hash: u64,
	queue: QueueAffinity,
	pub(in crate::runtime) buffers: [BufferUse; 2],
	pub(in crate::runtime) bytes: u64,
}

impl BufferCopyNode {
	// Record the donor Matrix copy's whole-value assignment through the
	// native transfer command. Other geometry and dtypes keep their shader.
	pub(in crate::runtime) fn from_whole_copy(
		node: &ComputeNode,
		queue: QueueAffinity,
	) -> Option<Self> {
		if node.kernel != KernelId::MatrixCopyF32 || node.indirect.is_some() || node.jit.is_some() {
			return None;
		}
		let [PushConstant::U32(elements)] = node.push_constants.as_slice() else {
			return None;
		};
		let [source, destination] = node.buffers.as_slice() else {
			return None;
		};
		let bytes = u64::from(*elements) * 4;
		if bytes == 0
			|| source.access != BufferAccess::Read
			|| destination.access != BufferAccess::Write
			|| source.buffer.size() != bytes
			|| destination.buffer.size() != bytes
			|| source.buffer.same_as(&destination.buffer)
		{
			return None;
		}
		Some(Self {
			operation: node.operation,
			semantic_ops: node.semantic_ops.clone(),
			op_contract_hash: node.op_contract_hash,
			queue,
			buffers: [source.clone(), destination.clone()],
			bytes,
		})
	}
}

#[derive(Clone)]
pub(in crate::runtime) struct ComputeNode {
	pub(in crate::runtime) operation: &'static str,
	pub(in crate::runtime) kernel: KernelId,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
	pub(in crate::runtime) workgroups: [u32; 3],
	pub(in crate::runtime) indirect: Option<IndirectUse>,
	pub(in crate::runtime) semantic_ops: Vec<SemanticOpId>,
	pub(in crate::runtime) op_contract_hash: u64,
	/// An admitted runtime specialization of this node's fused elementwise
	/// chain. When the device table still holds the pipeline, recording
	/// dispatches it against `buffers` in program order (inputs then outputs)
	/// with `push_constants` as the trailing push words; otherwise the node's
	/// canonical tape dispatch executes unchanged.
	pub(in crate::runtime) jit: Option<JitNode>,
	/// Observation identity (`vkcache::jit::signature_key` of the canonical
	/// program) set on fused nodes while `OA_JIT_FUSION` is active. Recording
	/// timestamps the dispatch so the route store learns tape vs specialized
	/// medians; `None` on canonical nodes costs nothing.
	pub(in crate::runtime) fusion_signature: Option<u64>,
}

impl ComputeNode {
	/// First-order submission cost estimate: a fixed dispatch floor plus one
	/// memory pass over every bound buffer at the device bandwidth proxy.
	/// Compute-bound kernels (matmul, convolution) underestimate; the proxy
	/// is a splitting heuristic, not a safe upper bound. OA-CC-12's measured
	/// route records refine it. `None` for indirect dispatches whose element count is
	/// not knowable at record time.
	pub(in crate::runtime) fn estimated_cost_ns(&self) -> Option<u64> {
		if self.indirect.is_some() {
			return None;
		}
		let bytes = self.buffers.iter().fold(0u64, |sum, binding| {
			sum.saturating_add(u64::try_from(binding.buffer.byte_len()).unwrap_or(u64::MAX))
		});
		Some(SUBMISSION_DISPATCH_FLOOR_NS.saturating_add(bytes / SUBMISSION_BYTES_PER_NS))
	}
}

/// Record-time parameters for one admitted specialization. The pipeline is
/// owned by the device's jit table and resolved by `key` at recording, so a
/// missing or failed admission always falls back to the node's tape route.
#[derive(Clone)]
pub(in crate::runtime) struct JitNode {
	pub(in crate::runtime) key: Vec<u8>,
	pub(in crate::runtime) buffers: Vec<BufferUse>,
	pub(in crate::runtime) push_constants: Vec<PushConstant>,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferUse {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) access: BufferAccess,
}

/// GPU-authored launch dimensions for one node: a retained buffer and the
/// four-byte-aligned byte offset of its `VkDispatchIndirectCommand` triple.
#[derive(Clone)]
pub(in crate::runtime) struct IndirectUse {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) offset: u64,
}

#[derive(Clone)]
pub(in crate::runtime) struct BufferHazard {
	pub(in crate::runtime) source_nodes: [usize; 2],
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) source: AccessState,
	pub(in crate::runtime) destination: AccessState,
	pub(in crate::runtime) source_queue: QueueAffinity,
	pub(in crate::runtime) destination_queue: QueueAffinity,
	/// Queue families the roles resolved to at capture. Cross-queue edges
	/// and ownership transfers key on families — distinct roles resolving
	/// to one family need neither.
	pub(in crate::runtime) source_family: u32,
	pub(in crate::runtime) destination_family: u32,
}

impl BufferHazard {
	/// Whether the hazard crosses queue families — a timeline-semaphore
	/// edge is required in place of an in-order pipeline barrier.
	pub(in crate::runtime) const fn is_cross_queue(&self) -> bool {
		self.source_family != self.destination_family
	}

	/// Whether the hazard requires a queue-family ownership release/acquire
	/// pair. Exclusive-owned buffers moving between families always do.
	pub(in crate::runtime) const fn is_ownership_transfer(&self) -> bool {
		self.is_cross_queue()
	}
}

pub(in crate::runtime) struct ResourceLifetime {
	pub(in crate::runtime) buffer: Buffer,
	pub(in crate::runtime) first_access: usize,
	pub(in crate::runtime) last_access: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::runtime) struct AccessState {
	read: bool,
	write: bool,
	indirect: bool,
	transfer_read: bool,
	transfer_write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedHazard<Resource> {
	source_nodes: [usize; 2],
	resource: Resource,
	source: AccessState,
	destination: AccessState,
}

/// Resolve borrowed storage bindings into owned buffer uses for one node.
///
/// Zero-byte storage legitimately owns no buffer, but Vulkan descriptor
/// bindings still require a valid buffer handle. Kernels early-return on
/// zero extents, so a minimal fresh backing buffer satisfies the binding
/// without being touched.
fn resolve_buffers(
	device: &Arc<Device>,
	operation: &'static str,
	bindings: &[BufferBinding<'_>],
) -> Result<Vec<BufferUse>> {
	let mut buffers = Vec::with_capacity(bindings.len());
	for binding in bindings {
		if !binding.storage.belongs_to(device) {
			return Err(Error::invalid_argument(format!(
				"{operation} buffers must belong to the submitting engine"
			)));
		}
		let buffer = match binding.storage.buffer() {
			Some(buffer) => buffer.clone(),
			None => Buffer::host_visible_storage(device, 4)?,
		};
		buffers.push(BufferUse {
			buffer,
			access: binding.access,
		});
	}
	Ok(buffers)
}

impl ExecutableGraph {
	/// Resolve borrowed domain dispatches into one transactionally built graph.
	pub(in crate::runtime) fn from_dispatches(
		device: &Arc<Device>,
		dispatches: &[ComputeDispatch<'_>],
	) -> Result<Self> {
		if dispatches.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one dispatch",
			));
		}

		let mut nodes: Vec<GraphNode> = Vec::with_capacity(dispatches.len());
		for dispatch in dispatches {
			device.require_kernel(dispatch.kernel)?;
			let operation = dispatch.kernel.report_name();
			let buffers = resolve_buffers(device, operation, dispatch.buffers)?;
			let indirect = dispatch
				.indirect
				.map(|arguments| {
					if !arguments.storage.belongs_to(device) {
						return Err(Error::invalid_argument(format!(
							"{operation} indirect dispatch arguments must belong to the submitting engine"
						)));
					}
					arguments
						.storage
						.validate_recording_access(BufferAccess::Read)?;
					let buffer = arguments.storage.buffer().cloned().ok_or_else(|| {
						Error::backend_failure(
							"Vulkan",
							"compute-dispatch indirect argument validation",
							std::io::Error::other(format!(
								"{operation} references indirect storage without a Vulkan buffer"
							)),
						)
					})?;
					if arguments.offset % 4 != 0
						|| arguments
							.offset
							.checked_add(12)
							.is_none_or(|end| end > arguments.storage.byte_len() as u64)
					{
						return Err(Error::invalid_argument(format!(
							"{operation} indirect dispatch arguments require a four-byte-aligned VkDispatchIndirectCommand range inside the storage"
						)));
					}
					Ok(IndirectUse {
						buffer,
						offset: arguments.offset,
					})
				})
				.transpose()?;
			let mut node = ComputeNode {
				operation,
				kernel: dispatch.kernel,
				buffers,
				push_constants: dispatch.push_constants.to_vec(),
				workgroups: dispatch.workgroups,
				indirect,
				semantic_ops: Vec::new(),
				op_contract_hash: 0,
				jit: None,
				fusion_signature: None,
			};
			// OA-CC-13/14: an admitted tile template may substitute its
			// specialization for the incumbent registry pipeline.
			crate::runtime::dna::tile::specialize_gemm_node(device, &mut node);
			crate::runtime::dna::tile::specialize_gemm_bias_node(device, &mut node);
			crate::runtime::dna::tile::specialize_gemv_node(device, &mut node);
			crate::runtime::dna::tile::specialize_softmax_node(device, &mut node);
			crate::runtime::dna::tile::specialize_log_softmax_node(device, &mut node);
			crate::runtime::dna::tile::specialize_rms_norm_node(device, &mut node);
			crate::runtime::dna::tile::specialize_layer_norm_node(device, &mut node);
			crate::runtime::dna::tile::specialize_softmax_backward_node(device, &mut node);
			crate::runtime::dna::tile::specialize_log_softmax_backward_node(device, &mut node);
			crate::runtime::dna::tile::specialize_linear_node(device, &mut node);
			crate::runtime::dna::tile::specialize_sum_node(device, &mut node);
			crate::runtime::dna::tile::specialize_sum_axis_node(device, &mut node);
			crate::runtime::dna::tile::specialize_sum_backward_node(device, &mut node);
			crate::runtime::dna::tile::specialize_embedding_backward_node(device, &mut node);
			crate::runtime::dna::tile::specialize_grouped_linear_node(device, &mut node);
			crate::runtime::dna::tile::specialize_grouped_gemm_node(device, &mut node);
			// A grouped data adjoint directly behind a prologue producer
			// stays unspecialized: the fused `grouped_backward_prologue`
			// route consumes the member and specializes at the fusion
			// push site, so a member admission attempt would be wasted.
			if !nodes
				.last()
				.and_then(GraphNode::compute)
				.is_some_and(|previous| {
					crate::runtime::dna::tile::pairs_with_backward_prologue(previous.kernel)
				}) {
				crate::runtime::dna::tile::specialize_grouped_data_backward_node(device, &mut node);
			}
			crate::runtime::dna::tile::specialize_grouped_parameter_backward_node(device, &mut node);
			crate::runtime::dna::tile::specialize_moe_plumbing_node(device, &mut node);
			crate::runtime::dna::tile::specialize_conv_node(device, &mut node);
			crate::runtime::dna::tile::specialize_pool_node(device, &mut node);
			crate::runtime::dna::tile::specialize_norm_backward_node(device, &mut node);
			// A linear data adjoint directly behind a prologue or bmm
			// producer stays unspecialized: the fused
			// `linear_backward_prologue`/`linear_backward_bmm_chain`
			// routes consume the member and specialize at the fusion
			// push site, so a member admission attempt would be wasted.
			if node.kernel != KernelId::MlLinearBackwardF32
				|| !nodes
					.last()
					.and_then(GraphNode::compute)
					.is_some_and(|previous| {
						crate::runtime::dna::tile::pairs_with_backward_prologue(previous.kernel)
							|| crate::runtime::dna::tile::pairs_with_backward_bmm(previous.kernel)
					}) {
				crate::runtime::dna::tile::specialize_linear_backward_node(device, &mut node);
			}
			crate::runtime::dna::tile::specialize_cross_entropy_node(device, &mut node);
			crate::runtime::dna::tile::specialize_max_node(device, &mut node);
			crate::runtime::dna::tile::specialize_attention_node(device, &mut node);
			crate::runtime::dna::tile::specialize_video_node(device, &mut node);
			crate::runtime::dna::tile::specialize_audio_normalize_node(device, &mut node);
			crate::runtime::dna::tile::specialize_audio_filterbank_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_resize_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_flip_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_pad_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_warp_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_remap_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_remap_index_node(device, &mut node);
			crate::runtime::dna::tile::specialize_image_filter_node(device, &mut node);
			nodes.push(GraphNode::Compute(node));
		}

		Self::from_nodes(nodes, QueueFamilies::from_physical(device.physical()))
	}

	/// Build a one-node graph for an extension kernel admitted through the
	/// public `oa::compute` surface.
	///
	/// The node reuses the jit pipeline-override path: the resolved extension
	/// pipeline substitutes the placeholder registry kernel at record time, so
	/// every kernel-derived contract (artifact write classification, bounded
	/// buffer count, secret gating) follows the jit route's admission-proved
	/// semantics. `jit.key` keeps plan-hash uniqueness across extension
	/// kernels; the placeholder identity only needs to exist for hashing and
	/// replay-role metadata, which are inert for extension dispatches.
	pub(in crate::runtime) fn from_extension(
		device: &Arc<Device>,
		operation: &'static str,
		extension_key: &[u8],
		bindings: &[BufferBinding<'_>],
		push_constants: &[PushConstant],
		workgroups: [u32; 3],
	) -> Result<Self> {
		let buffers = resolve_buffers(device, operation, bindings)?;
		let pushes = push_constants.to_vec();
		let node = ComputeNode {
			operation,
			kernel: KernelId::MatrixAddF32,
			buffers: buffers.clone(),
			push_constants: pushes.clone(),
			workgroups,
			indirect: None,
			semantic_ops: Vec::new(),
			op_contract_hash: 0,
			jit: Some(JitNode {
				key: extension_key.to_vec(),
				buffers,
				push_constants: pushes,
			}),
			fusion_signature: None,
		};
		Self::from_nodes(
			vec![GraphNode::Compute(node)],
			QueueFamilies::from_physical(device.physical()),
		)
	}

	pub(in crate::runtime) fn from_video_planes(
		device: &Arc<Device>,
		copy: Arc<DecodedPlaneCopy>,
		bindings: &[BufferBinding<'_>],
	) -> Result<Self> {
		let mut nodes = Vec::new();
		Self::append_video_planes(&mut nodes, device, copy, bindings)?;
		Self::from_nodes(nodes, QueueFamilies::from_physical(device.physical()))
	}

	pub(in crate::runtime) fn append_video_planes(
		nodes: &mut Vec<GraphNode>,
		device: &Arc<Device>,
		copy: Arc<DecodedPlaneCopy>,
		bindings: &[BufferBinding<'_>],
	) -> Result<()> {
		let cross = copy.crosses_family();
		let plane_copy = VideoPlaneNode {
			copy: copy.clone(),
			phase: PlaneCopyPhase::Copy,
			buffers: resolve_buffers(device, PlaneCopyPhase::Copy.name(), bindings)?,
		};
		plane_copy.validate()?;
		let transition = |phase| {
			GraphNode::VideoPlanes(VideoPlaneNode {
				copy: copy.clone(),
				phase,
				buffers: Vec::new(),
			})
		};
		if cross {
			nodes.push(transition(PlaneCopyPhase::Release));
		}
		nodes.push(GraphNode::VideoPlanes(plane_copy));
		if cross {
			nodes.push(transition(PlaneCopyPhase::Acquire));
		}
		Ok(())
	}

	/// Append the compute-family staging node for one slot's encode input
	/// planes. `bindings` name the plane buffers produced upstream; the
	/// `VideoEncode` node itself carries no buffers because its operands are
	/// session images and the bitstream slot, not graph storage.
	pub(in crate::runtime) fn append_encode_input(
		nodes: &mut Vec<GraphNode>,
		device: &Arc<Device>,
		copy: Arc<EncodePlaneCopy>,
		bindings: &[BufferBinding<'_>],
	) -> Result<()> {
		let node = VideoEncodeInputNode {
			buffers: resolve_buffers(device, copy.name(), bindings)?,
			copy,
		};
		node.validate()?;
		nodes.push(GraphNode::VideoEncodeInput(node));
		Ok(())
	}

	pub(in crate::runtime) fn validate_video_owners(&self) -> Result<()> {
		for node in &self.nodes {
			match node {
				GraphNode::VideoPlanes(video) => video.copy.validate_ownership()?,
				GraphNode::VideoDecode(video) => video.validate_ownership()?,
				GraphNode::VideoEncodeInput(video) => video.copy.validate_ownership()?,
				GraphNode::VideoEncode(video) => video.validate_ownership()?,
				_ => {}
			}
		}
		Ok(())
	}

	pub(in crate::runtime) fn invalidate_video_owners(&self) {
		for node in &self.nodes {
			match node {
				GraphNode::VideoPlanes(video) => video.copy.invalidate_ownership(),
				GraphNode::VideoDecode(video) => video.invalidate_ownership(),
				GraphNode::VideoEncodeInput(video) => video.copy.invalidate_ownership(),
				GraphNode::VideoEncode(video) => video.invalidate_ownership(),
				_ => {}
			}
		}
	}

	pub(in crate::runtime) fn has_video_decode(&self) -> bool {
		self
			.nodes
			.iter()
			.any(|node| matches!(node, GraphNode::VideoDecode(_)))
	}

	pub(in crate::runtime) fn has_video_encode(&self) -> bool {
		self
			.nodes
			.iter()
			.any(|node| matches!(node, GraphNode::VideoEncode(_)))
	}

	/// Any codec node requiring replay-local query pools, which forces
	/// fresh recording for every submission.
	pub(in crate::runtime) fn has_video(&self) -> bool {
		self.has_video_decode() || self.has_video_encode()
	}

	pub(in crate::runtime) fn image_transfer_count(&self) -> usize {
		self
			.nodes
			.iter()
			.filter(|node| {
				matches!(
					node,
					GraphNode::VideoPlanes(VideoPlaneNode {
						phase: PlaneCopyPhase::Release | PlaneCopyPhase::Acquire,
						..
					})
				)
			})
			.count()
	}

	pub(in crate::runtime) fn attach_direct_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.len() != 1 || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"direct semantic lowering requires one node and a nonzero contract hash",
			));
		}
		let node = self.nodes[0]
			.compute_mut()
			.ok_or_else(|| Error::internal("semantic recording requires a source compute dispatch"))?;
		node.operation = name;
		node.semantic_ops.push(operation);
		node.op_contract_hash = contract_hash;
		Ok(())
	}

	pub(in crate::runtime) fn attach_split_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.len() < 2 || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"split semantic lowering requires multiple nodes and a nonzero contract hash",
			));
		}
		for node in &mut self.nodes {
			let node = node
				.compute_mut()
				.ok_or_else(|| Error::internal("semantic recording requires source compute dispatches"))?;
			node.operation = name;
			node.semantic_ops.push(operation);
			node.op_contract_hash = contract_hash;
		}
		Ok(())
	}

	pub(in crate::runtime) fn attach_composite_semantic(
		&mut self,
		operation: SemanticOpId,
		name: &'static str,
		contract_hash: u64,
	) -> Result<()> {
		if self.nodes.is_empty() || contract_hash == 0 {
			return Err(Error::failed_precondition(
				"composite semantic lowering requires executable nodes and a nonzero contract hash",
			));
		}
		for node in &mut self.nodes {
			let node = node
				.compute_mut()
				.ok_or_else(|| Error::internal("semantic recording requires source compute dispatches"))?;
			node.operation = name;
			node.semantic_ops.clear();
			node.semantic_ops.push(operation);
			node.op_contract_hash = contract_hash;
		}
		Ok(())
	}

	pub(in crate::runtime) fn attach_fused_semantic(
		&mut self,
		operations: &[SemanticOpId],
	) -> Result<()> {
		if self.nodes.len() != 1 || operations.len() < 2 {
			return Err(Error::failed_precondition(
				"fused semantic lowering requires one node and at least two operations",
			));
		}
		let node = self.nodes[0]
			.compute_mut()
			.ok_or_else(|| Error::internal("semantic recording requires a source compute dispatch"))?;
		if node.kernel.semantic_contract().is_some() {
			return Err(Error::failed_precondition(
				"fused semantic lowering requires a distinct lowering-only kernel",
			));
		}
		node.semantic_ops.extend_from_slice(operations);
		Ok(())
	}

	pub(in crate::runtime) fn join(graphs: Vec<Self>) -> Result<Self> {
		let families = graphs.first().map(|graph| graph.families).ok_or_else(|| {
			Error::invalid_argument("an executable graph join requires at least one graph")
		})?;
		let mut nodes: Vec<_> = graphs.into_iter().flat_map(|graph| graph.nodes).collect();
		crate::runtime::dna::schedule_video_plane_returns(&mut nodes);
		Self::from_nodes(nodes, families)
	}

	pub(in crate::runtime) fn from_compute_nodes(
		nodes: Vec<ComputeNode>,
		families: QueueFamilies,
	) -> Result<Self> {
		Self::from_nodes(
			nodes.into_iter().map(GraphNode::Compute).collect(),
			families,
		)
	}

	pub(in crate::runtime) fn from_nodes(
		nodes: Vec<GraphNode>,
		families: QueueFamilies,
	) -> Result<Self> {
		if nodes.is_empty() {
			return Err(Error::invalid_argument(
				"an executable graph requires at least one node",
			));
		}
		let node_families = nodes
			.iter()
			.map(|node| families.resolve(node.queue()))
			.collect::<Result<Vec<_>>>()?;
		let barriers = plan_barriers(&nodes, &node_families)?;
		Ok(Self {
			nodes,
			barriers,
			families,
			node_families,
		})
	}

	pub(in crate::runtime) fn nodes(&self) -> &[GraphNode] {
		&self.nodes
	}

	pub(in crate::runtime) fn compute_nodes(&self) -> impl Iterator<Item = &ComputeNode> {
		self.nodes.iter().filter_map(GraphNode::compute)
	}

	/// Queue families this plan's node affinities resolved to at capture.
	pub(in crate::runtime) const fn queue_families(&self) -> QueueFamilies {
		self.families
	}

	/// Resolved queue family per node, in executable order.
	pub(in crate::runtime) fn node_families(&self) -> &[u32] {
		&self.node_families
	}

	/// Contiguous node runs per resolved queue family, in executable order.
	/// A family may recur in several runs; each run becomes one
	/// timeline-chained queue submission (OA-CC-22).
	pub(in crate::runtime) fn queue_runs(&self) -> Vec<QueueRun> {
		queue_runs(&self.node_families)
	}

	/// Greedy node ranges whose summed cost estimates each fit `budget_ns`.
	/// Splits only happen between nodes: the timeline chain makes the
	/// previous chunk's writes available and visible before the next
	/// submission's first command, so every node boundary is a legal
	/// boundary. Indirect-dispatch nodes are unestimable — they admit at the
	/// dispatch floor explicitly rather than blocking neighbouring splits.
	/// `Err((index, estimate))` names a single node over the budget; the
	/// caller turns that into the typed pre-submission error.
	/// Budget-split one node range. A submission range never crosses its
	/// bound's ends — queue-run callers bound to a run so every emitted
	/// chunk stays on one queue family.
	pub(in crate::runtime) fn budget_ranges_within(
		&self,
		bounds: std::ops::Range<usize>,
		budget_ns: u64,
	) -> std::result::Result<Vec<std::ops::Range<usize>>, (usize, u64)> {
		let mut ranges = Vec::new();
		let mut start = bounds.start;
		let mut sum = 0u64;
		for index in bounds.clone() {
			let node = &self.nodes[index];
			let cost = node
				.estimated_cost_ns()
				.unwrap_or(SUBMISSION_DISPATCH_FLOOR_NS);
			if cost > budget_ns {
				return Err((index, cost));
			}
			if index > start && sum.saturating_add(cost) > budget_ns {
				ranges.push(start..index);
				start = index;
				sum = 0;
			}
			sum = sum.saturating_add(cost);
		}
		ranges.push(start..bounds.end);
		Ok(ranges)
	}

	pub(in crate::runtime) fn validate_training_replay_safety(&self) -> Result<()> {
		validate_training_kernel_sequence(self.nodes.iter().map(|node| {
			let role = node.compute().map_or(TrainingReplayRole::Safe, |node| {
				node.kernel.training_replay_role()
			});
			(node.semantic().0, role)
		}))?;
		for (index, replay) in self.nodes.iter().enumerate() {
			let Some(replay) = replay.compute() else {
				continue;
			};
			if replay.kernel.training_replay_role() != TrainingReplayRole::ReplayRng {
				continue;
			}
			let advance = self.nodes[index + 1]
				.compute()
				.ok_or_else(|| Error::internal("validated RNG state advance lost its compute node"))?;
			let state = replay.buffers.last().ok_or_else(|| {
				Error::failed_precondition(format!(
					"training program replay RNG {} has no state binding",
					replay.operation
				))
			})?;
			let [advanced_state] = advance.buffers.as_slice() else {
				return Err(Error::failed_precondition(format!(
					"training program RNG state advance {} must bind exactly one state buffer",
					advance.operation
				)));
			};
			if state.access != BufferAccess::Read
				|| advanced_state.access != BufferAccess::ReadWrite
				|| !state.buffer.same_as(&advanced_state.buffer)
			{
				return Err(Error::failed_precondition(format!(
					"training program RNG state advance {} does not update the state read by {}",
					advance.operation, replay.operation
				)));
			}
		}
		Ok(())
	}

	pub(in crate::runtime) fn barriers_before(&self, node: usize) -> &[BufferHazard] {
		&self.barriers[node]
	}

	/// Buffer and image dependencies crossing resolved queue families.
	pub(in crate::runtime) fn cross_queue_edge_count(&self) -> usize {
		self
			.barriers
			.iter()
			.flat_map(|hazards| hazards.iter())
			.filter(|hazard| hazard.is_cross_queue())
			.count()
			+ self.image_transfer_count()
	}

	/// Ownership transfers for exclusive buffers and decoded images.
	pub(in crate::runtime) fn ownership_transfer_count(&self) -> usize {
		self
			.barriers
			.iter()
			.flat_map(|hazards| hazards.iter())
			.filter(|hazard| hazard.is_ownership_transfer())
			.count()
			+ self.image_transfer_count()
	}

	pub(in crate::runtime) fn barrier_count(&self) -> usize {
		self.barriers.iter().map(Vec::len).sum()
	}

	pub(in crate::runtime) fn identity(&self) -> u64 {
		let mut hash = StableHash::new();
		let mut resources = BTreeMap::<u32, u32>::new();
		let mut next_resource = 0_u32;
		hash.usize(self.nodes.len());
		for graph_node in &self.nodes {
			let (operation, semantic_ops, contract) = graph_node.semantic();
			hash.bytes(operation.as_bytes());
			hash.u64(contract);
			hash.usize(semantic_ops.len());
			for operation in semantic_ops {
				hash.u32(operation.index());
			}
			let node = match graph_node {
				GraphNode::Compute(node) => node,
				GraphNode::BufferCopy(copy) => {
					hash.u16(u16::MAX);
					hash.u64(copy.bytes);
					hash_buffer_uses(&mut hash, &mut resources, &mut next_resource, &copy.buffers);
					continue;
				}
				GraphNode::VideoDecode(video) => {
					hash.u16(u16::MAX - 2);
					for word in video.descriptor_words() {
						hash.u32(word);
					}
					continue;
				}
				GraphNode::VideoEncode(video) => {
					hash.u16(u16::MAX - 4);
					for word in video.descriptor_words() {
						hash.u32(word);
					}
					continue;
				}
				GraphNode::VideoEncodeInput(video) => {
					hash.u16(u16::MAX - 3);
					for word in video.copy.descriptor_words() {
						hash.u32(word);
					}
					hash_buffer_uses(
						&mut hash,
						&mut resources,
						&mut next_resource,
						&video.buffers,
					);
					continue;
				}

				GraphNode::VideoPlanes(video) => {
					hash.u16(u16::MAX - 1);
					for word in video.copy.descriptor_words() {
						hash.u32(word);
					}
					hash_buffer_uses(
						&mut hash,
						&mut resources,
						&mut next_resource,
						&video.buffers,
					);
					continue;
				}
			};
			hash.u16(node.kernel as u16);
			hash.u8(u8::from(node.jit.is_some()));
			if let Some(jit) = &node.jit {
				hash.usize(jit.key.len());
				hash.bytes(&jit.key);
			}
			let artifact = node.kernel.artifact();
			hash.u64(artifact.content_id());
			match artifact.physical_write {
				None => hash.u8(0),
				Some(contract) => {
					hash.u8(1);
					hash.usize(contract.writes.len());
					for write in contract.writes {
						hash.u8(write.binding);
						hash.u8(write.domain as u8);
						hash.u8(write.partition as u8);
						hash.u8(write.extent as u8);
						hash.u8(write.collision as u8);
						hash.u8(write.tail as u8);
					}
					hash.u8(contract.workspace as u8);
				}
			}
			hash_buffer_uses(&mut hash, &mut resources, &mut next_resource, &node.buffers);
			hash.usize(node.push_constants.len());
			for constant in &node.push_constants {
				match *constant {
					PushConstant::U32(value) => {
						hash.u8(0);
						hash.u32(value);
					}
					PushConstant::F32(value) => {
						hash.u8(1);
						hash.u32(value.to_bits());
					}
				}
			}
			for dimension in node.workgroups {
				hash.u32(dimension);
			}
			match &node.indirect {
				None => hash.u8(0),
				Some(indirect) => {
					hash.u8(1);
					let resource = *resources
						.entry(indirect.buffer.descriptor_index())
						.or_insert_with(|| {
							let resource = next_resource;
							next_resource = next_resource.wrapping_add(1);
							resource
						});
					hash.u32(resource);
					hash.u64(indirect.offset);
				}
			}
		}
		hash.finish()
	}

	pub(in crate::runtime) fn read_only_buffers(&self) -> Vec<Buffer> {
		let mut resources = BTreeMap::<u32, (Buffer, AccessState)>::new();
		for node in &self.nodes {
			for (index, buffer_use) in node.buffers().iter().enumerate() {
				if node
					.compute()
					.is_some_and(|node| fusion::is_tape_binding(node, index))
				{
					continue;
				}
				let state = node.access(buffer_use.access);
				resources
					.entry(buffer_use.buffer.descriptor_index())
					.and_modify(|(_, access)| access.merge(state))
					.or_insert_with(|| (buffer_use.buffer.clone(), state));
			}
			if let Some(indirect) = node.indirect() {
				resources
					.entry(indirect.buffer.descriptor_index())
					.and_modify(|(_, access)| access.merge(AccessState::INDIRECT))
					.or_insert_with(|| (indirect.buffer.clone(), AccessState::INDIRECT));
			}
		}
		resources
			.into_values()
			.filter_map(|(buffer, access)| {
				(!access.writes() && (access.reads() || access.indirect)).then_some(buffer)
			})
			.collect()
	}

	pub(in crate::runtime) fn resource_lifetimes(&self) -> Vec<ResourceLifetime> {
		let mut resources = Vec::<ResourceLifetime>::new();
		for (node_index, node) in self.nodes.iter().enumerate() {
			for buffer in node
				.buffers()
				.iter()
				.map(|buffer_use| &buffer_use.buffer)
				.chain(node.indirect().into_iter().map(|indirect| &indirect.buffer))
			{
				if let Some(lifetime) = resources
					.iter_mut()
					.find(|lifetime| lifetime.buffer.same_as(buffer))
				{
					lifetime.last_access = node_index;
				} else {
					resources.push(ResourceLifetime {
						buffer: buffer.clone(),
						first_access: node_index,
						last_access: node_index,
					});
				}
			}
		}
		resources
	}

	pub(in crate::runtime) fn rebind_read_only(
		&mut self,
		current: &Buffer,
		replacement: &Buffer,
		profile: ExecutionProfile,
	) -> Result<()> {
		let mut found = false;
		for node in &self.nodes {
			for buffer_use in node.buffers() {
				if buffer_use.buffer.same_as(current) {
					found = true;
					if buffer_use.access != BufferAccess::Read {
						return Err(Error::invalid_argument(
							"only read-only execution-plan Matrix inputs can be rebound",
						));
					}
					if profile == ExecutionProfile::Core
						&& node
							.compute()
							.is_some_and(|node| fusion::is_fused_chain_kernel(node.kernel))
					{
						return Err(Error::invalid_argument(
							"Core execution plans cannot rebind fused elementwise operands",
						));
					}
				} else if buffer_use.buffer.same_as(replacement) {
					return Err(Error::invalid_argument(
						"execution-plan input rebinding cannot introduce resource aliasing",
					));
				}
			}
			if let Some(indirect) = node.indirect() {
				if indirect.buffer.same_as(current) {
					found = true;
				} else if indirect.buffer.same_as(replacement) {
					return Err(Error::invalid_argument(
						"execution-plan input rebinding cannot introduce resource aliasing",
					));
				}
			}
		}
		if !found {
			return Err(Error::invalid_argument(
				"captured Matrix is not an input of this execution plan",
			));
		}
		let mut candidate = self.clone();
		for node in &mut candidate.nodes {
			for buffer_use in node.buffers_mut() {
				if buffer_use.buffer.same_as(current) {
					buffer_use.buffer = replacement.clone();
				}
			}
			let Some(node) = node.compute_mut() else {
				continue;
			};
			if let Some(jit) = &mut node.jit {
				for buffer_use in &mut jit.buffers {
					if buffer_use.buffer.same_as(current) {
						buffer_use.buffer = replacement.clone();
					}
				}
			}
			if let Some(indirect) = &mut node.indirect
				&& indirect.buffer.same_as(current)
			{
				indirect.buffer = replacement.clone();
			}
		}
		candidate.barriers = plan_barriers(&candidate.nodes, &candidate.node_families)?;
		*self = candidate;
		Ok(())
	}

	pub(in crate::runtime) fn materialize_alias_replacements(
		&mut self,
		replacements: &[(Buffer, Buffer)],
	) -> Result<()> {
		let mut candidate = self.clone();
		for node in &mut candidate.nodes {
			for buffer_use in node.buffers_mut() {
				if let Some((_, replacement)) = replacements
					.iter()
					.find(|(source, _)| source.same_as(&buffer_use.buffer))
				{
					buffer_use.buffer = replacement.clone();
				}
			}
			let Some(node) = node.compute_mut() else {
				continue;
			};
			if let Some(jit) = &mut node.jit {
				for buffer_use in &mut jit.buffers {
					if let Some((_, replacement)) = replacements
						.iter()
						.find(|(source, _)| source.same_as(&buffer_use.buffer))
					{
						buffer_use.buffer = replacement.clone();
					}
				}
			}
			if let Some(indirect) = &mut node.indirect
				&& let Some((_, replacement)) = replacements
					.iter()
					.find(|(source, _)| source.same_as(&indirect.buffer))
			{
				indirect.buffer = replacement.clone();
			}
		}
		candidate.barriers = plan_barriers(&candidate.nodes, &candidate.node_families)?;
		*self = candidate;
		Ok(())
	}

	pub(in crate::runtime) fn analyze_semantic_lowering(
		&self,
		semantic: &SemanticGraph,
	) -> Result<SemanticLoweringAnalysis> {
		semantic.validate()?;
		let mut analysis = SemanticLoweringAnalysis::empty(semantic.operations().len());
		let mut fusion_membership = vec![false; semantic.operations().len()];
		for node in &self.nodes {
			let (name, owners, contract) = node.semantic();
			if owners.is_empty() {
				analysis.note_compatibility_node();
				continue;
			}
			analysis.note_schema_node(owners.len())?;
			if owners.len() > 1 && contract != 0 {
				return Err(Error::failed_precondition(
					"fused executable node must have a distinct implementation identity",
				));
			}
			let mut seen = BTreeMap::new();
			for owner in owners {
				let operation = semantic
					.operations()
					.get(owner.index() as usize)
					.ok_or_else(|| {
						Error::out_of_range("executable node references an unknown semantic operation")
					})?;
				if seen.insert(owner.index(), ()).is_some() {
					return Err(Error::already_exists(
						"executable node repeats a semantic operation owner",
					));
				}
				if owners.len() == 1 && (name != operation.name() || contract != operation.contract_hash())
				{
					return Err(Error::failed_precondition(
						"executable node identity does not match its semantic operation",
					));
				}
				let count = &mut analysis.node_counts_mut()[owner.index() as usize];
				*count = count.saturating_add(1);
				if owners.len() > 1 {
					fusion_membership[owner.index() as usize] = true;
				}
			}
		}
		analysis.finish(&fusion_membership)?;
		Ok(analysis)
	}
}

fn validate_training_kernel_sequence<'a>(
	steps: impl IntoIterator<Item = (&'a str, TrainingReplayRole)>,
) -> Result<()> {
	let mut optimizer_state_advances = 0_usize;
	let mut optimizer_state_updates = 0_usize;
	let mut advance_seen = false;
	let mut awaiting_rng_advance = false;
	for (operation, role) in steps {
		if awaiting_rng_advance && role != TrainingReplayRole::RngStateAdvance {
			return Err(Error::failed_precondition(format!(
				"training program replay RNG must be followed immediately by its state advance; found {operation}"
			)));
		}
		match role {
			TrainingReplayRole::Safe => {}
			TrainingReplayRole::FrozenRng => {
				return Err(Error::failed_precondition(format!(
					"training program operation {operation} embeds a host seed; use its replay-state kernel"
				)));
			}
			TrainingReplayRole::ReplayRng => {
				awaiting_rng_advance = true;
			}
			TrainingReplayRole::RngStateAdvance => {
				if !awaiting_rng_advance {
					return Err(Error::failed_precondition(format!(
						"training program operation {operation} advances RNG state without a preceding replay RNG operation"
					)));
				}
				awaiting_rng_advance = false;
			}
			TrainingReplayRole::HostSteppedOptimizer => {
				return Err(Error::failed_precondition(format!(
					"training program operation {operation} embeds host-stepped optimizer state; use a replay-state kernel"
				)));
			}
			TrainingReplayRole::OptimizerStateAdvance => {
				optimizer_state_advances = optimizer_state_advances.saturating_add(1);
				advance_seen = true;
			}
			TrainingReplayRole::OptimizerStateUpdate => {
				if !advance_seen {
					return Err(Error::failed_precondition(format!(
						"training program operation {operation} reads optimizer replay state before it is advanced"
					)));
				}
				optimizer_state_updates = optimizer_state_updates.saturating_add(1);
			}
		}
	}
	if awaiting_rng_advance {
		return Err(Error::failed_precondition(
			"training program contains a replay RNG operation without its state advance",
		));
	}
	if optimizer_state_updates != 0 && optimizer_state_advances != 1 {
		return Err(Error::failed_precondition(format!(
			"training program requires exactly one optimizer-state advance before replay updates; recorded {optimizer_state_advances}"
		)));
	}
	Ok(())
}

fn hash_buffer_uses(
	hash: &mut StableHash,
	resources: &mut BTreeMap<u32, u32>,
	next_resource: &mut u32,
	buffers: &[BufferUse],
) {
	hash.usize(buffers.len());
	for buffer_use in buffers {
		let descriptor = buffer_use.buffer.descriptor_index();
		let resource = *resources.entry(descriptor).or_insert_with(|| {
			let resource = *next_resource;
			*next_resource = next_resource.wrapping_add(1);
			resource
		});
		hash.u32(resource);
		hash.u8(match buffer_use.access {
			BufferAccess::Read => 0,
			BufferAccess::Write => 1,
			BufferAccess::ReadWrite => 2,
		});
	}
}

struct StableHash(u64);

impl StableHash {
	const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
	const PRIME: u64 = 0x0000_0100_0000_01b3;

	const fn new() -> Self {
		Self(Self::OFFSET)
	}

	fn bytes(&mut self, bytes: &[u8]) {
		self.usize(bytes.len());
		self.raw(bytes);
	}

	fn raw(&mut self, bytes: &[u8]) {
		for byte in bytes {
			self.0 ^= u64::from(*byte);
			self.0 = self.0.wrapping_mul(Self::PRIME);
		}
	}

	fn u8(&mut self, value: u8) {
		self.raw(&value.to_le_bytes());
	}

	fn u16(&mut self, value: u16) {
		self.raw(&value.to_le_bytes());
	}

	fn u32(&mut self, value: u32) {
		self.raw(&value.to_le_bytes());
	}

	fn u64(&mut self, value: u64) {
		self.raw(&value.to_le_bytes());
	}

	fn usize(&mut self, value: usize) {
		self.raw(value.to_string().as_bytes());
		self.u8(0xff);
	}

	const fn finish(self) -> u64 {
		self.0
	}
}

impl AccessState {
	const INDIRECT: Self = Self {
		read: false,
		write: false,
		indirect: true,
		transfer_read: false,
		transfer_write: false,
	};

	pub(in crate::runtime) fn from_access(access: BufferAccess) -> Self {
		Self {
			read: access != BufferAccess::Write,
			write: access != BufferAccess::Read,
			..Self::default()
		}
	}

	fn from_copy_access(access: BufferAccess) -> Self {
		Self {
			transfer_read: access != BufferAccess::Write,
			transfer_write: access != BufferAccess::Read,
			..Self::default()
		}
	}

	fn merge(&mut self, other: Self) {
		self.read |= other.read;
		self.write |= other.write;
		self.indirect |= other.indirect;
		self.transfer_read |= other.transfer_read;
		self.transfer_write |= other.transfer_write;
	}

	const fn covers(self, reads: Self) -> bool {
		(!reads.read || self.read)
			&& (!reads.indirect || self.indirect)
			&& (!reads.transfer_read || self.transfer_read)
	}

	pub(in crate::runtime) const fn reads(self) -> bool {
		self.read || self.transfer_read
	}

	pub(in crate::runtime) const fn writes(self) -> bool {
		self.write || self.transfer_write
	}

	pub(in crate::runtime) const fn uses_indirect_commands(self) -> bool {
		self.indirect
	}

	pub(in crate::runtime) fn vk_stages(self) -> vk::PipelineStageFlags2 {
		let mut flags = vk::PipelineStageFlags2::empty();
		if self.read || self.write {
			flags = flags | vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT;
		}
		if self.indirect {
			flags = flags | vk::PipelineStageFlagBits2::DRAW_INDIRECT_BIT;
		}
		if self.transfer_read || self.transfer_write {
			flags = flags | vk::PipelineStageFlagBits2::COPY_BIT;
		}
		flags
	}

	pub(in crate::runtime) fn vk_access(self) -> vk::AccessFlags2 {
		let mut flags = vk::AccessFlags2::empty();
		if self.read {
			flags = flags | vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT;
		}
		if self.write {
			flags = flags | vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT;
		}
		if self.indirect {
			flags = flags | vk::AccessFlagBits2::INDIRECT_COMMAND_READ_BIT;
		}
		if self.transfer_read {
			flags = flags | vk::AccessFlagBits2::TRANSFER_READ_BIT;
		}
		if self.transfer_write {
			flags = flags | vk::AccessFlagBits2::TRANSFER_WRITE_BIT;
		}
		flags
	}
}

/// Maximal contiguous runs of one resolved queue family. Runs split only
/// at family transitions — a family may recur in several runs.
fn queue_runs(families: &[u32]) -> Vec<QueueRun> {
	let mut runs: Vec<QueueRun> = Vec::new();
	for (index, &family) in families.iter().enumerate() {
		match runs.last_mut() {
			Some(run) if run.family == family => run.nodes.end = index + 1,
			_ => runs.push(QueueRun {
				family,
				nodes: index..index + 1,
			}),
		}
	}
	runs
}

fn plan_barriers(nodes: &[GraphNode], node_families: &[u32]) -> Result<Vec<Vec<BufferHazard>>> {
	let mut resources = BTreeMap::<u32, Buffer>::new();
	let accesses = nodes
		.iter()
		.map(|node| {
			let mut bindings =
				Vec::with_capacity(node.buffers().len() + usize::from(node.indirect().is_some()));
			for buffer_use in node.buffers() {
				let identity = buffer_use.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| buffer_use.buffer.clone());
				bindings.push((identity, node.access(buffer_use.access)));
			}
			if let Some(indirect) = node.indirect() {
				let identity = indirect.buffer.descriptor_index();
				resources
					.entry(identity)
					.or_insert_with(|| indirect.buffer.clone());
				bindings.push((identity, AccessState::INDIRECT));
			}
			aggregate_accesses(bindings)
		})
		.collect::<Vec<_>>();

	plan_access_states(&accesses, |index| node_families[index])
		.into_iter()
		.enumerate()
		.map(|(node_index, hazards)| {
			hazards
				.into_iter()
				.map(|hazard| {
					let buffer = resources.get(&hazard.resource).cloned().ok_or_else(|| {
						Error::backend_failure(
							"Vulkan",
							"executable-graph hazard planning",
							std::io::Error::other("planned hazard lost its retained buffer identity"),
						)
					})?;
					Ok(BufferHazard {
						source_nodes: hazard.source_nodes,
						buffer,
						source: hazard.source,
						destination: hazard.destination,
						source_queue: nodes[hazard.source_nodes[1]].queue(),
						destination_queue: nodes[node_index].queue(),
						source_family: node_families[hazard.source_nodes[1]],
						destination_family: node_families[node_index],
					})
				})
				.collect::<Result<Vec<_>>>()
		})
		.collect()
}

fn aggregate_accesses(
	accesses: impl IntoIterator<Item = (u32, AccessState)>,
) -> BTreeMap<u32, AccessState> {
	let mut aggregate = BTreeMap::new();
	for (identity, access) in accesses {
		aggregate
			.entry(identity)
			.and_modify(|state: &mut AccessState| state.merge(access))
			.or_insert(access);
	}
	aggregate
}

/// A write remains the visibility source until a newer write. Readers of
/// different pipeline stages may need separate dependencies from that write.
/// Track the whole run of readers so a later overwrite orders all their stages.
#[derive(Clone, Copy)]
struct AccessHistory {
	nodes: [usize; 2],
	access: AccessState,
	writer: Option<(usize, AccessState)>,
	visible_to: AccessState,
}

impl AccessHistory {
	fn new(index: usize, access: AccessState) -> Self {
		Self {
			nodes: [index; 2],
			access,
			writer: access.writes().then_some((index, access)),
			visible_to: AccessState::default(),
		}
	}

	fn dependency(
		&mut self,
		index: usize,
		destination: AccessState,
		family: impl Fn(usize) -> u32,
	) -> Option<([usize; 2], AccessState)> {
		let previous = (self.nodes, self.access);
		// Exclusive ownership changes on read-to-read transitions, too.
		if family(self.nodes[1]) != family(index) || destination.writes() {
			*self = Self::new(index, destination);
			return Some(previous);
		}
		if self.access.writes() {
			self.nodes = [index; 2];
			self.access = destination;
			self.visible_to = destination;
			return Some(previous);
		}
		self.nodes[1] = index;
		self.access.merge(destination);
		if let Some((writer, source)) = self.writer
			&& !self.visible_to.covers(destination)
		{
			self.visible_to.merge(destination);
			return Some(([writer; 2], source));
		}
		None
	}
}

fn plan_access_states<Resource: Copy + Ord>(
	nodes: &[BTreeMap<Resource, AccessState>],
	family: impl Fn(usize) -> u32,
) -> Vec<Vec<PlannedHazard<Resource>>> {
	let mut history = BTreeMap::<Resource, AccessHistory>::new();
	let mut plan = Vec::with_capacity(nodes.len());

	for (index, current) in nodes.iter().enumerate() {
		let mut barriers = Vec::new();
		for (resource, destination) in current {
			match history.entry(*resource) {
				std::collections::btree_map::Entry::Vacant(entry) => {
					entry.insert(AccessHistory::new(index, *destination));
				}
				std::collections::btree_map::Entry::Occupied(mut entry) => {
					if let Some((source_nodes, source)) =
						entry.get_mut().dependency(index, *destination, &family)
					{
						barriers.push(PlannedHazard {
							source_nodes,
							resource: *resource,
							source,
							destination: *destination,
						});
					}
				}
			}
		}
		plan.push(barriers);
	}

	plan
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/executable_hazards_unit.rs"]
mod tests;
