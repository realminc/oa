// AV1 vkCmdDecodeVideoKHR record path for oa::VideoDecoder.

#include "../videoDecoderRecordAccess.h"
#include "../../../codec/vcpAv1.h"
#include <oa/runtime/engine/deviceAccess.h>

oa::Status oa::VideoDecoder::recordAV1DecodeCommands(
	oa::I32 inDpbSlot,
	const oa::Av1PictureDesc& inDesc,
	const oa::I32 inReferenceNameSlotIndices[oa::Av1MaxReferencesPerFrame])
{
	const oa::Av1FrameHeaderInfo& fh = inDesc.frameHeader;
	const oa::Av1SequenceHeaderInfo& seq = inDesc.sequenceHeader;
	// The AV1 codec path uploads the decode OBU slice before recording.
	const oa::Usize inFrameOffset = 0;
	const oa::Usize inBitstreamBase = inDesc.decodeObuOffset;
	const oa::Usize inFrameSize = inDesc.decodeObuSize;
	const oa::Span<const oa::U32> inTileOffsets(inDesc.tileOffsets.data(), inDesc.tileOffsets.size());
	const oa::Span<const oa::U32> inTileSizes(inDesc.tileSizes.data(), inDesc.tileSizes.size());
	const oa::U32 inTileCols = fh.tileCols;
	const oa::U32 inTileRows = fh.tileRows;
	const oa::U32 inTileSizeBytesMinus1 = fh.tileSizeBytesMinus1;
	const VkExtent2D pictureExtent = {
		seq.maxFrameWidthMinus1 + 1u,
		seq.maxFrameHeightMinus1 + 1u};
	const StdVideoAV1FrameType inFrameType = fh.frameType;
	const oa::U32 inOrderHint = fh.orderHint;
	const oa::U32 inPrimaryRefFrame = fh.primaryRefFrame;
	const oa::U32 inRefreshFrameFlags = fh.refreshFrameFlags;
	const oa::U8* inOrderHints = fh.orderHints;
	const bool inUse128x128Superblock = seq.use128x128Superblock;
	const bool inDisableCdfUpdate = fh.disableCdfUpdate;
	const bool inDisableFrameEndUpdateCdf = fh.disableFrameEndUpdateCdf;
	const bool inAllowScreenContentTools = fh.allowScreenContentTools;
	const oa::U32 inBaseQIdx = fh.baseQIdx;
	const oa::I32 inDeltaQYDc = fh.deltaQYDc;
	const oa::I32 inDeltaQUDc = fh.deltaQUDc;
	const oa::I32 inDeltaQUAc = fh.deltaQUAc;
	const oa::I32 inDeltaQVDc = fh.deltaQVDc;
	const oa::I32 inDeltaQVAc = fh.deltaQVAc;
	const bool inUsingQMatrix = fh.usingQMatrix;
	const bool inDiffUvDelta = fh.diffUvDelta;
	const oa::U32 inQmY = fh.qmY;
	const oa::U32 inQmU = fh.qmU;
	const oa::U32 inQmV = fh.qmV;
	const bool inSegmentationEnabled = fh.segmentationEnabled;
	const bool inSegmentationUpdateMap = fh.segmentationUpdateMap;
	const bool inSegmentationTemporalUpdate = fh.segmentationTemporalUpdate;
	const bool inSegmentationUpdateData = fh.segmentationUpdateData;
	const oa::U8 (&inSegmentFeatureEnabled)[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = fh.segmentFeatureEnabled;
	const oa::I16 (&inSegmentFeatureData)[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = fh.segmentFeatureData;
	const bool inDeltaQPresent = fh.deltaQPresent;
	const oa::U32 inDeltaQRes = fh.deltaQRes;
	const bool inDeltaLfPresent = fh.deltaLfPresent;
	const oa::U32 inDeltaLfRes = fh.deltaLfRes;
	const bool inDeltaLfMulti = fh.deltaLfMulti;
	const oa::U8* inLoopFilterLevels = fh.loopFilterLevels;
	const oa::U32 inLoopFilterSharpness = fh.loopFilterSharpness;
	const bool inLoopFilterDeltaEnabled = fh.loopFilterDeltaEnabled;
	const bool inLoopFilterDeltaUpdate = fh.loopFilterDeltaUpdate;
	const oa::U8* inLoopFilterUpdateRefDelta = fh.loopFilterUpdateRefDelta;
	const oa::I8* inLoopFilterRefDeltas = fh.loopFilterRefDeltas;
	const oa::U8* inLoopFilterUpdateModeDelta = fh.loopFilterUpdateModeDelta;
	const oa::I8* inLoopFilterModeDeltas = fh.loopFilterModeDeltas;
	const oa::U32 inCdefDampingMinus3 = fh.cdefDampingMinus3;
	const oa::U32 inCdefBits = fh.cdefBits;
	const oa::U8* inCdefYPriStrength = fh.cdefYPriStrength;
	const oa::U8* inCdefYSecStrength = fh.cdefYSecStrength;
	const oa::U8* inCdefUvPriStrength = fh.cdefUvPriStrength;
	const oa::U8* inCdefUvSecStrength = fh.cdefUvSecStrength;
	const StdVideoAV1FrameRestorationType* inRestorationTypes = fh.restorationTypes;
	const oa::U16* inRestorationSizes = fh.restorationSizes;
	const bool inUsesLr = fh.usesLr;
	const bool inUsesChromaLr = fh.usesChromaLr;
	const StdVideoAV1TxMode inTxMode = fh.txMode;
	const bool inReducedTxSet = fh.reducedTxSet;
	const bool inErrorResilientMode = fh.errorResilientMode;
	const oa::U32 inContextUpdateTileId = fh.contextUpdateTileId;
	const bool inFrameSizeOverrideFlag = fh.frameSizeOverrideFlag;
	const bool inUseSuperres = fh.useSuperres;
	const oa::U8 inCodedDenom = fh.codedDenom;
	const bool inRenderAndFrameSizeDifferent = fh.renderAndFrameSizeDifferent;
	const bool inAllowIntraBc = fh.allowIntraBc;
	const bool inAllowHighPrecisionMv = fh.allowHighPrecisionMv;
	const bool inIsMotionModeSwitchable = fh.isMotionModeSwitchable;
	const bool inUseRefFrameMvs = fh.useRefFrameMvs;
	const bool inFrameRefsShortSignaling = fh.frameRefsShortSignaling;
	const bool inForceIntegerMv = fh.forceIntegerMv;
	const bool inIsFilterSwitchable = fh.isFilterSwitchable;
	const StdVideoAV1InterpolationFilter inInterpolationFilter = fh.interpolationFilter;
	(void)inSegmentationEnabled;
	(void)inSegmentationUpdateMap;
	(void)inSegmentationTemporalUpdate;
	(void)inSegmentationUpdateData;
	(void)inDeltaQPresent;
	(void)inDeltaLfPresent;
	(void)inDeltaLfMulti;

	BitstreamSlot& bitstream = impl_->bitstreamRing[impl_->currentBitstreamIndex];
	if (!impl_->engine || impl_->session.handle() == VK_NULL_HANDLE || impl_->sessionParameters.handle() == VK_NULL_HANDLE || !impl_->commandBuffers[0] || bitstream.buffer.getBuffer() == VK_NULL_HANDLE) {
		return oa::Status::error("AV1 decoder command resources are not initialized");
	}
	const auto& deviceDispatch =
		oa::EngineDeviceAccess::get(*impl_->engine).deviceDispatch;
	if (!deviceDispatch.vkCmdBeginVideoCodingKHR
		|| !deviceDispatch.vkCmdDecodeVideoKHR
		|| !deviceDispatch.vkCmdEndVideoCodingKHR) {
		return oa::Status::error("vulkan Video decode command functions are not loaded");
	}
	if (inDpbSlot < 0 || static_cast<oa::U32>(inDpbSlot) >= impl_->dpbSlotCapacity) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 DPB slot");
	}
	if (inFrameSize == 0 || inFrameOffset + inFrameSize > bitstream.size) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame range");
	}
	if (inBitstreamBase > inDesc.frame.size) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame header offset");
	}
	oa::Vector<oa::U32> relTileOffsets;
	relTileOffsets.reserve(inTileOffsets.size());
	for (oa::Usize i = 0; i < inTileOffsets.size(); ++i) {
		if (inTileOffsets[i] < inBitstreamBase) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile offset");
		}
		relTileOffsets.pushBack(static_cast<oa::U32>(inTileOffsets[i] - inBitstreamBase));
	}
	if (relTileOffsets.empty() || relTileOffsets.size() != inTileSizes.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile metadata");
	}
	if (inTileOffsets.size() > STD_VIDEO_AV1_MAX_TILE_COLS * STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile count exceeds vulkan std-video limits");
	}
	if (inOrderHint > 0xffu) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 order hint");
	}
	if (inPrimaryRefFrame > STD_VIDEO_AV1_PRIMARY_REF_NONE || inRefreshFrameFlags > 0xffu) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 reference metadata");
	}
	if (inQmY > 15u || inQmU > 15u || inQmV > 15u) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 quantization matrix id");
	}
	for (oa::U32 i = 0; inReferenceNameSlotIndices && i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		const oa::I32 slot = inReferenceNameSlotIndices[i];
		if (slot < -1 || (slot >= 0 && static_cast<oa::U32>(slot) >= impl_->dpbSlotCapacity)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 reference slot index");
		}
	}
	if (inDeltaQRes > 3u || inDeltaLfRes > 3u) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 delta resolution");
	}
	for (oa::Usize i = 0; i < relTileOffsets.size(); ++i) {
		if (inTileSizes[i] == 0 || static_cast<oa::U64>(relTileOffsets[i]) + inTileSizes[i] > inFrameSize) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile range");
		}
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	oa::U32 setupDpbBaseLayer = 0;
	if (!oa::VideoDecoderRecordAccess::getDpbView(*this, inDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return oa::Status::error(oa::StatusCode::Unavailable, "AV1 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(oa::VideoDecoderRecordAccess::resolveOutputView(*this, inDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	oa::VideoDecoderRecordAccess::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, oa::VideoDecoderRecordAccess::begin(*this, "AV1 decode"));

	if (inTileCols == 0 || inTileRows == 0 || inTileCols > STD_VIDEO_AV1_MAX_TILE_COLS || inTileRows > STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile grid");
	}
	if (static_cast<oa::U64>(inTileCols) * inTileRows != inTileOffsets.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile grid does not match tile payload count");
	}

	// StdVideoAV1TileInfo::{pMiColStarts,pMiRowStarts} are defined in 4x4-MI
	// units (not superblock indices), despite the field names. The NVIDIA
	// vk_video_samples reference (VulkanAV1Decoder::DecodeTileInfo) shifts the
	// superblock start by sbShift before storing it here. pWidthInSbsMinus1 and
	// pHeightInSbsMinus1 remain in superblock units, as their names indicate.
	oa::Array<uint16_t, STD_VIDEO_AV1_MAX_TILE_COLS + 1> miColStarts = {};
	oa::Array<uint16_t, STD_VIDEO_AV1_MAX_TILE_ROWS + 1> miRowStarts = {};
	oa::Array<uint16_t, STD_VIDEO_AV1_MAX_TILE_COLS> widthInSbsMinus1 = {};
	oa::Array<uint16_t, STD_VIDEO_AV1_MAX_TILE_ROWS> heightInSbsMinus1 = {};
	const oa::U32 miCols = (impl_->codedWidth + 7u) >> 3u << 1u;
	const oa::U32 miRows = (impl_->codedHeight + 7u) >> 3u << 1u;
	const oa::U32 sbShift = inUse128x128Superblock ? 5u : 4u;
	const oa::U32 sbCols = (miCols + (1u << sbShift) - 1u) >> sbShift;
	const oa::U32 sbRows = (miRows + (1u << sbShift) - 1u) >> sbShift;
	const oa::U32 tileWidthSb = (sbCols + inTileCols - 1u) / inTileCols;
	const oa::U32 tileHeightSb = (sbRows + inTileRows - 1u) / inTileRows;
	for (oa::U32 i = 0; i <= inTileCols; ++i) {
		const oa::U32 startSb = i * tileWidthSb < sbCols ? i * tileWidthSb : sbCols;
		// The last entry is the frame boundary in MI units, not the superblock-aligned edge.
		miColStarts[i] = (i == inTileCols)
			? static_cast<uint16_t>(miCols)
			: static_cast<uint16_t>(startSb << sbShift);
		if (i > 0) {
			const oa::U32 prevStartSb = (i - 1u) * tileWidthSb < sbCols ? (i - 1u) * tileWidthSb : sbCols;
			widthInSbsMinus1[i - 1u] = static_cast<uint16_t>(startSb > prevStartSb ? startSb - prevStartSb - 1u : 0u);
		}
	}
	for (oa::U32 i = 0; i <= inTileRows; ++i) {
		const oa::U32 startSb = i * tileHeightSb < sbRows ? i * tileHeightSb : sbRows;
		// The last entry is the frame boundary in MI units, not the superblock-aligned edge.
		miRowStarts[i] = (i == inTileRows)
			? static_cast<uint16_t>(miRows)
			: static_cast<uint16_t>(startSb << sbShift);
		if (i > 0) {
			const oa::U32 prevStartSb = (i - 1u) * tileHeightSb < sbRows ? (i - 1u) * tileHeightSb : sbRows;
			heightInSbsMinus1[i - 1u] = static_cast<uint16_t>(startSb > prevStartSb ? startSb - prevStartSb - 1u : 0u);
		}
	}

	StdVideoAV1TileInfo tileInfo = {};
	tileInfo.flags.uniform_tile_spacing_flag = true;
	tileInfo.TileCols = static_cast<uint8_t>(inTileCols);
	tileInfo.TileRows = static_cast<uint8_t>(inTileRows);
	tileInfo.context_update_tile_id = static_cast<uint8_t>(inContextUpdateTileId);
	tileInfo.tile_size_bytes_minus_1 = static_cast<uint8_t>(inTileSizeBytesMinus1);
	tileInfo.pMiColStarts = miColStarts.data();
	tileInfo.pMiRowStarts = miRowStarts.data();
	tileInfo.pWidthInSbsMinus1 = widthInSbsMinus1.data();
	tileInfo.pHeightInSbsMinus1 = heightInSbsMinus1.data();

	StdVideoAV1Quantization quantization = {};
	quantization.base_q_idx = static_cast<uint8_t>(inBaseQIdx);
	quantization.DeltaQYDc = static_cast<int8_t>(inDeltaQYDc);
	quantization.DeltaQUDc = static_cast<int8_t>(inDeltaQUDc);
	quantization.DeltaQUAc = static_cast<int8_t>(inDeltaQUAc);
	quantization.DeltaQVDc = static_cast<int8_t>(inDeltaQVDc);
	quantization.DeltaQVAc = static_cast<int8_t>(inDeltaQVAc);
	quantization.flags.using_qmatrix = inUsingQMatrix;
	quantization.flags.diff_uv_delta = inDiffUvDelta;
	quantization.qm_y = static_cast<uint8_t>(inQmY);
	quantization.qm_u = static_cast<uint8_t>(inQmU);
	quantization.qm_v = static_cast<uint8_t>(inQmV);
	StdVideoAV1Segmentation segmentation = {};
	for (oa::U32 segment = 0; segment < STD_VIDEO_AV1_MAX_SEGMENTS; ++segment) {
		for (oa::U32 feature = 0; feature < STD_VIDEO_AV1_SEG_LVL_MAX; ++feature) {
			segmentation.FeatureEnabled[segment] |= static_cast<uint8_t>((inSegmentFeatureEnabled[segment][feature] ? 1u : 0u) << feature);
			segmentation.FeatureData[segment][feature] = inSegmentFeatureData[segment][feature];
		}
	}
	StdVideoAV1LoopFilter loopFilter = {};
	for (oa::U32 i = 0; i < 4; ++i) {
		loopFilter.loop_filter_level[i] = inLoopFilterLevels ? inLoopFilterLevels[i] : 0;
	}
	loopFilter.loop_filter_sharpness = static_cast<uint8_t>(inLoopFilterSharpness);
	loopFilter.flags.loop_filter_delta_enabled = inLoopFilterDeltaEnabled;
	loopFilter.flags.loop_filter_delta_update = inLoopFilterDeltaUpdate;
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME; ++i) {
		if (inLoopFilterUpdateRefDelta && inLoopFilterUpdateRefDelta[i]) {
			loopFilter.update_ref_delta |= static_cast<uint8_t>(1u << i);
		}
		loopFilter.loop_filter_ref_deltas[i] = inLoopFilterRefDeltas ? inLoopFilterRefDeltas[i] : 0;
	}
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS; ++i) {
		if (inLoopFilterUpdateModeDelta && inLoopFilterUpdateModeDelta[i]) {
			loopFilter.update_mode_delta |= static_cast<uint8_t>(1u << i);
		}
		loopFilter.loop_filter_mode_deltas[i] = inLoopFilterModeDeltas ? inLoopFilterModeDeltas[i] : 0;
	}
	StdVideoAV1CDEF cdef = {};
	cdef.cdef_damping_minus_3 = static_cast<uint8_t>(inCdefDampingMinus3);
	cdef.cdef_bits = static_cast<uint8_t>(inCdefBits);
	const oa::U32 cdefStrengthCount = 1u << (inCdefBits > 3 ? 3 : inCdefBits);
	for (oa::U32 i = 0; i < cdefStrengthCount; ++i) {
		cdef.cdef_y_pri_strength[i] = inCdefYPriStrength ? inCdefYPriStrength[i] : 0;
		cdef.cdef_y_sec_strength[i] = inCdefYSecStrength ? inCdefYSecStrength[i] : 0;
		cdef.cdef_uv_pri_strength[i] = inCdefUvPriStrength ? inCdefUvPriStrength[i] : 0;
		cdef.cdef_uv_sec_strength[i] = inCdefUvSecStrength ? inCdefUvSecStrength[i] : 0;
	}
	StdVideoAV1LoopRestoration loopRestoration = {};
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_MAX_NUM_PLANES; ++i) {
		loopRestoration.FrameRestorationType[i] = inRestorationTypes
			? inRestorationTypes[i]
			: STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE;
		loopRestoration.LoopRestorationSize[i] = inRestorationSizes ? inRestorationSizes[i] : 0;
	}
	StdVideoAV1GlobalMotion globalMotion = {};
	// AV1 identity global motion is an affine identity matrix in Q16, not an
	// all-zero matrix.  The std-video structure is consumed verbatim by the
	// implementation, so value-initialising gm_params would collapse both axes
	// to zero even though gmType defaults to IDENTITY.  Match the AV1 reference
	// parser's default_warp_params for every reference slot.
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		// AV1 transformation_type IDENTITY is encoded as zero; std-video keeps
		// gmType as a byte rather than publishing the bitstream enum.
		globalMotion.GmType[i] = 0;
		globalMotion.gm_params[i][2] = 1 << 16;
		globalMotion.gm_params[i][5] = 1 << 16;
	}
	StdVideoAV1FilmGrain filmGrain = {};

	StdVideoDecodeAV1PictureInfo stdPicture = {};
	stdPicture.flags.error_resilient_mode = inErrorResilientMode;
	stdPicture.flags.disable_cdf_update = inDisableCdfUpdate;
	stdPicture.flags.use_superres = inUseSuperres;
	stdPicture.flags.render_and_frame_size_different = inRenderAndFrameSizeDifferent;
	stdPicture.flags.allow_screen_content_tools = inAllowScreenContentTools;
	stdPicture.flags.is_filter_switchable = inIsFilterSwitchable;
	stdPicture.flags.force_integer_mv = inForceIntegerMv;
	stdPicture.flags.frame_size_override_flag = inFrameSizeOverrideFlag;
	stdPicture.flags.allow_intrabc = inAllowIntraBc;
	stdPicture.flags.frame_refs_short_signaling = inFrameRefsShortSignaling;
	stdPicture.flags.allow_high_precision_mv = inAllowHighPrecisionMv;
	stdPicture.flags.is_motion_mode_switchable = inIsMotionModeSwitchable;
	stdPicture.flags.use_ref_frame_mvs = inUseRefFrameMvs;
	stdPicture.flags.disable_frame_end_update_cdf = inDisableFrameEndUpdateCdf;
	stdPicture.flags.reduced_tx_set = inReducedTxSet;
	stdPicture.flags.reference_select = fh.referenceSelect;
	stdPicture.flags.skip_mode_present = fh.skipModePresent;
	stdPicture.flags.allow_warped_motion = fh.allowWarpedMotion;
	stdPicture.flags.segmentation_enabled = inSegmentationEnabled;
	stdPicture.flags.segmentation_update_map = inSegmentationUpdateMap;
	stdPicture.flags.segmentation_temporal_update = inSegmentationTemporalUpdate;
	stdPicture.flags.segmentation_update_data = inSegmentationUpdateData;
	stdPicture.flags.delta_q_present = inDeltaQPresent;
	stdPicture.flags.delta_lf_present = inDeltaLfPresent;
	stdPicture.flags.delta_lf_multi = inDeltaLfMulti;
	stdPicture.flags.UsesLr = inUsesLr;
	stdPicture.flags.usesChromaLr = inUsesChromaLr;
	stdPicture.flags.apply_grain = fh.applyGrain;
	stdPicture.frame_type = inFrameType;
	stdPicture.OrderHint = static_cast<uint8_t>(inOrderHint);
	stdPicture.primary_ref_frame = static_cast<uint8_t>(inPrimaryRefFrame);
	stdPicture.refresh_frame_flags = static_cast<uint8_t>(inRefreshFrameFlags);
	stdPicture.interpolation_filter = inInterpolationFilter;
	stdPicture.TxMode = inTxMode;
	stdPicture.delta_q_res = static_cast<uint8_t>(inDeltaQRes);
	stdPicture.delta_lf_res = static_cast<uint8_t>(inDeltaLfRes);
	stdPicture.coded_denom = inCodedDenom;
	stdPicture.SkipModeFrame[0] = fh.skipModeFrame[0];
	stdPicture.SkipModeFrame[1] = fh.skipModeFrame[1];
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		stdPicture.OrderHints[i] = inOrderHints ? inOrderHints[i] : 0;
	}
	stdPicture.pTileInfo = &tileInfo;
	stdPicture.pQuantization = &quantization;
	stdPicture.pSegmentation = &segmentation;
	stdPicture.pLoopFilter = &loopFilter;
	stdPicture.pCDEF = &cdef;
	stdPicture.pLoopRestoration = &loopRestoration;
	stdPicture.pGlobalMotion = &globalMotion;
	// The std-video contract only supplies film-grain parameters when grain is
	// actually applied. Passing a non-null zero structure for ordinary streams
	// disagrees with FFmpeg and can trigger implementation-specific processing.
	stdPicture.pFilmGrain = fh.applyGrain ? &filmGrain : nullptr;

	VkVideoDecodeAV1PictureInfoKHR av1Picture = {};
	av1Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR;
	av1Picture.pStdPictureInfo = &stdPicture;
	for (oa::U32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		av1Picture.referenceNameSlotIndices[i] = inReferenceNameSlotIndices ? inReferenceNameSlotIndices[i] : -1;
	}
	// vulkan wants the frame-header OBU offset, including its OBU header—not the
	// uncompressed-header payload offset. The uploaded range starts at that OBU.
	av1Picture.frameHeaderOffset = static_cast<uint32_t>(inFrameOffset);
	av1Picture.tileCount = static_cast<uint32_t>(relTileOffsets.size());
	av1Picture.pTileOffsets = relTileOffsets.data();
	av1Picture.pTileSizes = inTileSizes.data();

	StdVideoDecodeAV1ReferenceInfo setupStdRef = {};
	setupStdRef.frame_type = static_cast<uint8_t>(inFrameType);
	setupStdRef.OrderHint = static_cast<uint8_t>(inOrderHint);
	setupStdRef.RefFrameSignBias = fh.refFrameSignBias;
	setupStdRef.flags.disable_frame_end_update_cdf = inDisableFrameEndUpdateCdf;
	setupStdRef.flags.segmentation_enabled = inSegmentationEnabled;
	for (oa::U32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		setupStdRef.SavedOrderHints[i] = inOrderHints ? inOrderHints[i] : 0;
	}
	impl_->av1DpbReferenceInfos[inDpbSlot] = setupStdRef;

	VkVideoDecodeAV1DpbSlotInfoKHR setupAV1Slot = {};
	setupAV1Slot.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR;
	setupAV1Slot.pStdReferenceInfo = &impl_->av1DpbReferenceInfos[inDpbSlot];

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = pictureExtent;
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupAV1Slot;
	setupSlot.slotIndex = inDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = pictureExtent;
	// dst layer must match setup layer when reusing the DPB image as output.
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	// Build the active reference slots from the per-name DPB indices the parser
	// resolved (inReferenceNameSlotIndices). Each distinct slot is bound once,
	// with its picture resource + AV1 reference info, so vkCmdDecodeVideoKHR can
	// reference it by slotIndex. Without this, inter frames pass
	// referenceNameSlotIndices pointing at unbound slots — tripping
	// VUID-vkCmdDecodeVideoKHR-referenceNameSlotIndices-09262 and crashing the
	// driver. Mirrors the H.264/H.265/VP9 paths.
	oa::Array<VkVideoPictureResourceInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refResources = {};
	oa::Array<StdVideoDecodeAV1ReferenceInfo, STD_VIDEO_AV1_NUM_REF_FRAMES> refStdInfos = {};
	oa::Array<VkVideoDecodeAV1DpbSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refDpbInfos = {};
	oa::Array<VkVideoReferenceSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refSlots = {};
	oa::U32 refCount = 0;
	auto refAlreadyBound = [&](oa::I32 s) -> bool {
		for (oa::U32 j = 0; j < refCount; ++j) {
			if (refSlots[j].slotIndex == s) { return true; }
		}
		return false;
	};
	if (inReferenceNameSlotIndices) {
		for (oa::U32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
			const oa::I32 s = inReferenceNameSlotIndices[i];
			// A reference must never alias the setup/destination slot — that would
			// bind one image layer as both the reconstructed picture and a reference
			// (VUID-07238/07176/07239). The unbound-name pass below maps the offending
			// name index to -1 so the std picture info stays consistent.
			if (s < 0 || static_cast<oa::U32>(s) >= impl_->dpbSlotCapacity || s == inDpbSlot || refAlreadyBound(s)) {
				continue;
			}
			VkImageView refView = VK_NULL_HANDLE;
			oa::U32 refBaseLayer = 0;
			if (!oa::VideoDecoderRecordAccess::getDpbView(*this, s, refView, refBaseLayer)) {
				continue;
			}
			refStdInfos[refCount] = impl_->av1DpbReferenceInfos[s];
			refDpbInfos[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR;
			refDpbInfos[refCount].pStdReferenceInfo = &refStdInfos[refCount];
			refResources[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
			refResources[refCount].codedExtent = pictureExtent;
			refResources[refCount].baseArrayLayer = refBaseLayer;
			refResources[refCount].imageViewBinding = refView;
			refSlots[refCount].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
			refSlots[refCount].pNext = &refDpbInfos[refCount];
			refSlots[refCount].slotIndex = s;
			refSlots[refCount].pPictureResource = &refResources[refCount];
			++refCount;
		}
	}
	// Any reference name resolving to an unbound slot must be marked unused (-1),
	// or the driver rejects it (09262).
	for (oa::U32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		if (av1Picture.referenceNameSlotIndices[i] >= 0 && !refAlreadyBound(av1Picture.referenceNameSlotIndices[i])) {
			av1Picture.referenceNameSlotIndices[i] = -1;
		}
	}

	// Mark DPB slot as active BEFORE adding to reference slots (AV1)
	// This is required by vulkan spec VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239
	if (!impl_->dpbSlots[inDpbSlot].inUse) {
		impl_->dpbSlots[inDpbSlot].inUse = true;
		impl_->dpbSlots[inDpbSlot].frameNumber = impl_->currentFrameNumber;
		impl_->dpbSlots[inDpbSlot].picOrderCnt = static_cast<oa::I32>(inOrderHint);
	}

	// BeginCoding binds every active reference slot plus the setup slot.
	oa::Array<VkVideoReferenceSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES + 1> beginRefSlots = {};
	for (oa::U32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// The current reconstruction picture is bound for layout/resource tracking,
	// but is not an active reference until this decode completes. FFmpeg's
	// vulkan decoder uses the same inactive (-1) begin-coding association and
	// supplies the real slot only through decodeInfo.pSetupReferenceSlot.
	beginRefSlots[refCount].slotIndex = -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = impl_->session.handle();
	beginCoding.videoSessionParameters = impl_->sessionParameters.handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.data();
	deviceDispatch.vkCmdBeginVideoCodingKHR(cmd.cb, &beginCoding);

	oa::VideoDecoderRecordAccess::resetSessionIfNeeded(cmd, *this);

	// Match NVIDIA VkVideoDecoder ordering: BeginCoding → reset → image barriers → decode.
	oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, inDpbSlot);
	for (oa::U32 i = 0; i < refCount; ++i) {
		oa::VideoDecoderRecordAccess::ensureDpbLayer(cmd, *this, refSlots[i].slotIndex);
	}
	oa::VideoDecoderRecordAccess::ensureDistinctOutput(cmd, *this, inDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &av1Picture;
	// srcBufferRange must be a multiple of the profile's minBitstreamBufferSizeAlignment.
	const oa::U64 av1SizeAlignment = bitstream.buffer.getSizeAlignment() == 0 ? 1 : bitstream.buffer.getSizeAlignment();
	decodeInfo.srcBuffer = bitstream.buffer.getBuffer();
	decodeInfo.srcBufferOffset = static_cast<VkDeviceSize>(inFrameOffset);
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(oa::alignUp(static_cast<oa::Usize>(inFrameSize), static_cast<oa::Usize>(av1SizeAlignment)));
	decodeInfo.dstPictureResource = dstResource;
	decodeInfo.pSetupReferenceSlot = &setupSlot;
	decodeInfo.referenceSlotCount = refCount;
	decodeInfo.pReferenceSlots = refCount > 0 ? refSlots.data() : nullptr;

	oa::VideoDecoderRecordAccess::emitBitstreamDecodeBarrier(
		cmd,
		decodeInfo.srcBuffer,
		decodeInfo.srcBufferOffset,
		decodeInfo.srcBufferRange
	);

	deviceDispatch.vkCmdDecodeVideoKHR(cmd.cb, &decodeInfo);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	deviceDispatch.vkCmdEndVideoCodingKHR(cmd.cb, &endCoding);

	return oa::VideoDecoderRecordAccess::finishAndSubmit(*this, cmd, {
		.dpbSlot = inDpbSlot,
		.hasDistinctOutput = hasDistinctOutput,
		.errorContext = "AV1 video decode",
	});
}
