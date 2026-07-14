// AV1 vkCmdDecodeVideoKHR record path (FnVideoDecoder* impl).

#include "FnVideoDecoderAv1.h"
#include "../FnVideoDecoderRecordShared.h"
#include "../../../Video/Codec/VcpAv1.h"

OaStatus OaVideoDecoder::RecordAV1DecodeCommands(
	OaI32 InDpbSlot,
	const OaAv1PictureDesc& InDesc,
	const OaI32 InReferenceNameSlotIndices[OaAv1MaxReferencesPerFrame])
{
	const OaAv1FrameHeaderInfo& fh = InDesc.FrameHeader;
	const OaAv1SequenceHeaderInfo& seq = InDesc.SequenceHeader;
	// FnVideoDecoderAv1 uploads the decode OBU slice (see DecodeFrame).
	const OaUsize InFrameOffset = 0;
	const OaUsize InBitstreamBase = InDesc.DecodeObuOffset;
	const OaUsize InFrameSize = InDesc.DecodeObuSize;
	const OaSpan<const OaU32> InTileOffsets(InDesc.TileOffsets.Data(), InDesc.TileOffsets.Size());
	const OaSpan<const OaU32> InTileSizes(InDesc.TileSizes.Data(), InDesc.TileSizes.Size());
	const OaU32 InTileCols = fh.TileCols;
	const OaU32 InTileRows = fh.TileRows;
	const OaU32 InTileSizeBytesMinus1 = fh.TileSizeBytesMinus1;
	const VkExtent2D pictureExtent = {
		seq.MaxFrameWidthMinus1 + 1u,
		seq.MaxFrameHeightMinus1 + 1u};
	const StdVideoAV1FrameType InFrameType = fh.FrameType;
	const OaU32 InOrderHint = fh.OrderHint;
	const OaU32 InPrimaryRefFrame = fh.PrimaryRefFrame;
	const OaU32 InRefreshFrameFlags = fh.RefreshFrameFlags;
	const OaU8* InOrderHints = fh.OrderHints;
	const bool InUse128x128Superblock = seq.Use128x128Superblock;
	const bool InDisableCdfUpdate = fh.DisableCdfUpdate;
	const bool InDisableFrameEndUpdateCdf = fh.DisableFrameEndUpdateCdf;
	const bool InAllowScreenContentTools = fh.AllowScreenContentTools;
	const OaU32 InBaseQIdx = fh.BaseQIdx;
	const OaI32 InDeltaQYDc = fh.DeltaQYDc;
	const OaI32 InDeltaQUDc = fh.DeltaQUDc;
	const OaI32 InDeltaQUAc = fh.DeltaQUAc;
	const OaI32 InDeltaQVDc = fh.DeltaQVDc;
	const OaI32 InDeltaQVAc = fh.DeltaQVAc;
	const bool InUsingQMatrix = fh.UsingQMatrix;
	const bool InDiffUvDelta = fh.DiffUvDelta;
	const OaU32 InQmY = fh.QmY;
	const OaU32 InQmU = fh.QmU;
	const OaU32 InQmV = fh.QmV;
	const bool InSegmentationEnabled = fh.SegmentationEnabled;
	const bool InSegmentationUpdateMap = fh.SegmentationUpdateMap;
	const bool InSegmentationTemporalUpdate = fh.SegmentationTemporalUpdate;
	const bool InSegmentationUpdateData = fh.SegmentationUpdateData;
	const OaU8 (&InSegmentFeatureEnabled)[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = fh.SegmentFeatureEnabled;
	const OaI16 (&InSegmentFeatureData)[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX] = fh.SegmentFeatureData;
	const bool InDeltaQPresent = fh.DeltaQPresent;
	const OaU32 InDeltaQRes = fh.DeltaQRes;
	const bool InDeltaLfPresent = fh.DeltaLfPresent;
	const OaU32 InDeltaLfRes = fh.DeltaLfRes;
	const bool InDeltaLfMulti = fh.DeltaLfMulti;
	const OaU8* InLoopFilterLevels = fh.LoopFilterLevels;
	const OaU32 InLoopFilterSharpness = fh.LoopFilterSharpness;
	const bool InLoopFilterDeltaEnabled = fh.LoopFilterDeltaEnabled;
	const bool InLoopFilterDeltaUpdate = fh.LoopFilterDeltaUpdate;
	const OaU8* InLoopFilterUpdateRefDelta = fh.LoopFilterUpdateRefDelta;
	const OaI8* InLoopFilterRefDeltas = fh.LoopFilterRefDeltas;
	const OaU8* InLoopFilterUpdateModeDelta = fh.LoopFilterUpdateModeDelta;
	const OaI8* InLoopFilterModeDeltas = fh.LoopFilterModeDeltas;
	const OaU32 InCdefDampingMinus3 = fh.CdefDampingMinus3;
	const OaU32 InCdefBits = fh.CdefBits;
	const OaU8* InCdefYPriStrength = fh.CdefYPriStrength;
	const OaU8* InCdefYSecStrength = fh.CdefYSecStrength;
	const OaU8* InCdefUvPriStrength = fh.CdefUvPriStrength;
	const OaU8* InCdefUvSecStrength = fh.CdefUvSecStrength;
	const StdVideoAV1FrameRestorationType* InRestorationTypes = fh.RestorationTypes;
	const OaU16* InRestorationSizes = fh.RestorationSizes;
	const bool InUsesLr = fh.UsesLr;
	const bool InUsesChromaLr = fh.UsesChromaLr;
	const StdVideoAV1TxMode InTxMode = fh.TxMode;
	const bool InReducedTxSet = fh.ReducedTxSet;
	const bool InErrorResilientMode = fh.ErrorResilientMode;
	const OaU32 InContextUpdateTileId = fh.ContextUpdateTileId;
	const bool InFrameSizeOverrideFlag = fh.FrameSizeOverrideFlag;
	const bool InUseSuperres = fh.UseSuperres;
	const OaU8 InCodedDenom = fh.CodedDenom;
	const bool InRenderAndFrameSizeDifferent = fh.RenderAndFrameSizeDifferent;
	const bool InAllowIntraBc = fh.AllowIntraBc;
	const bool InAllowHighPrecisionMv = fh.AllowHighPrecisionMv;
	const bool InIsMotionModeSwitchable = fh.IsMotionModeSwitchable;
	const bool InUseRefFrameMvs = fh.UseRefFrameMvs;
	const bool InFrameRefsShortSignaling = fh.FrameRefsShortSignaling;
	const bool InForceIntegerMv = fh.ForceIntegerMv;
	const bool InIsFilterSwitchable = fh.IsFilterSwitchable;
	const StdVideoAV1InterpolationFilter InInterpolationFilter = fh.InterpolationFilter;
	(void)InSegmentationEnabled;
	(void)InSegmentationUpdateMap;
	(void)InSegmentationTemporalUpdate;
	(void)InSegmentationUpdateData;
	(void)InDeltaQPresent;
	(void)InDeltaLfPresent;
	(void)InDeltaLfMulti;

	BitstreamSlot& bitstream = BitstreamRing_[CurrentBitstreamIndex_];
	if (!Rt_ || Session_.Handle() == VK_NULL_HANDLE || SessionParams_.Handle() == VK_NULL_HANDLE || !CmdBuffers_[0] || bitstream.Buffer.GetBuffer() == VK_NULL_HANDLE) {
		return OaStatus::Error("AV1 decoder command resources are not initialized");
	}
	if (!vkCmdBeginVideoCodingKHR || !vkCmdDecodeVideoKHR || !vkCmdEndVideoCodingKHR) {
		return OaStatus::Error("Vulkan Video decode command functions are not loaded");
	}
	if (InDpbSlot < 0 || static_cast<OaU32>(InDpbSlot) >= DpbSlotCapacity_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 DPB slot");
	}
	if (InFrameSize == 0 || InFrameOffset + InFrameSize > bitstream.Size) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame range");
	}
	if (InBitstreamBase > InDesc.Frame.Size) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame header offset");
	}
	OaVec<OaU32> relTileOffsets;
	relTileOffsets.Reserve(InTileOffsets.Size());
	for (OaUsize i = 0; i < InTileOffsets.Size(); ++i) {
		if (InTileOffsets[i] < InBitstreamBase) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile offset");
		}
		relTileOffsets.PushBack(static_cast<OaU32>(InTileOffsets[i] - InBitstreamBase));
	}
	if (relTileOffsets.Empty() || relTileOffsets.Size() != InTileSizes.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile metadata");
	}
	if (InTileOffsets.Size() > STD_VIDEO_AV1_MAX_TILE_COLS * STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile count exceeds Vulkan std-video limits");
	}
	if (InOrderHint > 0xffu) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 order hint");
	}
	if (InPrimaryRefFrame > STD_VIDEO_AV1_PRIMARY_REF_NONE || InRefreshFrameFlags > 0xffu) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 reference metadata");
	}
	if (InQmY > 15u || InQmU > 15u || InQmV > 15u) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 quantization matrix id");
	}
	for (OaU32 i = 0; InReferenceNameSlotIndices && i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		const OaI32 slot = InReferenceNameSlotIndices[i];
		if (slot < -1 || (slot >= 0 && static_cast<OaU32>(slot) >= DpbSlotCapacity_)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 reference slot index");
		}
	}
	if (InDeltaQRes > 3u || InDeltaLfRes > 3u) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 delta resolution");
	}
	for (OaUsize i = 0; i < relTileOffsets.Size(); ++i) {
		if (InTileSizes[i] == 0 || static_cast<OaU64>(relTileOffsets[i]) + InTileSizes[i] > InFrameSize) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile range");
		}
	}

	VkImageView setupDpbView = VK_NULL_HANDLE;
	OaU32 setupDpbBaseLayer = 0;
	if (!OaFnVideoDecoderRecord::GetDpbView(*this, InDpbSlot, setupDpbView, setupDpbBaseLayer)) {
		return OaStatus::Error(OaStatusCode::Unavailable, "AV1 decode requires profile-compatible DPB image views");
	}

	VkImageView dstView = VK_NULL_HANDLE;
	bool hasDistinctOutput = false;
	OA_RETURN_IF_ERROR(OaFnVideoDecoderRecord::ResolveOutputView(*this, InDpbSlot, setupDpbView, dstView, hasDistinctOutput));

	OaFnVideoDecoderRecord::ActiveCmd cmd;
	OA_ASSIGN_OR_RETURN(cmd, OaFnVideoDecoderRecord::Begin(*this, "AV1 decode"));

	if (InTileCols == 0 || InTileRows == 0 || InTileCols > STD_VIDEO_AV1_MAX_TILE_COLS || InTileRows > STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile grid");
	}
	if (static_cast<OaU64>(InTileCols) * InTileRows != InTileOffsets.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile grid does not match tile payload count");
	}

	// StdVideoAV1TileInfo::{pMiColStarts,pMiRowStarts} are defined in 4x4-MI
	// units (not superblock indices), despite the field names. The NVIDIA
	// vk_video_samples reference (VulkanAV1Decoder::DecodeTileInfo) shifts the
	// superblock start by sbShift before storing it here. pWidthInSbsMinus1 and
	// pHeightInSbsMinus1 remain in superblock units, as their names indicate.
	OaArray<uint16_t, STD_VIDEO_AV1_MAX_TILE_COLS + 1> miColStarts = {};
	OaArray<uint16_t, STD_VIDEO_AV1_MAX_TILE_ROWS + 1> miRowStarts = {};
	OaArray<uint16_t, STD_VIDEO_AV1_MAX_TILE_COLS> widthInSbsMinus1 = {};
	OaArray<uint16_t, STD_VIDEO_AV1_MAX_TILE_ROWS> heightInSbsMinus1 = {};
	const OaU32 miCols = (CodedWidth_ + 7u) >> 3u << 1u;
	const OaU32 miRows = (CodedHeight_ + 7u) >> 3u << 1u;
	const OaU32 sbShift = InUse128x128Superblock ? 5u : 4u;
	const OaU32 sbCols = (miCols + (1u << sbShift) - 1u) >> sbShift;
	const OaU32 sbRows = (miRows + (1u << sbShift) - 1u) >> sbShift;
	const OaU32 tileWidthSb = (sbCols + InTileCols - 1u) / InTileCols;
	const OaU32 tileHeightSb = (sbRows + InTileRows - 1u) / InTileRows;
	for (OaU32 i = 0; i <= InTileCols; ++i) {
		const OaU32 startSb = i * tileWidthSb < sbCols ? i * tileWidthSb : sbCols;
		// The last entry is the frame boundary in MI units, not the superblock-aligned edge.
		miColStarts[i] = (i == InTileCols)
			? static_cast<uint16_t>(miCols)
			: static_cast<uint16_t>(startSb << sbShift);
		if (i > 0) {
			const OaU32 prevStartSb = (i - 1u) * tileWidthSb < sbCols ? (i - 1u) * tileWidthSb : sbCols;
			widthInSbsMinus1[i - 1u] = static_cast<uint16_t>(startSb > prevStartSb ? startSb - prevStartSb - 1u : 0u);
		}
	}
	for (OaU32 i = 0; i <= InTileRows; ++i) {
		const OaU32 startSb = i * tileHeightSb < sbRows ? i * tileHeightSb : sbRows;
		// The last entry is the frame boundary in MI units, not the superblock-aligned edge.
		miRowStarts[i] = (i == InTileRows)
			? static_cast<uint16_t>(miRows)
			: static_cast<uint16_t>(startSb << sbShift);
		if (i > 0) {
			const OaU32 prevStartSb = (i - 1u) * tileHeightSb < sbRows ? (i - 1u) * tileHeightSb : sbRows;
			heightInSbsMinus1[i - 1u] = static_cast<uint16_t>(startSb > prevStartSb ? startSb - prevStartSb - 1u : 0u);
		}
	}

	StdVideoAV1TileInfo tileInfo = {};
	tileInfo.flags.uniform_tile_spacing_flag = true;
	tileInfo.TileCols = static_cast<uint8_t>(InTileCols);
	tileInfo.TileRows = static_cast<uint8_t>(InTileRows);
	tileInfo.context_update_tile_id = static_cast<uint8_t>(InContextUpdateTileId);
	tileInfo.tile_size_bytes_minus_1 = static_cast<uint8_t>(InTileSizeBytesMinus1);
	tileInfo.pMiColStarts = miColStarts.Data();
	tileInfo.pMiRowStarts = miRowStarts.Data();
	tileInfo.pWidthInSbsMinus1 = widthInSbsMinus1.Data();
	tileInfo.pHeightInSbsMinus1 = heightInSbsMinus1.Data();

	StdVideoAV1Quantization quantization = {};
	quantization.base_q_idx = static_cast<uint8_t>(InBaseQIdx);
	quantization.DeltaQYDc = static_cast<int8_t>(InDeltaQYDc);
	quantization.DeltaQUDc = static_cast<int8_t>(InDeltaQUDc);
	quantization.DeltaQUAc = static_cast<int8_t>(InDeltaQUAc);
	quantization.DeltaQVDc = static_cast<int8_t>(InDeltaQVDc);
	quantization.DeltaQVAc = static_cast<int8_t>(InDeltaQVAc);
	quantization.flags.using_qmatrix = InUsingQMatrix;
	quantization.flags.diff_uv_delta = InDiffUvDelta;
	quantization.qm_y = static_cast<uint8_t>(InQmY);
	quantization.qm_u = static_cast<uint8_t>(InQmU);
	quantization.qm_v = static_cast<uint8_t>(InQmV);
	StdVideoAV1Segmentation segmentation = {};
	for (OaU32 segment = 0; segment < STD_VIDEO_AV1_MAX_SEGMENTS; ++segment) {
		for (OaU32 feature = 0; feature < STD_VIDEO_AV1_SEG_LVL_MAX; ++feature) {
			segmentation.FeatureEnabled[segment] |= static_cast<uint8_t>((InSegmentFeatureEnabled[segment][feature] ? 1u : 0u) << feature);
			segmentation.FeatureData[segment][feature] = InSegmentFeatureData[segment][feature];
		}
	}
	StdVideoAV1LoopFilter loopFilter = {};
	for (OaU32 i = 0; i < 4; ++i) {
		loopFilter.loop_filter_level[i] = InLoopFilterLevels ? InLoopFilterLevels[i] : 0;
	}
	loopFilter.loop_filter_sharpness = static_cast<uint8_t>(InLoopFilterSharpness);
	loopFilter.flags.loop_filter_delta_enabled = InLoopFilterDeltaEnabled;
	loopFilter.flags.loop_filter_delta_update = InLoopFilterDeltaUpdate;
	for (OaU32 i = 0; i < STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME; ++i) {
		if (InLoopFilterUpdateRefDelta && InLoopFilterUpdateRefDelta[i]) {
			loopFilter.update_ref_delta |= static_cast<uint8_t>(1u << i);
		}
		loopFilter.loop_filter_ref_deltas[i] = InLoopFilterRefDeltas ? InLoopFilterRefDeltas[i] : 0;
	}
	for (OaU32 i = 0; i < STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS; ++i) {
		if (InLoopFilterUpdateModeDelta && InLoopFilterUpdateModeDelta[i]) {
			loopFilter.update_mode_delta |= static_cast<uint8_t>(1u << i);
		}
		loopFilter.loop_filter_mode_deltas[i] = InLoopFilterModeDeltas ? InLoopFilterModeDeltas[i] : 0;
	}
	StdVideoAV1CDEF cdef = {};
	cdef.cdef_damping_minus_3 = static_cast<uint8_t>(InCdefDampingMinus3);
	cdef.cdef_bits = static_cast<uint8_t>(InCdefBits);
	const OaU32 cdefStrengthCount = 1u << (InCdefBits > 3 ? 3 : InCdefBits);
	for (OaU32 i = 0; i < cdefStrengthCount; ++i) {
		cdef.cdef_y_pri_strength[i] = InCdefYPriStrength ? InCdefYPriStrength[i] : 0;
		cdef.cdef_y_sec_strength[i] = InCdefYSecStrength ? InCdefYSecStrength[i] : 0;
		cdef.cdef_uv_pri_strength[i] = InCdefUvPriStrength ? InCdefUvPriStrength[i] : 0;
		cdef.cdef_uv_sec_strength[i] = InCdefUvSecStrength ? InCdefUvSecStrength[i] : 0;
	}
	StdVideoAV1LoopRestoration loopRestoration = {};
	for (OaU32 i = 0; i < STD_VIDEO_AV1_MAX_NUM_PLANES; ++i) {
		loopRestoration.FrameRestorationType[i] = InRestorationTypes
			? InRestorationTypes[i]
			: STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE;
		loopRestoration.LoopRestorationSize[i] = InRestorationSizes ? InRestorationSizes[i] : 0;
	}
	StdVideoAV1GlobalMotion globalMotion = {};
	// AV1 identity global motion is an affine identity matrix in Q16, not an
	// all-zero matrix.  The std-video structure is consumed verbatim by the
	// implementation, so value-initialising gm_params would collapse both axes
	// to zero even though GmType defaults to IDENTITY.  Match the AV1 reference
	// parser's default_warp_params for every reference slot.
	for (OaU32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		// AV1 transformation_type IDENTITY is encoded as zero; std-video keeps
		// GmType as a byte rather than publishing the bitstream enum.
		globalMotion.GmType[i] = 0;
		globalMotion.gm_params[i][2] = 1 << 16;
		globalMotion.gm_params[i][5] = 1 << 16;
	}
	StdVideoAV1FilmGrain filmGrain = {};

	StdVideoDecodeAV1PictureInfo stdPicture = {};
	stdPicture.flags.error_resilient_mode = InErrorResilientMode;
	stdPicture.flags.disable_cdf_update = InDisableCdfUpdate;
	stdPicture.flags.use_superres = InUseSuperres;
	stdPicture.flags.render_and_frame_size_different = InRenderAndFrameSizeDifferent;
	stdPicture.flags.allow_screen_content_tools = InAllowScreenContentTools;
	stdPicture.flags.is_filter_switchable = InIsFilterSwitchable;
	stdPicture.flags.force_integer_mv = InForceIntegerMv;
	stdPicture.flags.frame_size_override_flag = InFrameSizeOverrideFlag;
	stdPicture.flags.allow_intrabc = InAllowIntraBc;
	stdPicture.flags.frame_refs_short_signaling = InFrameRefsShortSignaling;
	stdPicture.flags.allow_high_precision_mv = InAllowHighPrecisionMv;
	stdPicture.flags.is_motion_mode_switchable = InIsMotionModeSwitchable;
	stdPicture.flags.use_ref_frame_mvs = InUseRefFrameMvs;
	stdPicture.flags.disable_frame_end_update_cdf = InDisableFrameEndUpdateCdf;
	stdPicture.flags.reduced_tx_set = InReducedTxSet;
	stdPicture.flags.reference_select = fh.ReferenceSelect;
	stdPicture.flags.skip_mode_present = fh.SkipModePresent;
	stdPicture.flags.allow_warped_motion = fh.AllowWarpedMotion;
	stdPicture.flags.segmentation_enabled = InSegmentationEnabled;
	stdPicture.flags.segmentation_update_map = InSegmentationUpdateMap;
	stdPicture.flags.segmentation_temporal_update = InSegmentationTemporalUpdate;
	stdPicture.flags.segmentation_update_data = InSegmentationUpdateData;
	stdPicture.flags.delta_q_present = InDeltaQPresent;
	stdPicture.flags.delta_lf_present = InDeltaLfPresent;
	stdPicture.flags.delta_lf_multi = InDeltaLfMulti;
	stdPicture.flags.UsesLr = InUsesLr;
	stdPicture.flags.usesChromaLr = InUsesChromaLr;
	stdPicture.flags.apply_grain = fh.ApplyGrain;
	stdPicture.frame_type = InFrameType;
	stdPicture.OrderHint = static_cast<uint8_t>(InOrderHint);
	stdPicture.primary_ref_frame = static_cast<uint8_t>(InPrimaryRefFrame);
	stdPicture.refresh_frame_flags = static_cast<uint8_t>(InRefreshFrameFlags);
	stdPicture.interpolation_filter = InInterpolationFilter;
	stdPicture.TxMode = InTxMode;
	stdPicture.delta_q_res = static_cast<uint8_t>(InDeltaQRes);
	stdPicture.delta_lf_res = static_cast<uint8_t>(InDeltaLfRes);
	stdPicture.coded_denom = InCodedDenom;
	stdPicture.SkipModeFrame[0] = fh.SkipModeFrame[0];
	stdPicture.SkipModeFrame[1] = fh.SkipModeFrame[1];
	for (OaU32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		stdPicture.OrderHints[i] = InOrderHints ? InOrderHints[i] : 0;
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
	stdPicture.pFilmGrain = fh.ApplyGrain ? &filmGrain : nullptr;

	VkVideoDecodeAV1PictureInfoKHR av1Picture = {};
	av1Picture.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR;
	av1Picture.pStdPictureInfo = &stdPicture;
	for (OaU32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		av1Picture.referenceNameSlotIndices[i] = InReferenceNameSlotIndices ? InReferenceNameSlotIndices[i] : -1;
	}
	// Vulkan wants the frame-header OBU offset, including its OBU header—not the
	// uncompressed-header payload offset. The uploaded range starts at that OBU.
	av1Picture.frameHeaderOffset = static_cast<uint32_t>(InFrameOffset);
	av1Picture.tileCount = static_cast<uint32_t>(relTileOffsets.Size());
	av1Picture.pTileOffsets = relTileOffsets.Data();
	av1Picture.pTileSizes = InTileSizes.Data();

	StdVideoDecodeAV1ReferenceInfo setupStdRef = {};
	setupStdRef.frame_type = static_cast<uint8_t>(InFrameType);
	setupStdRef.OrderHint = static_cast<uint8_t>(InOrderHint);
	setupStdRef.RefFrameSignBias = fh.RefFrameSignBias;
	setupStdRef.flags.disable_frame_end_update_cdf = InDisableFrameEndUpdateCdf;
	setupStdRef.flags.segmentation_enabled = InSegmentationEnabled;
	for (OaU32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
		setupStdRef.SavedOrderHints[i] = InOrderHints ? InOrderHints[i] : 0;
	}
	Av1DpbReferenceInfos_[InDpbSlot] = setupStdRef;

	VkVideoDecodeAV1DpbSlotInfoKHR setupAV1Slot = {};
	setupAV1Slot.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR;
	setupAV1Slot.pStdReferenceInfo = &Av1DpbReferenceInfos_[InDpbSlot];

	VkVideoPictureResourceInfoKHR setupResource = {};
	setupResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	setupResource.codedExtent = pictureExtent;
	setupResource.baseArrayLayer = setupDpbBaseLayer;
	setupResource.imageViewBinding = setupDpbView;

	VkVideoReferenceSlotInfoKHR setupSlot = {};
	setupSlot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
	setupSlot.pNext = &setupAV1Slot;
	setupSlot.slotIndex = InDpbSlot;
	setupSlot.pPictureResource = &setupResource;

	VkVideoPictureResourceInfoKHR dstResource = {};
	dstResource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
	dstResource.codedExtent = pictureExtent;
	// dst layer must match setup layer when reusing the DPB image as output.
	dstResource.baseArrayLayer = hasDistinctOutput ? 0u : setupDpbBaseLayer;
	dstResource.imageViewBinding = dstView;

	// Build the active reference slots from the per-name DPB indices the parser
	// resolved (InReferenceNameSlotIndices). Each distinct slot is bound once,
	// with its picture resource + AV1 reference info, so vkCmdDecodeVideoKHR can
	// reference it by slotIndex. Without this, inter frames pass
	// referenceNameSlotIndices pointing at unbound slots — tripping
	// VUID-vkCmdDecodeVideoKHR-referenceNameSlotIndices-09262 and crashing the
	// driver. Mirrors the H.264/H.265/VP9 paths.
	OaArray<VkVideoPictureResourceInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refResources = {};
	OaArray<StdVideoDecodeAV1ReferenceInfo, STD_VIDEO_AV1_NUM_REF_FRAMES> refStdInfos = {};
	OaArray<VkVideoDecodeAV1DpbSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refDpbInfos = {};
	OaArray<VkVideoReferenceSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES> refSlots = {};
	OaU32 refCount = 0;
	auto refAlreadyBound = [&](OaI32 s) -> bool {
		for (OaU32 j = 0; j < refCount; ++j) {
			if (refSlots[j].slotIndex == s) { return true; }
		}
		return false;
	};
	if (InReferenceNameSlotIndices) {
		for (OaU32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
			const OaI32 s = InReferenceNameSlotIndices[i];
			// A reference must never alias the setup/destination slot — that would
			// bind one image layer as both the reconstructed picture and a reference
			// (VUID-07238/07176/07239). The unbound-name pass below maps the offending
			// name index to -1 so the std picture info stays consistent.
			if (s < 0 || static_cast<OaU32>(s) >= DpbSlotCapacity_ || s == InDpbSlot || refAlreadyBound(s)) {
				continue;
			}
			VkImageView refView = VK_NULL_HANDLE;
			OaU32 refBaseLayer = 0;
			if (!OaFnVideoDecoderRecord::GetDpbView(*this, s, refView, refBaseLayer)) {
				continue;
			}
			refStdInfos[refCount] = Av1DpbReferenceInfos_[s];
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
	for (OaU32 i = 0; i < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++i) {
		if (av1Picture.referenceNameSlotIndices[i] >= 0 && !refAlreadyBound(av1Picture.referenceNameSlotIndices[i])) {
			av1Picture.referenceNameSlotIndices[i] = -1;
		}
	}

	// Mark DPB slot as active BEFORE adding to reference slots (AV1)
	// This is required by Vulkan spec VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239
	if (!DpbSlots_[InDpbSlot].InUse) {
		DpbSlots_[InDpbSlot].InUse = true;
		DpbSlots_[InDpbSlot].FrameNumber = CurrentFrameNumber_;
		DpbSlots_[InDpbSlot].PicOrderCnt = static_cast<OaI32>(InOrderHint);
	}

	// BeginCoding binds every active reference slot plus the setup slot.
	OaArray<VkVideoReferenceSlotInfoKHR, STD_VIDEO_AV1_NUM_REF_FRAMES + 1> beginRefSlots = {};
	for (OaU32 i = 0; i < refCount; ++i) {
		beginRefSlots[i] = refSlots[i];
	}
	beginRefSlots[refCount] = setupSlot;
	// The current reconstruction picture is bound for layout/resource tracking,
	// but is not an active reference until this decode completes. FFmpeg's
	// Vulkan decoder uses the same inactive (-1) begin-coding association and
	// supplies the real slot only through decodeInfo.pSetupReferenceSlot.
	beginRefSlots[refCount].slotIndex = -1;

	VkVideoBeginCodingInfoKHR beginCoding = {};
	beginCoding.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
	beginCoding.videoSession = Session_.Handle();
	beginCoding.videoSessionParameters = SessionParams_.Handle();
	beginCoding.referenceSlotCount = refCount + 1;
	beginCoding.pReferenceSlots = beginRefSlots.Data();
	vkCmdBeginVideoCodingKHR(cmd.Cb, &beginCoding);

	OaFnVideoDecoderRecord::ResetSessionIfNeeded(cmd, *this);

	// Match NVIDIA VkVideoDecoder ordering: BeginCoding → Reset → image barriers → decode.
	OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, InDpbSlot);
	for (OaU32 i = 0; i < refCount; ++i) {
		OaFnVideoDecoderRecord::EnsureDpbLayer(cmd, *this, refSlots[i].slotIndex);
	}
	OaFnVideoDecoderRecord::EnsureDistinctOutput(cmd, *this, InDpbSlot, hasDistinctOutput);

	VkVideoDecodeInfoKHR decodeInfo = {};
	decodeInfo.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
	decodeInfo.pNext = &av1Picture;
	// srcBufferRange must be a multiple of the profile's minBitstreamBufferSizeAlignment.
	const OaU64 av1SizeAlignment = bitstream.Buffer.GetSizeAlignment() == 0 ? 1 : bitstream.Buffer.GetSizeAlignment();
	decodeInfo.srcBuffer = bitstream.Buffer.GetBuffer();
	decodeInfo.srcBufferOffset = static_cast<VkDeviceSize>(InFrameOffset);
	decodeInfo.srcBufferRange = static_cast<VkDeviceSize>(OaAlignUp(static_cast<OaUsize>(InFrameSize), static_cast<OaUsize>(av1SizeAlignment)));
	decodeInfo.dstPictureResource = dstResource;
	decodeInfo.pSetupReferenceSlot = &setupSlot;
	decodeInfo.referenceSlotCount = refCount;
	decodeInfo.pReferenceSlots = refCount > 0 ? refSlots.Data() : nullptr;

	OaFnVideoDecoderRecord::EmitBitstreamDecodeBarrier(
		cmd,
		decodeInfo.srcBuffer,
		decodeInfo.srcBufferOffset,
		decodeInfo.srcBufferRange
	);

	vkCmdDecodeVideoKHR(cmd.Cb, &decodeInfo);

	VkVideoEndCodingInfoKHR endCoding = {};
	endCoding.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;
	vkCmdEndVideoCodingKHR(cmd.Cb, &endCoding);

	return OaFnVideoDecoderRecord::FinishAndSubmit(*this, cmd, {
		.DpbSlot = InDpbSlot,
		.HasDistinctOutput = hasDistinctOutput,
		.ErrorContext = "AV1 video decode",
	});
}
