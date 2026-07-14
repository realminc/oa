// OA Vision — VP9 Codec Parser Implementation

#include "VcpVp9.h"
#include "CodecRegistry.h"
#include "BitstreamReader.h"
#include <Oa/Vision/VideoDecoder.h>

namespace {

static constexpr OaU32 kVp9FrameMarker = 2u;
static constexpr OaU32 kVp9FrameSyncCode = 0x498342u;
static constexpr OaU8 kVp9MaxProbability = 255u;
static constexpr OaU32 kVp9MinTileWidthB64 = 4u;
static constexpr OaU32 kVp9MaxTileWidthB64 = 64u;

static OaU32 ReadLe32(const OaU8* InData)
{
	return static_cast<OaU32>(InData[0])
		| (static_cast<OaU32>(InData[1]) << 8u)
		| (static_cast<OaU32>(InData[2]) << 16u)
		| (static_cast<OaU32>(InData[3]) << 24u);
}

static OaU64 ReadLe64(const OaU8* InData)
{
	return static_cast<OaU64>(ReadLe32(InData))
		| (static_cast<OaU64>(ReadLe32(InData + 4)) << 32u);
}

class OaVp9BitReader {
public:
	explicit OaVp9BitReader(const OaU8* InData, OaUsize InSize)
		: Reader_(InData, InSize) {}

	OaU32 ReadBits(OaU32 InCount)
	{
		ConsumedBits_ += InCount;
		return Reader_.ReadBits(InCount);
	}

	OaU32 ConsumedBytes() const
	{
		return static_cast<OaU32>((ConsumedBits_ + 7u) >> 3u);
	}

private:
	OaBitstreamReader Reader_;
	OaU64 ConsumedBits_ = 0;
};

static OaStatus ExtractVp9FramePayload(
	const OaSpan<const OaU8>& InBitstream,
	OaVp9IvfFrame& OutFrame)
{
	const OaU8* data = InBitstream.Data();
	const OaUsize size = InBitstream.Size();
	if (size >= 32u &&
		data[0] == 'D' && data[1] == 'K' && data[2] == 'I' && data[3] == 'F' &&
		data[8] == 'V' && data[9] == 'P' && data[10] == '9' && data[11] == '0') {
		const OaU32 headerSize = static_cast<OaU32>(data[6]) | (static_cast<OaU32>(data[7]) << 8u);
		if (headerSize < 32u || static_cast<OaUsize>(headerSize) + 12u > size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 IVF header size");
		}
		const OaUsize frameHeaderOffset = headerSize;
		const OaU32 frameSize = ReadLe32(data + frameHeaderOffset);
		if (frameSize == 0u || frameHeaderOffset + 12u + frameSize > size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 IVF frame size");
		}
		OutFrame.Offset = frameHeaderOffset + 12u;
		OutFrame.Size = frameSize;
		OutFrame.Timestamp = ReadLe64(data + frameHeaderOffset + 4u);
		return OaStatus::Ok();
	}

	OutFrame.Offset = 0;
	OutFrame.Size = size;
	OutFrame.Timestamp = 0;
	return OaStatus::Ok();
}

static void ParseSuperFrameIndex(
	const OaU8* InData,
	OaU32 InSize,
	OaU32 OutFrameSizes[8],
	OaU32& OutFrameCount)
{
	OutFrameCount = 0;
	if (InSize == 0) {
		return;
	}
	const OaU8 finalByte = InData[InSize - 1u];
	if ((finalByte & 0xe0u) != 0xc0u) {
		return;
	}
	const OaU32 frames = (finalByte & 0x7u) + 1u;
	const OaU32 mag = ((finalByte >> 3u) & 0x3u) + 1u;
	const OaU32 indexSize = 2u + mag * frames;
	if (InSize < indexSize || InData[InSize - indexSize] != finalByte) {
		return;
	}
	const OaU8* cursor = InData + InSize - indexSize + 1u;
	for (OaU32 i = 0; i < frames; ++i) {
		OaU32 frameSize = 0;
		for (OaU32 j = 0; j < mag; ++j) {
			frameSize |= static_cast<OaU32>(*cursor++) << (j * 8u);
		}
		OutFrameSizes[i] = frameSize;
	}
	OutFrameCount = frames;
}

} // namespace

OaStatus OaVcpVp9::ParseSps(const OaSpan<const OaU8>&)
{
	return OaStatus::Error(
		OaStatusCode::Unimplemented,
		"VP9 does not use separate SPS; use ParseAccessUnit");
}

OaStatus OaVcpVp9::ParsePps(const OaSpan<const OaU8>&)
{
	return OaStatus::Error(
		OaStatusCode::Unimplemented,
		"VP9 does not use separate PPS; use ParseAccessUnit");
}

void OaVcpVp9::ClearParameterSets()
{
	LoopFilterRefDeltas_[0] = 1;
	LoopFilterRefDeltas_[1] = 0;
	LoopFilterRefDeltas_[2] = -1;
	LoopFilterRefDeltas_[3] = -1;
	OaMemzero(LoopFilterModeDeltas_, sizeof(LoopFilterModeDeltas_));
	LastFrameWidth_ = 0;
	LastFrameHeight_ = 0;
	LastShowFrame_ = false;
	OaMemzero(BufferWidth_, sizeof(BufferWidth_));
	OaMemzero(BufferHeight_, sizeof(BufferHeight_));
	ColorConfig_ = {};
	HasColorConfig_ = false;
}

static OaI32 ReadDeltaQ(OaVp9BitReader& InReader)
{
	if (InReader.ReadBits(1u) == 0u) {
		return 0;
	}
	OaI32 delta = static_cast<OaI32>(InReader.ReadBits(4u));
	if (InReader.ReadBits(1u) != 0u) {
		delta = -delta;
	}
	return delta;
}

OaStatus OaVcpVp9::ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaVp9PictureDesc& OutDesc)
{
	OutDesc = {};

	OaVp9IvfFrame frame = {};
	OA_RETURN_IF_ERROR(ExtractVp9FramePayload(InBitstream, frame));
	OutDesc.Frame = frame;

	const OaU8* frameData = InBitstream.Data() + frame.Offset;
	OaU32 frameSize = static_cast<OaU32>(frame.Size);
	if (frameSize == 0u) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "VP9 frame is empty");
	}

	OaU32 superframeSizes[8] = {};
	OaU32 superframeCount = 0;
	ParseSuperFrameIndex(frameData, frameSize, superframeSizes, superframeCount);

	const OaU8* dataStart = frameData;
	OaU32 dataSize = frameSize;
	if (superframeCount > 0u) {
		dataSize = superframeSizes[0];
		if (dataSize == 0u || dataSize > frameSize) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 superframe size");
		}
	}

	if (dataSize > 0u && (dataStart[0] & 0xe0u) == 0xc0u) {
		const OaU8 marker = dataStart[0];
		const OaU32 frames = (marker & 0x7u) + 1u;
		const OaU32 mag = ((marker >> 3u) & 0x3u) + 1u;
		const OaU32 indexSize = 2u + mag * frames;
		if (dataSize >= indexSize && dataStart[indexSize - 1u] == marker) {
			dataStart += indexSize;
			dataSize -= indexSize;
		}
	}

	if (dataSize < 2u) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "VP9 frame too small");
	}

	OaVp9BitReader reader(dataStart, dataSize);
	StdVideoDecodeVP9PictureInfo& stdPic = OutDesc.StdPictureInfo;
	StdVideoVP9ColorConfig& color = OutDesc.ColorConfig;
	StdVideoVP9LoopFilter& loopFilter = OutDesc.LoopFilter;
	StdVideoVP9Segmentation& segment = OutDesc.Segmentation;
	// VP9 signals color configuration only on key/intra-only frames. Every
	// StdVideoDecodeVP9PictureInfo still points at a complete color structure,
	// so inter frames inherit the most recently signalled configuration.
	if (HasColorConfig_) {
		color = ColorConfig_;
	}

	if (reader.ReadBits(2u) != kVp9FrameMarker) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 frame marker");
	}

	OaU32 profile = reader.ReadBits(1u);
	profile |= reader.ReadBits(1u) << 1u;
	stdPic.profile = static_cast<StdVideoVP9Profile>(profile);
	if (stdPic.profile == STD_VIDEO_VP9_PROFILE_3 && reader.ReadBits(1u) != 0u) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 profile 3 syntax");
	}

	OutDesc.ShowExistingFrame = reader.ReadBits(1u) != 0u;
	if (OutDesc.ShowExistingFrame) {
		OutDesc.FrameToShowMapIdx = static_cast<OaU8>(reader.ReadBits(3u));
		OutDesc.UncompressedHeaderOffset = reader.ConsumedBytes();
		OutDesc.CompressedHeaderSize = 0;
		stdPic.refresh_frame_flags = 0;
		loopFilter.loop_filter_level = 0;
		return OaStatus::Ok();
	}

	stdPic.frame_type = static_cast<StdVideoVP9FrameType>(reader.ReadBits(1u));
	stdPic.flags.show_frame = reader.ReadBits(1u);
	stdPic.flags.error_resilient_mode = reader.ReadBits(1u);

	auto parseColorConfig = [&]() -> OaStatus {
		if (stdPic.profile >= STD_VIDEO_VP9_PROFILE_2) {
			color.BitDepth = reader.ReadBits(1u) != 0u ? 12u : 10u;
		} else {
			color.BitDepth = 8u;
		}
		color.color_space = static_cast<StdVideoVP9ColorSpace>(reader.ReadBits(3u));
		if (color.color_space != STD_VIDEO_VP9_COLOR_SPACE_RGB) {
			color.flags.color_range = reader.ReadBits(1u);
			if (stdPic.profile == STD_VIDEO_VP9_PROFILE_1 || stdPic.profile == STD_VIDEO_VP9_PROFILE_3) {
				color.subsampling_x = static_cast<OaU8>(reader.ReadBits(1u));
				color.subsampling_y = static_cast<OaU8>(reader.ReadBits(1u));
				if (reader.ReadBits(1u) != 0u) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 color config");
				}
			} else {
				color.subsampling_x = 1u;
				color.subsampling_y = 1u;
			}
		} else {
			color.flags.color_range = 1u;
			if (stdPic.profile == STD_VIDEO_VP9_PROFILE_1 || stdPic.profile == STD_VIDEO_VP9_PROFILE_3) {
				color.subsampling_x = 0u;
				color.subsampling_y = 0u;
				if (reader.ReadBits(1u) != 0u) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 RGB color config");
				}
			}
		}
		return OaStatus::Ok();
	};

	auto parseFrameAndRenderSize = [&]() {
		OutDesc.FrameWidth = reader.ReadBits(16u) + 1u;
		OutDesc.FrameHeight = reader.ReadBits(16u) + 1u;
		if (reader.ReadBits(1u) != 0u) {
			OutDesc.RenderWidth = reader.ReadBits(16u) + 1u;
			OutDesc.RenderHeight = reader.ReadBits(16u) + 1u;
		} else {
			OutDesc.RenderWidth = OutDesc.FrameWidth;
			OutDesc.RenderHeight = OutDesc.FrameHeight;
		}
	};

	auto computeImageSize = [&]() {
		if (static_cast<OaI32>(OutDesc.FrameHeight) != LastFrameHeight_ ||
			static_cast<OaI32>(OutDesc.FrameWidth) != LastFrameWidth_) {
			stdPic.flags.UsePrevFrameMvs = false;
		} else {
			const bool intraOnly = stdPic.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY || stdPic.flags.intra_only;
			stdPic.flags.UsePrevFrameMvs = LastShowFrame_ &&
				stdPic.flags.error_resilient_mode == 0u &&
				!intraOnly;
		}
		LastFrameWidth_ = static_cast<OaI32>(OutDesc.FrameWidth);
		LastFrameHeight_ = static_cast<OaI32>(OutDesc.FrameHeight);
		LastShowFrame_ = stdPic.flags.show_frame != 0u;
	};

	if (stdPic.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY) {
		if (reader.ReadBits(24u) != kVp9FrameSyncCode) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 frame sync code");
		}
		OA_RETURN_IF_ERROR(parseColorConfig());
		parseFrameAndRenderSize();
		stdPic.refresh_frame_flags = static_cast<OaU8>((1u << STD_VIDEO_VP9_NUM_REF_FRAMES) - 1u);
		OutDesc.FrameIsIntra = true;
		for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
			OutDesc.RefFrameIdx[i] = 0;
		}
	} else {
		stdPic.flags.intra_only = stdPic.flags.show_frame ? 0u : reader.ReadBits(1u);
		OutDesc.FrameIsIntra = stdPic.flags.intra_only != 0u;
		stdPic.reset_frame_context = stdPic.flags.error_resilient_mode != 0u ? 0u : static_cast<OaU8>(reader.ReadBits(2u));

		if (stdPic.flags.intra_only != 0u) {
			if (reader.ReadBits(24u) != kVp9FrameSyncCode) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid VP9 intra-only sync code");
			}
			if (stdPic.profile > STD_VIDEO_VP9_PROFILE_0) {
				OA_RETURN_IF_ERROR(parseColorConfig());
			} else {
				color.color_space = STD_VIDEO_VP9_COLOR_SPACE_BT_601;
				color.subsampling_x = 1u;
				color.subsampling_y = 1u;
				color.BitDepth = 8u;
			}
			stdPic.refresh_frame_flags = static_cast<OaU8>(reader.ReadBits(STD_VIDEO_VP9_NUM_REF_FRAMES));
			parseFrameAndRenderSize();
		} else {
			stdPic.refresh_frame_flags = static_cast<OaU8>(reader.ReadBits(STD_VIDEO_VP9_NUM_REF_FRAMES));
			stdPic.ref_frame_sign_bias_mask = 0u;
			for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
				OutDesc.RefFrameIdx[i] = static_cast<OaU8>(reader.ReadBits(3u));
				stdPic.ref_frame_sign_bias_mask |= static_cast<OaU8>(reader.ReadBits(1u) << (STD_VIDEO_VP9_REFERENCE_NAME_LAST_FRAME + i));
			}

			bool foundRef = false;
			for (OaU32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
				if (reader.ReadBits(1u) != 0u) {
					foundRef = true;
					const OaU8 refIdx = OutDesc.RefFrameIdx[i];
					OutDesc.FrameWidth = BufferWidth_[refIdx];
					OutDesc.FrameHeight = BufferHeight_[refIdx];
					if (reader.ReadBits(1u) != 0u) {
						OutDesc.RenderWidth = reader.ReadBits(16u) + 1u;
						OutDesc.RenderHeight = reader.ReadBits(16u) + 1u;
					} else {
						OutDesc.RenderWidth = OutDesc.FrameWidth;
						OutDesc.RenderHeight = OutDesc.FrameHeight;
					}
					break;
				}
			}
			if (!foundRef) {
				parseFrameAndRenderSize();
			}

			stdPic.flags.allow_high_precision_mv = reader.ReadBits(1u);
			if (reader.ReadBits(1u) != 0u) {
				stdPic.interpolation_filter = STD_VIDEO_VP9_INTERPOLATION_FILTER_SWITCHABLE;
			} else {
				static constexpr StdVideoVP9InterpolationFilter kFilters[] = {
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SHARP,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_BILINEAR,
				};
				stdPic.interpolation_filter = kFilters[reader.ReadBits(2u)];
			}
		}
	}

	computeImageSize();

	if (stdPic.flags.error_resilient_mode == 0u) {
		stdPic.flags.refresh_frame_context = reader.ReadBits(1u);
		stdPic.flags.frame_parallel_decoding_mode = reader.ReadBits(1u);
	} else {
		stdPic.flags.refresh_frame_context = 0u;
		stdPic.flags.frame_parallel_decoding_mode = 1u;
	}

	stdPic.frame_context_idx = static_cast<OaU8>(reader.ReadBits(2u));
	if (OutDesc.FrameIsIntra || stdPic.flags.error_resilient_mode != 0u) {
		OaMemzero(segment.FeatureEnabled, sizeof(segment.FeatureEnabled));
		OaMemzero(segment.FeatureData, sizeof(segment.FeatureData));
		stdPic.frame_context_idx = 0;
	}

	if (OutDesc.FrameIsIntra || stdPic.flags.error_resilient_mode != 0u) {
		LoopFilterRefDeltas_[0] = 1;
		LoopFilterRefDeltas_[1] = 0;
		LoopFilterRefDeltas_[2] = -1;
		LoopFilterRefDeltas_[3] = -1;
		OaMemzero(LoopFilterModeDeltas_, sizeof(LoopFilterModeDeltas_));
	}

	loopFilter.loop_filter_level = static_cast<OaU8>(reader.ReadBits(6u));
	loopFilter.loop_filter_sharpness = static_cast<OaU8>(reader.ReadBits(3u));
	loopFilter.flags.loop_filter_delta_enabled = reader.ReadBits(1u);
	if (loopFilter.flags.loop_filter_delta_enabled != 0u) {
		loopFilter.flags.loop_filter_delta_update = reader.ReadBits(1u);
		if (loopFilter.flags.loop_filter_delta_update != 0u) {
			for (OaU32 i = 0; i < STD_VIDEO_VP9_MAX_REF_FRAMES; ++i) {
				const OaU8 update = static_cast<OaU8>(reader.ReadBits(1u));
				loopFilter.update_ref_delta |= static_cast<OaU8>(update << i);
				if (update != 0u) {
					LoopFilterRefDeltas_[i] = static_cast<OaI8>(reader.ReadBits(6u));
					if (reader.ReadBits(1u) != 0u) {
						LoopFilterRefDeltas_[i] = static_cast<OaI8>(-LoopFilterRefDeltas_[i]);
					}
				}
			}
			for (OaU32 i = 0; i < STD_VIDEO_VP9_LOOP_FILTER_ADJUSTMENTS; ++i) {
				const OaU8 update = static_cast<OaU8>(reader.ReadBits(1u));
				loopFilter.update_mode_delta |= static_cast<OaU8>(update << i);
				if (update != 0u) {
					LoopFilterModeDeltas_[i] = static_cast<OaI8>(reader.ReadBits(6u));
					if (reader.ReadBits(1u) != 0u) {
						LoopFilterModeDeltas_[i] = static_cast<OaI8>(-LoopFilterModeDeltas_[i]);
					}
				}
			}
		}
	}
	OaMemcpy(loopFilter.loop_filter_ref_deltas, LoopFilterRefDeltas_, sizeof(LoopFilterRefDeltas_));
	OaMemcpy(loopFilter.loop_filter_mode_deltas, LoopFilterModeDeltas_, sizeof(LoopFilterModeDeltas_));

	stdPic.base_q_idx = static_cast<OaU8>(reader.ReadBits(8u));
	stdPic.delta_q_y_dc = static_cast<OaI8>(ReadDeltaQ(reader));
	stdPic.delta_q_uv_dc = static_cast<OaI8>(ReadDeltaQ(reader));
	stdPic.delta_q_uv_ac = static_cast<OaI8>(ReadDeltaQ(reader));

	segment.flags.segmentation_update_map = 0u;
	segment.flags.segmentation_temporal_update = 0u;
	stdPic.flags.segmentation_enabled = reader.ReadBits(1u);
	if (stdPic.flags.segmentation_enabled != 0u) {
		segment.flags.segmentation_update_map = reader.ReadBits(1u);
		if (segment.flags.segmentation_update_map != 0u) {
			for (OaU32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTATION_TREE_PROBS; ++i) {
				segment.segmentation_tree_probs[i] = reader.ReadBits(1u) != 0u
					? static_cast<OaU8>(reader.ReadBits(8u))
					: kVp9MaxProbability;
			}
			segment.flags.segmentation_temporal_update = reader.ReadBits(1u);
			for (OaU32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTATION_PRED_PROB; ++i) {
				if (segment.flags.segmentation_temporal_update != 0u) {
					segment.segmentation_pred_prob[i] = reader.ReadBits(1u) != 0u
						? static_cast<OaU8>(reader.ReadBits(8u))
						: kVp9MaxProbability;
				} else {
					segment.segmentation_pred_prob[i] = kVp9MaxProbability;
				}
			}
		}

		segment.flags.segmentation_update_data = reader.ReadBits(1u);
		if (segment.flags.segmentation_update_data != 0u) {
			segment.flags.segmentation_abs_or_delta_update = reader.ReadBits(1u);
			OaMemzero(segment.FeatureEnabled, sizeof(segment.FeatureEnabled));
			OaMemzero(segment.FeatureData, sizeof(segment.FeatureData));
			static constexpr OaU8 kFeatureBits[STD_VIDEO_VP9_SEG_LVL_MAX] = {8, 6, 2, 0};
			static constexpr OaU8 kFeatureSigned[STD_VIDEO_VP9_SEG_LVL_MAX] = {1, 1, 0, 0};
			for (OaU32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTS; ++i) {
				for (OaU32 j = 0; j < STD_VIDEO_VP9_SEG_LVL_MAX; ++j) {
					const OaU8 enabled = static_cast<OaU8>(reader.ReadBits(1u));
					segment.FeatureEnabled[i] |= static_cast<OaU8>(enabled << j);
					if (enabled != 0u) {
						segment.FeatureData[i][j] = static_cast<OaI8>(reader.ReadBits(kFeatureBits[j]));
						if (kFeatureSigned[j] != 0u && reader.ReadBits(1u) != 0u) {
							segment.FeatureData[i][j] = static_cast<OaI8>(-segment.FeatureData[i][j]);
						}
					}
				}
			}
		}
	}

	const OaU32 miCols = (OutDesc.FrameWidth + 7u) >> 3u;
	const OaU32 sb64Cols = (miCols + 7u) >> 3u;
	OaU32 minLog2TileCols = 0u;
	while ((kVp9MaxTileWidthB64 << minLog2TileCols) < sb64Cols) {
		++minLog2TileCols;
	}
	OaU32 maxLog2TileCols = 1u;
	while ((sb64Cols >> maxLog2TileCols) >= kVp9MinTileWidthB64) {
		++maxLog2TileCols;
	}
	maxLog2TileCols -= 1u;

	stdPic.tile_cols_log2 = static_cast<OaU8>(minLog2TileCols);
	while (stdPic.tile_cols_log2 < maxLog2TileCols) {
		if (reader.ReadBits(1u) == 0u) {
			break;
		}
		++stdPic.tile_cols_log2;
	}

	stdPic.tile_rows_log2 = static_cast<OaU8>(reader.ReadBits(1u));
	if (stdPic.tile_rows_log2 != 0u) {
		stdPic.tile_rows_log2 = static_cast<OaU8>(stdPic.tile_rows_log2 + reader.ReadBits(1u));
	}

	OutDesc.CompressedHeaderSize = reader.ReadBits(16u);
	OutDesc.UncompressedHeaderOffset = 0u;
	OutDesc.CompressedHeaderOffset = reader.ConsumedBytes();
	OutDesc.TilesOffset = OutDesc.CompressedHeaderOffset + OutDesc.CompressedHeaderSize;
	OutDesc.NumTiles = (1u << stdPic.tile_rows_log2) * (1u << stdPic.tile_cols_log2);
	OutDesc.ChromaFormat = (color.subsampling_x == 1u && color.subsampling_y == 1u) ? 1u : 0u;

	OutDesc.HasPicture = true;
	if (OutDesc.FrameIsIntra) {
		ColorConfig_ = color;
		HasColorConfig_ = true;
	}

	for (OaU32 mask = stdPic.refresh_frame_flags, refIndex = 0u; mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u) {
			BufferWidth_[refIndex] = OutDesc.FrameWidth;
			BufferHeight_[refIndex] = OutDesc.FrameHeight;
		}
	}

	return OaStatus::Ok();
}

namespace {
struct OaVp9CodecRegistrar {
	OaVp9CodecRegistrar() {
		auto parser = OaStdMakeUnique<OaVcpVp9>();
		OaVideoCodecRegistry::GetInstance().RegisterParser(
			OaVideoCodec::VP9,
			OaStdMove(parser));
	}
};
static OaVp9CodecRegistrar g_Vp9Registrar __attribute__((used));
} // namespace
