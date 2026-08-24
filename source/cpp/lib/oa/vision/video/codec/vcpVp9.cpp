// OA Vision — VP9 codec parser Implementation

#include "vcpVp9.h"
#include "codecRegistry.h"
#include "bitstreamReader.h"
#include <oa/vision/videoDecoder.h>

namespace {

static constexpr oa::U32 kVp9FrameMarker = 2u;
static constexpr oa::U32 kVp9FrameSyncCode = 0x498342u;
static constexpr oa::U8 kVp9MaxProbability = 255u;
static constexpr oa::U32 kVp9MinTileWidthB64 = 4u;
static constexpr oa::U32 kVp9MaxTileWidthB64 = 64u;

static oa::U32 readLe32(const oa::U8* inData)
{
	return static_cast<oa::U32>(inData[0])
		| (static_cast<oa::U32>(inData[1]) << 8u)
		| (static_cast<oa::U32>(inData[2]) << 16u)
		| (static_cast<oa::U32>(inData[3]) << 24u);
}

static oa::U64 readLe64(const oa::U8* inData)
{
	return static_cast<oa::U64>(readLe32(inData))
		| (static_cast<oa::U64>(readLe32(inData + 4)) << 32u);
}

class Vp9BitReader {
public:
	explicit Vp9BitReader(const oa::U8* inData, oa::Usize inSize)
		: reader_(inData, inSize) {}

	oa::U32 readBits(oa::U32 inCount)
	{
		consumedBits_ += inCount;
		return reader_.readBits(inCount);
	}

	oa::U32 consumedBytes() const
	{
		return static_cast<oa::U32>((consumedBits_ + 7u) >> 3u);
	}

private:
	oa::BitstreamReader reader_;
	oa::U64 consumedBits_ = 0;
};

static oa::Status extractVp9FramePayload(
	const oa::Span<const oa::U8>& inBitstream,
	oa::Vp9IvfFrame& outFrame)
{
	const oa::U8* data = inBitstream.data();
	const oa::Usize size = inBitstream.size();
	if (size >= 32u &&
		data[0] == 'D' && data[1] == 'K' && data[2] == 'I' && data[3] == 'F' &&
		data[8] == 'V' && data[9] == 'P' && data[10] == '9' && data[11] == '0') {
		const oa::U32 headerSize = static_cast<oa::U32>(data[6]) | (static_cast<oa::U32>(data[7]) << 8u);
		if (headerSize < 32u || static_cast<oa::Usize>(headerSize) + 12u > size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 IVF header size");
		}
		const oa::Usize frameHeaderOffset = headerSize;
		const oa::U32 frameSize = readLe32(data + frameHeaderOffset);
		if (frameSize == 0u || frameHeaderOffset + 12u + frameSize > size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 IVF frame size");
		}
		outFrame.offset = frameHeaderOffset + 12u;
		outFrame.size = frameSize;
		outFrame.timestamp = readLe64(data + frameHeaderOffset + 4u);
		return oa::Status::ok();
	}

	outFrame.offset = 0;
	outFrame.size = size;
	outFrame.timestamp = 0;
	return oa::Status::ok();
}

static void parseSuperFrameIndex(
	const oa::U8* inData,
	oa::U32 inSize,
	oa::U32 outFrameSizes[8],
	oa::U32& outFrameCount)
{
	outFrameCount = 0;
	if (inSize == 0) {
		return;
	}
	const oa::U8 finalByte = inData[inSize - 1u];
	if ((finalByte & 0xe0u) != 0xc0u) {
		return;
	}
	const oa::U32 frames = (finalByte & 0x7u) + 1u;
	const oa::U32 mag = ((finalByte >> 3u) & 0x3u) + 1u;
	const oa::U32 indexSize = 2u + mag * frames;
	if (inSize < indexSize || inData[inSize - indexSize] != finalByte) {
		return;
	}
	const oa::U8* cursor = inData + inSize - indexSize + 1u;
	for (oa::U32 i = 0; i < frames; ++i) {
		oa::U32 frameSize = 0;
		for (oa::U32 j = 0; j < mag; ++j) {
			frameSize |= static_cast<oa::U32>(*cursor++) << (j * 8u);
		}
		outFrameSizes[i] = frameSize;
	}
	outFrameCount = frames;
}

} // namespace

oa::Status oa::VcpVp9::parseSps(const oa::Span<const oa::U8>&)
{
	return oa::Status::error(
		oa::StatusCode::Unimplemented,
		"VP9 does not use separate SPS; use parseAccessUnit");
}

oa::Status oa::VcpVp9::parsePps(const oa::Span<const oa::U8>&)
{
	return oa::Status::error(
		oa::StatusCode::Unimplemented,
		"VP9 does not use separate PPS; use parseAccessUnit");
}

void oa::VcpVp9::clearParameterSets()
{
	loopFilterRefDeltas_[0] = 1;
	loopFilterRefDeltas_[1] = 0;
	loopFilterRefDeltas_[2] = -1;
	loopFilterRefDeltas_[3] = -1;
	oa::memzero(loopFilterModeDeltas_, sizeof(loopFilterModeDeltas_));
	lastFrameWidth_ = 0;
	lastFrameHeight_ = 0;
	lastShowFrame_ = false;
	oa::memzero(bufferWidth_, sizeof(bufferWidth_));
	oa::memzero(bufferHeight_, sizeof(bufferHeight_));
	colorConfig_ = {};
	hasColorConfig_ = false;
}

static oa::I32 readDeltaQ(Vp9BitReader& inReader)
{
	if (inReader.readBits(1u) == 0u) {
		return 0;
	}
	oa::I32 delta = static_cast<oa::I32>(inReader.readBits(4u));
	if (inReader.readBits(1u) != 0u) {
		delta = -delta;
	}
	return delta;
}

oa::Status oa::VcpVp9::parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, oa::Vp9PictureDesc& outDesc)
{
	outDesc = {};

	oa::Vp9IvfFrame frame = {};
	OA_RETURN_IF_ERROR(extractVp9FramePayload(inBitstream, frame));
	outDesc.frame = frame;

	const oa::U8* frameData = inBitstream.data() + frame.offset;
	oa::U32 frameSize = static_cast<oa::U32>(frame.size);
	if (frameSize == 0u) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "VP9 frame is empty");
	}

	oa::U32 superframeSizes[8] = {};
	oa::U32 superframeCount = 0;
	parseSuperFrameIndex(frameData, frameSize, superframeSizes, superframeCount);

	const oa::U8* dataStart = frameData;
	oa::U32 dataSize = frameSize;
	if (superframeCount > 0u) {
		dataSize = superframeSizes[0];
		if (dataSize == 0u || dataSize > frameSize) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 superframe size");
		}
	}

	if (dataSize > 0u && (dataStart[0] & 0xe0u) == 0xc0u) {
		const oa::U8 marker = dataStart[0];
		const oa::U32 frames = (marker & 0x7u) + 1u;
		const oa::U32 mag = ((marker >> 3u) & 0x3u) + 1u;
		const oa::U32 indexSize = 2u + mag * frames;
		if (dataSize >= indexSize && dataStart[indexSize - 1u] == marker) {
			dataStart += indexSize;
			dataSize -= indexSize;
		}
	}

	if (dataSize < 2u) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "VP9 frame too small");
	}

	Vp9BitReader reader(dataStart, dataSize);
	StdVideoDecodeVP9PictureInfo& stdPic = outDesc.stdPictureInfo;
	StdVideoVP9ColorConfig& color = outDesc.colorConfig;
	StdVideoVP9LoopFilter& loopFilter = outDesc.loopFilter;
	StdVideoVP9Segmentation& segment = outDesc.segmentation;
	// VP9 signals color configuration only on key/intra-only frames. Every
	// StdVideoDecodeVP9PictureInfo still points at a complete color structure,
	// so inter frames inherit the most recently signalled configuration.
	if (hasColorConfig_) {
		color = colorConfig_;
	}

	if (reader.readBits(2u) != kVp9FrameMarker) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 frame marker");
	}

	oa::U32 profile = reader.readBits(1u);
	profile |= reader.readBits(1u) << 1u;
	stdPic.profile = static_cast<StdVideoVP9Profile>(profile);
	if (stdPic.profile == STD_VIDEO_VP9_PROFILE_3 && reader.readBits(1u) != 0u) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 profile 3 syntax");
	}

	outDesc.showExistingFrame = reader.readBits(1u) != 0u;
	if (outDesc.showExistingFrame) {
		outDesc.frameToShowMapIdx = static_cast<oa::U8>(reader.readBits(3u));
		outDesc.uncompressedHeaderOffset = reader.consumedBytes();
		outDesc.compressedHeaderSize = 0;
		stdPic.refresh_frame_flags = 0;
		loopFilter.loop_filter_level = 0;
		return oa::Status::ok();
	}

	stdPic.frame_type = static_cast<StdVideoVP9FrameType>(reader.readBits(1u));
	stdPic.flags.show_frame = reader.readBits(1u);
	stdPic.flags.error_resilient_mode = reader.readBits(1u);

	auto parseColorConfig = [&]() -> oa::Status {
		if (stdPic.profile >= STD_VIDEO_VP9_PROFILE_2) {
			color.BitDepth = reader.readBits(1u) != 0u ? 12u : 10u;
		} else {
			color.BitDepth = 8u;
		}
		color.color_space = static_cast<StdVideoVP9ColorSpace>(reader.readBits(3u));
		if (color.color_space != STD_VIDEO_VP9_COLOR_SPACE_RGB) {
			color.flags.color_range = reader.readBits(1u);
			if (stdPic.profile == STD_VIDEO_VP9_PROFILE_1 || stdPic.profile == STD_VIDEO_VP9_PROFILE_3) {
				color.subsampling_x = static_cast<oa::U8>(reader.readBits(1u));
				color.subsampling_y = static_cast<oa::U8>(reader.readBits(1u));
				if (reader.readBits(1u) != 0u) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 color config");
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
				if (reader.readBits(1u) != 0u) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 RGB color config");
				}
			}
		}
		return oa::Status::ok();
	};

	auto parseFrameAndRenderSize = [&]() {
		outDesc.frameWidth = reader.readBits(16u) + 1u;
		outDesc.frameHeight = reader.readBits(16u) + 1u;
		if (reader.readBits(1u) != 0u) {
			outDesc.renderWidth = reader.readBits(16u) + 1u;
			outDesc.renderHeight = reader.readBits(16u) + 1u;
		} else {
			outDesc.renderWidth = outDesc.frameWidth;
			outDesc.renderHeight = outDesc.frameHeight;
		}
	};

	auto computeImageSize = [&]() {
		if (static_cast<oa::I32>(outDesc.frameHeight) != lastFrameHeight_ ||
			static_cast<oa::I32>(outDesc.frameWidth) != lastFrameWidth_) {
			stdPic.flags.UsePrevFrameMvs = false;
		} else {
			const bool intraOnly = stdPic.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY || stdPic.flags.intra_only;
			stdPic.flags.UsePrevFrameMvs = lastShowFrame_ &&
				stdPic.flags.error_resilient_mode == 0u &&
				!intraOnly;
		}
		lastFrameWidth_ = static_cast<oa::I32>(outDesc.frameWidth);
		lastFrameHeight_ = static_cast<oa::I32>(outDesc.frameHeight);
		lastShowFrame_ = stdPic.flags.show_frame != 0u;
	};

	if (stdPic.frame_type == STD_VIDEO_VP9_FRAME_TYPE_KEY) {
		if (reader.readBits(24u) != kVp9FrameSyncCode) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 frame sync code");
		}
		OA_RETURN_IF_ERROR(parseColorConfig());
		parseFrameAndRenderSize();
		stdPic.refresh_frame_flags = static_cast<oa::U8>((1u << STD_VIDEO_VP9_NUM_REF_FRAMES) - 1u);
		outDesc.frameIsIntra = true;
		for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
			outDesc.refFrameIdx[i] = 0;
		}
	} else {
		stdPic.flags.intra_only = stdPic.flags.show_frame ? 0u : reader.readBits(1u);
		outDesc.frameIsIntra = stdPic.flags.intra_only != 0u;
		stdPic.reset_frame_context = stdPic.flags.error_resilient_mode != 0u ? 0u : static_cast<oa::U8>(reader.readBits(2u));

		if (stdPic.flags.intra_only != 0u) {
			if (reader.readBits(24u) != kVp9FrameSyncCode) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid VP9 intra-only sync code");
			}
			if (stdPic.profile > STD_VIDEO_VP9_PROFILE_0) {
				OA_RETURN_IF_ERROR(parseColorConfig());
			} else {
				color.color_space = STD_VIDEO_VP9_COLOR_SPACE_BT_601;
				color.subsampling_x = 1u;
				color.subsampling_y = 1u;
				color.BitDepth = 8u;
			}
			stdPic.refresh_frame_flags = static_cast<oa::U8>(reader.readBits(STD_VIDEO_VP9_NUM_REF_FRAMES));
			parseFrameAndRenderSize();
		} else {
			stdPic.refresh_frame_flags = static_cast<oa::U8>(reader.readBits(STD_VIDEO_VP9_NUM_REF_FRAMES));
			stdPic.ref_frame_sign_bias_mask = 0u;
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
				outDesc.refFrameIdx[i] = static_cast<oa::U8>(reader.readBits(3u));
				stdPic.ref_frame_sign_bias_mask |= static_cast<oa::U8>(reader.readBits(1u) << (STD_VIDEO_VP9_REFERENCE_NAME_LAST_FRAME + i));
			}

			bool foundRef = false;
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; ++i) {
				if (reader.readBits(1u) != 0u) {
					foundRef = true;
					const oa::U8 refIdx = outDesc.refFrameIdx[i];
					outDesc.frameWidth = bufferWidth_[refIdx];
					outDesc.frameHeight = bufferHeight_[refIdx];
					if (reader.readBits(1u) != 0u) {
						outDesc.renderWidth = reader.readBits(16u) + 1u;
						outDesc.renderHeight = reader.readBits(16u) + 1u;
					} else {
						outDesc.renderWidth = outDesc.frameWidth;
						outDesc.renderHeight = outDesc.frameHeight;
					}
					break;
				}
			}
			if (!foundRef) {
				parseFrameAndRenderSize();
			}

			stdPic.flags.allow_high_precision_mv = reader.readBits(1u);
			if (reader.readBits(1u) != 0u) {
				stdPic.interpolation_filter = STD_VIDEO_VP9_INTERPOLATION_FILTER_SWITCHABLE;
			} else {
				static constexpr StdVideoVP9InterpolationFilter kFilters[] = {
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_EIGHTTAP_SHARP,
					STD_VIDEO_VP9_INTERPOLATION_FILTER_BILINEAR,
				};
				stdPic.interpolation_filter = kFilters[reader.readBits(2u)];
			}
		}
	}

	computeImageSize();

	if (stdPic.flags.error_resilient_mode == 0u) {
		stdPic.flags.refresh_frame_context = reader.readBits(1u);
		stdPic.flags.frame_parallel_decoding_mode = reader.readBits(1u);
	} else {
		stdPic.flags.refresh_frame_context = 0u;
		stdPic.flags.frame_parallel_decoding_mode = 1u;
	}

	stdPic.frame_context_idx = static_cast<oa::U8>(reader.readBits(2u));
	if (outDesc.frameIsIntra || stdPic.flags.error_resilient_mode != 0u) {
		oa::memzero(segment.FeatureEnabled, sizeof(segment.FeatureEnabled));
		oa::memzero(segment.FeatureData, sizeof(segment.FeatureData));
		stdPic.frame_context_idx = 0;
	}

	if (outDesc.frameIsIntra || stdPic.flags.error_resilient_mode != 0u) {
		loopFilterRefDeltas_[0] = 1;
		loopFilterRefDeltas_[1] = 0;
		loopFilterRefDeltas_[2] = -1;
		loopFilterRefDeltas_[3] = -1;
		oa::memzero(loopFilterModeDeltas_, sizeof(loopFilterModeDeltas_));
	}

	loopFilter.loop_filter_level = static_cast<oa::U8>(reader.readBits(6u));
	loopFilter.loop_filter_sharpness = static_cast<oa::U8>(reader.readBits(3u));
	loopFilter.flags.loop_filter_delta_enabled = reader.readBits(1u);
	if (loopFilter.flags.loop_filter_delta_enabled != 0u) {
		loopFilter.flags.loop_filter_delta_update = reader.readBits(1u);
		if (loopFilter.flags.loop_filter_delta_update != 0u) {
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_MAX_REF_FRAMES; ++i) {
				const oa::U8 update = static_cast<oa::U8>(reader.readBits(1u));
				loopFilter.update_ref_delta |= static_cast<oa::U8>(update << i);
				if (update != 0u) {
					loopFilterRefDeltas_[i] = static_cast<oa::I8>(reader.readBits(6u));
					if (reader.readBits(1u) != 0u) {
						loopFilterRefDeltas_[i] = static_cast<oa::I8>(-loopFilterRefDeltas_[i]);
					}
				}
			}
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_LOOP_FILTER_ADJUSTMENTS; ++i) {
				const oa::U8 update = static_cast<oa::U8>(reader.readBits(1u));
				loopFilter.update_mode_delta |= static_cast<oa::U8>(update << i);
				if (update != 0u) {
					loopFilterModeDeltas_[i] = static_cast<oa::I8>(reader.readBits(6u));
					if (reader.readBits(1u) != 0u) {
						loopFilterModeDeltas_[i] = static_cast<oa::I8>(-loopFilterModeDeltas_[i]);
					}
				}
			}
		}
	}
	oa::memcpy(loopFilter.loop_filter_ref_deltas, loopFilterRefDeltas_, sizeof(loopFilterRefDeltas_));
	oa::memcpy(loopFilter.loop_filter_mode_deltas, loopFilterModeDeltas_, sizeof(loopFilterModeDeltas_));

	stdPic.base_q_idx = static_cast<oa::U8>(reader.readBits(8u));
	stdPic.delta_q_y_dc = static_cast<oa::I8>(readDeltaQ(reader));
	stdPic.delta_q_uv_dc = static_cast<oa::I8>(readDeltaQ(reader));
	stdPic.delta_q_uv_ac = static_cast<oa::I8>(readDeltaQ(reader));

	segment.flags.segmentation_update_map = 0u;
	segment.flags.segmentation_temporal_update = 0u;
	stdPic.flags.segmentation_enabled = reader.readBits(1u);
	if (stdPic.flags.segmentation_enabled != 0u) {
		segment.flags.segmentation_update_map = reader.readBits(1u);
		if (segment.flags.segmentation_update_map != 0u) {
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTATION_TREE_PROBS; ++i) {
				segment.segmentation_tree_probs[i] = reader.readBits(1u) != 0u
					? static_cast<oa::U8>(reader.readBits(8u))
					: kVp9MaxProbability;
			}
			segment.flags.segmentation_temporal_update = reader.readBits(1u);
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTATION_PRED_PROB; ++i) {
				if (segment.flags.segmentation_temporal_update != 0u) {
					segment.segmentation_pred_prob[i] = reader.readBits(1u) != 0u
						? static_cast<oa::U8>(reader.readBits(8u))
						: kVp9MaxProbability;
				} else {
					segment.segmentation_pred_prob[i] = kVp9MaxProbability;
				}
			}
		}

		segment.flags.segmentation_update_data = reader.readBits(1u);
		if (segment.flags.segmentation_update_data != 0u) {
			segment.flags.segmentation_abs_or_delta_update = reader.readBits(1u);
			oa::memzero(segment.FeatureEnabled, sizeof(segment.FeatureEnabled));
			oa::memzero(segment.FeatureData, sizeof(segment.FeatureData));
			static constexpr oa::U8 kFeatureBits[STD_VIDEO_VP9_SEG_LVL_MAX] = {8, 6, 2, 0};
			static constexpr oa::U8 kFeatureSigned[STD_VIDEO_VP9_SEG_LVL_MAX] = {1, 1, 0, 0};
			for (oa::U32 i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTS; ++i) {
				for (oa::U32 j = 0; j < STD_VIDEO_VP9_SEG_LVL_MAX; ++j) {
					const oa::U8 enabled = static_cast<oa::U8>(reader.readBits(1u));
					segment.FeatureEnabled[i] |= static_cast<oa::U8>(enabled << j);
					if (enabled != 0u) {
						segment.FeatureData[i][j] = static_cast<oa::I8>(reader.readBits(kFeatureBits[j]));
						if (kFeatureSigned[j] != 0u && reader.readBits(1u) != 0u) {
							segment.FeatureData[i][j] = static_cast<oa::I8>(-segment.FeatureData[i][j]);
						}
					}
				}
			}
		}
	}

	const oa::U32 miCols = (outDesc.frameWidth + 7u) >> 3u;
	const oa::U32 sb64Cols = (miCols + 7u) >> 3u;
	oa::U32 minLog2TileCols = 0u;
	while ((kVp9MaxTileWidthB64 << minLog2TileCols) < sb64Cols) {
		++minLog2TileCols;
	}
	oa::U32 maxLog2TileCols = 1u;
	while ((sb64Cols >> maxLog2TileCols) >= kVp9MinTileWidthB64) {
		++maxLog2TileCols;
	}
	maxLog2TileCols -= 1u;

	stdPic.tile_cols_log2 = static_cast<oa::U8>(minLog2TileCols);
	while (stdPic.tile_cols_log2 < maxLog2TileCols) {
		if (reader.readBits(1u) == 0u) {
			break;
		}
		++stdPic.tile_cols_log2;
	}

	stdPic.tile_rows_log2 = static_cast<oa::U8>(reader.readBits(1u));
	if (stdPic.tile_rows_log2 != 0u) {
		stdPic.tile_rows_log2 = static_cast<oa::U8>(stdPic.tile_rows_log2 + reader.readBits(1u));
	}

	outDesc.compressedHeaderSize = reader.readBits(16u);
	outDesc.uncompressedHeaderOffset = 0u;
	outDesc.compressedHeaderOffset = reader.consumedBytes();
	outDesc.tilesOffset = outDesc.compressedHeaderOffset + outDesc.compressedHeaderSize;
	outDesc.numTiles = (1u << stdPic.tile_rows_log2) * (1u << stdPic.tile_cols_log2);
	outDesc.chromaFormat = (color.subsampling_x == 1u && color.subsampling_y == 1u) ? 1u : 0u;

	outDesc.hasPicture = true;
	if (outDesc.frameIsIntra) {
		colorConfig_ = color;
		hasColorConfig_ = true;
	}

	for (oa::U32 mask = stdPic.refresh_frame_flags, refIndex = 0u; mask != 0u; mask >>= 1u, ++refIndex) {
		if ((mask & 1u) != 0u) {
			bufferWidth_[refIndex] = outDesc.frameWidth;
			bufferHeight_[refIndex] = outDesc.frameHeight;
		}
	}

	return oa::Status::ok();
}

namespace {
struct Vp9CodecRegistrar {
	Vp9CodecRegistrar() {
		auto parser = oa::makeUnique<oa::VcpVp9>();
		oa::VideoCodecRegistry::getInstance().registerParser(
			oa::VideoCodec::VP9,
			oa::move(parser));
	}
};
static Vp9CodecRegistrar g_Vp9Registrar __attribute__((used));
} // namespace
