// OA Vision — AV1 codec parser Implementation
// Extracts and converts AV1 oBUs (open bitstream Units)

#include "vcpAv1.h"
#include "codecRegistry.h"
#include <oa/vision/videoDecoder.h>

class Av1BitReader {
public:
	Av1BitReader(const oa::U8* inData, oa::Usize inSize)
		: data_(inData), size_(inSize)
	{}

	bool readBits(oa::U32 inCount, oa::U32& outValue)
	{
		if (inCount > 32 || bitOffset_ + inCount > size_ * 8u) {
			return false;
		}
		oa::U32 value = 0;
		for (oa::U32 i = 0; i < inCount; ++i) {
			const oa::Usize bitIndex = bitOffset_++;
			const oa::U8 byte = data_[bitIndex >> 3u];
			value = (value << 1u) | ((byte >> (7u - (bitIndex & 7u))) & 1u);
		}
		outValue = value;
		return true;
	}

	bool readBit(bool& outValue)
	{
		oa::U32 value = 0;
		if (!readBits(1, value)) {
			return false;
		}
		outValue = value != 0;
		return true;
	}

	bool skipBits(oa::U32 inCount)
	{
		oa::U32 ignored = 0;
		while (inCount > 0) {
			const oa::U32 chunk = inCount > 32 ? 32 : inCount;
			if (!readBits(chunk, ignored)) {
				return false;
			}
			inCount -= chunk;
		}
		return true;
	}

	bool readUvlc(oa::U32& outValue)
	{
		oa::U32 leadingZeros = 0;
		bool bit = false;
		while (leadingZeros < 32u) {
			if (!readBit(bit)) return false;
			if (bit) break;
			++leadingZeros;
		}
		if (leadingZeros == 32u) {
			outValue = 0xffffffffu;
			return true;
		}
		oa::U32 suffix = 0;
		if (leadingZeros > 0u && !readBits(leadingZeros, suffix)) return false;
		outValue = ((1u << leadingZeros) - 1u) + suffix;
		return true;
	}

	void byteAlign() {
		bitOffset_ = oa::alignUp(bitOffset_, static_cast<oa::Usize>(8));
	}

	oa::Usize byteOffset() const {
		return (bitOffset_ + 7u) >> 3u;
	}

	oa::Usize debugBitPos() const { return bitOffset_; }

private:
	const oa::U8* data_ = nullptr;
	oa::Usize size_ = 0;
	oa::Usize bitOffset_ = 0;
};

static oa::U32 readLe32(const oa::U8* inData) {
	return static_cast<oa::U32>(inData[0]) |
		(static_cast<oa::U32>(inData[1]) << 8u) |
		(static_cast<oa::U32>(inData[2]) << 16u) |
		(static_cast<oa::U32>(inData[3]) << 24u);
}

static oa::U64 readLe64(const oa::U8* inData) {
	oa::U64 value = 0;
	for (oa::U32 i = 0; i < 8; ++i) {
		value |= static_cast<oa::U64>(inData[i]) << (i * 8u);
	}
	return value;
}

static bool readLeb128(const oa::U8* inData, oa::Usize inSize, oa::Usize& inOutOffset, oa::U64& outValue)
{
	outValue = 0;
	for (oa::U32 i = 0; i < 8 && inOutOffset < inSize; ++i) {
		const oa::U8 byte = inData[inOutOffset++];
		outValue |= static_cast<oa::U64>(byte & 0x7fu) << (i * 7u);
		if ((byte & 0x80u) == 0) {
			return true;
		}
	}
	return false;
}

static oa::U32 av1FloorLog2(oa::U32 inValue)
{
	oa::U32 result = 0;
	while (inValue > 1) {
		inValue >>= 1u;
		++result;
	}
	return result;
}

static oa::Status readAv1DeltaQ(Av1BitReader& inReader, oa::I32& outDelta)
{
	bool deltaCoded = false;
	if (!inReader.readBit(deltaCoded)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 quantization delta flag");
	}
	outDelta = 0;
	if (!deltaCoded) {
		return oa::Status::ok();
	}
	oa::U32 value = 0;
	if (!inReader.readBits(7, value)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 quantization delta");
	}
	outDelta = (value & 1u) != 0
		? -static_cast<oa::I32>((value + 1u) >> 1u)
		: static_cast<oa::I32>(value >> 1u);
	return oa::Status::ok();
}

static oa::Status readAv1SignedFeatureData(Av1BitReader& inReader, oa::U32 inBits, oa::I16& outValue)
{
	if (inBits == 0) {
		outValue = 0;
		return oa::Status::ok();
	}
	oa::U32 value = 0;
	bool sign = false;
	if (!inReader.readBits(inBits, value) || !inReader.readBit(sign)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 segmentation feature data");
	}
	outValue = static_cast<oa::I16>(sign ? -static_cast<oa::I32>(value) : static_cast<oa::I32>(value));
	return oa::Status::ok();
}

static oa::Status readAv1InverseSignedLiteral(Av1BitReader& inReader, oa::U32 inBits, oa::I8& outValue)
{
	oa::U32 value = 0;
	if (!inReader.readBits(inBits, value)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 signed literal");
	}
	const oa::I32 decoded = (value & 1u) != 0
		? -static_cast<oa::I32>((value >> 1u) + 1u)
		: static_cast<oa::I32>(value >> 1u);
	outValue = static_cast<oa::I8>(decoded);
	return oa::Status::ok();
}

static oa::Status readAv1RestorationType(Av1BitReader& inReader, StdVideoAV1FrameRestorationType& outType)
{
	bool useWiener = false;
	if (!inReader.readBit(useWiener)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 restoration type");
	}
	bool useSgrproj = false;
	if (!inReader.readBit(useSgrproj)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 restoration type");
	}
	if (useWiener && useSgrproj) {
		outType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE;
	} else if (useWiener) {
		outType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER;
	} else if (useSgrproj) {
		outType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ;
	} else {
		outType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE;
	}
	return oa::Status::ok();
}

static oa::Status parseAv1RestorationHeader(
	Av1BitReader& inReader,
	bool inUse128x128Superblock,
	oa::Av1FrameHeaderInfo& inOutInfo)
{
	for (oa::U32 plane = 0; plane < STD_VIDEO_AV1_MAX_NUM_PLANES; ++plane) {
		OA_RETURN_IF_ERROR(readAv1RestorationType(inReader, inOutInfo.restorationTypes[plane]));
	}

	inOutInfo.usesLr = false;
	inOutInfo.usesChromaLr = false;
	for (oa::U32 plane = 0; plane < STD_VIDEO_AV1_MAX_NUM_PLANES; ++plane) {
		if (inOutInfo.restorationTypes[plane] != STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE) {
			inOutInfo.usesLr = true;
			inOutInfo.usesChromaLr = inOutInfo.usesChromaLr || plane > 0;
		}
	}
	if (!inOutInfo.usesLr) {
		return oa::Status::ok();
	}

	bool flag = false;
	oa::U32 lrUnitShift = 0;
	if (inUse128x128Superblock) {
		if (!inReader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 restoration unit shift");
		}
		lrUnitShift = flag ? 2u : 1u;
	} else {
		if (!inReader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 restoration unit shift");
		}
		lrUnitShift = flag ? 1u : 0u;
		if (lrUnitShift > 0) {
			if (!inReader.readBit(flag)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 restoration unit extra shift");
			}
			lrUnitShift += flag ? 1u : 0u;
		}
	}
	// vulkan std-video / NVIDIA vk_video_samples store log2(restoration_unit_size)-5,
	// which for AV1 equals 1 + lr_unit_shift (not the pixel width).
	const oa::U16 lumaRestorationSize = static_cast<oa::U16>(1u + lrUnitShift);
	inOutInfo.restorationSizes[0] = lumaRestorationSize;

	oa::U32 lrUvShift = 0;
	if (inOutInfo.usesChromaLr) {
		if (!inReader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 chroma restoration unit shift");
		}
		lrUvShift = flag ? 1u : 0u;
	}
	inOutInfo.restorationSizes[1] = static_cast<oa::U16>(lumaRestorationSize - lrUvShift);
	inOutInfo.restorationSizes[2] = inOutInfo.restorationSizes[1];
	return oa::Status::ok();
}

static oa::U32 readLeBytes(const oa::U8* inData, oa::U32 inByteCount)
{
	oa::U32 value = 0;
	for (oa::U32 i = 0; i < inByteCount; ++i) {
		value |= static_cast<oa::U32>(inData[i]) << (i * 8u);
	}
	return value;
}

static oa::Status parseAv1ColorConfig(
	Av1BitReader& inReader,
	StdVideoAV1Profile inProfile,
	StdVideoAV1ColorConfig& outConfig)
{
	oa::U32 value = 0;
	bool flag = false;
	bool highBitDepth = false;
	if (!inReader.readBit(highBitDepth)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 high bit depth flag");
	}
	if (inProfile == STD_VIDEO_AV1_PROFILE_PROFESSIONAL && highBitDepth) {
		bool twelveBit = false;
		if (!inReader.readBit(twelveBit)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 twelve bit flag");
		}
		outConfig.BitDepth = twelveBit ? 12u : 10u;
	} else {
		outConfig.BitDepth = highBitDepth ? 10u : 8u;
	}

	bool monoChrome = false;
	if (inProfile != STD_VIDEO_AV1_PROFILE_HIGH) {
		if (!inReader.readBit(monoChrome)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 mono chrome flag");
		}
	}
	outConfig.flags.mono_chrome = monoChrome ? 1u : 0u;

	bool colorDescriptionPresent = false;
	if (!inReader.readBit(colorDescriptionPresent)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 color description flag");
	}
	outConfig.flags.color_description_present_flag = colorDescriptionPresent ? 1u : 0u;
	if (colorDescriptionPresent) {
		if (!inReader.readBits(8, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 color primaries");
		}
		outConfig.color_primaries = static_cast<StdVideoAV1ColorPrimaries>(value);
		if (!inReader.readBits(8, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 transfer characteristics");
		}
		outConfig.transfer_characteristics = static_cast<StdVideoAV1TransferCharacteristics>(value);
		if (!inReader.readBits(8, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 matrix coefficients");
		}
		outConfig.matrix_coefficients = static_cast<StdVideoAV1MatrixCoefficients>(value);
	} else {
		outConfig.color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED;
		outConfig.transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
		outConfig.matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED;
	}

	if (monoChrome) {
		if (!inReader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 mono chrome color range");
		}
		outConfig.flags.color_range = flag ? 1u : 0u;
		outConfig.subsampling_x = 1u;
		outConfig.subsampling_y = 1u;
		outConfig.flags.separate_uv_delta_q = 0u;
	} else {
		if (outConfig.color_primaries == STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709 &&
			outConfig.transfer_characteristics == STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SRGB &&
			outConfig.matrix_coefficients == STD_VIDEO_AV1_MATRIX_COEFFICIENTS_IDENTITY) {
			outConfig.subsampling_x = 0u;
			outConfig.subsampling_y = 0u;
			outConfig.flags.color_range = 1u;
		} else {
			bool colorRange = false;
			if (!inReader.readBit(colorRange)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 color range");
			}
			outConfig.flags.color_range = colorRange ? 1u : 0u;
			if (inProfile == STD_VIDEO_AV1_PROFILE_MAIN) {
				outConfig.subsampling_x = 1u;
				outConfig.subsampling_y = 1u;
			} else if (inProfile == STD_VIDEO_AV1_PROFILE_HIGH) {
				outConfig.subsampling_x = 0u;
				outConfig.subsampling_y = 0u;
			} else if (outConfig.BitDepth == 12u) {
				bool subX = false;
				if (!inReader.readBit(subX)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 chroma subsampling X");
				}
				outConfig.subsampling_x = subX ? 1u : 0u;
				if (subX) {
					bool subY = false;
					if (!inReader.readBit(subY)) {
						return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 chroma subsampling Y");
					}
					outConfig.subsampling_y = subY ? 1u : 0u;
				} else {
					outConfig.subsampling_y = 0u;
				}
			} else {
				outConfig.subsampling_x = 1u;
				outConfig.subsampling_y = 0u;
			}
		}
		if (outConfig.subsampling_x != 0u && outConfig.subsampling_y != 0u) {
			if (!inReader.readBits(2, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 chroma sample position");
			}
			outConfig.chroma_sample_position = static_cast<StdVideoAV1ChromaSamplePosition>(value);
		}
		if (!inReader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 separate uv delta q");
		}
		outConfig.flags.separate_uv_delta_q = flag ? 1u : 0u;
	}
	return oa::Status::ok();
}

static oa::Status parseAv1SequenceHeader(const oa::Av1Obu& inObu, const oa::Span<const oa::U8>& inFrame, oa::Av1SequenceHeaderInfo& outInfo)
{
	if (inObu.payloadSize == 0 || inObu.payloadOffset + inObu.payloadSize > inFrame.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 sequence header payload");
	}

	Av1BitReader reader(inFrame.data() + inObu.payloadOffset, inObu.payloadSize);
	oa::U32 value = 0;
	bool flag = false;
	if (!reader.readBits(3, value) || !reader.readBit(outInfo.stillPicture) || !reader.readBit(outInfo.reducedStillPictureHeader)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 sequence header");
	}
	outInfo.seqProfile = static_cast<StdVideoAV1Profile>(value);

	bool decoderModelInfoPresent = false;
	oa::U32 bufferDelayLengthMinus1 = 0;
	if (outInfo.reducedStillPictureHeader) {
		value = 0;
	} else {
		if (!reader.readBit(outInfo.timingInfoPresent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 sequence timing flag");
		}
		if (outInfo.timingInfoPresent) {
			if (!reader.readBits(32, outInfo.timingInfo.num_units_in_display_tick)
				|| !reader.readBits(32, outInfo.timingInfo.time_scale)
				|| !reader.readBit(flag)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 timing info");
			}
			outInfo.timingInfo.flags.equal_picture_interval = flag ? 1u : 0u;
			if (flag && !reader.readUvlc(outInfo.timingInfo.num_ticks_per_picture_minus_1)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument,
					"truncated AV1 equal picture interval");
			}
			if (!reader.readBit(decoderModelInfoPresent)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument,
					"truncated AV1 decoder model flag");
			}
			if (decoderModelInfoPresent
				&& (!reader.readBits(5, bufferDelayLengthMinus1)
					|| !reader.skipBits(32 + 5 + 5))) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 decoder model info");
			}
		}
		if (!reader.readBit(outInfo.initialDisplayDelayPresent) || !reader.readBits(5, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 operating point header");
		}
	}

	const oa::U32 operatingPoints = value + 1u;
	for (oa::U32 i = 0; i < operatingPoints; ++i) {
		if (!reader.skipBits(12) || !reader.readBits(5, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 operating point");
		}
		if (value > 7 && !reader.skipBits(1)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 operating point tier");
		}
		if (decoderModelInfoPresent) {
			if (!reader.readBit(flag)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 decoder model flag");
			}
			const oa::U32 delayBits = bufferDelayLengthMinus1 + 1u;
			if (flag && !reader.skipBits(delayBits + delayBits + 1u)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 operating decoder model");
			}
		}
		if (outInfo.initialDisplayDelayPresent) {
			if (!reader.readBit(flag)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 initial display delay flag");
			}
			if (flag && !reader.skipBits(4)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 initial display delay");
			}
		}
	}

	if (!reader.readBits(4, outInfo.frameWidthBitsMinus1) ||
		!reader.readBits(4, outInfo.frameHeightBitsMinus1) ||
		!reader.readBits(outInfo.frameWidthBitsMinus1 + 1u, outInfo.maxFrameWidthMinus1) ||
		!reader.readBits(outInfo.frameHeightBitsMinus1 + 1u, outInfo.maxFrameHeightMinus1)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame size bits");
	}

	if (outInfo.reducedStillPictureHeader) {
		outInfo.frameIdNumbersPresent = false;
	} else {
		if (!reader.readBit(outInfo.frameIdNumbersPresent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame id flag");
		}
		if (outInfo.frameIdNumbersPresent) {
			oa::U32 deltaLen = 0;
			oa::U32 addLen = 0;
			if (!reader.readBits(4, deltaLen) || !reader.readBits(3, addLen)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame id fields");
			}
			outInfo.deltaFrameIdLengthMinus2 = static_cast<oa::U8>(deltaLen);
			outInfo.additionalFrameIdLengthMinus1 = static_cast<oa::U8>(addLen);
			const oa::U32 frameIdLength = 3u + (deltaLen + 2u) + (addLen + 1u);
			if (frameIdLength > 16u) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame id length");
			}
		}
	}

	if (!reader.readBit(outInfo.use128x128Superblock) ||
		!reader.readBit(outInfo.enableFilterIntra) ||
		!reader.readBit(outInfo.enableIntraEdgeFilter)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 coding tool flags");
	}
	if (outInfo.reducedStillPictureHeader) {
		outInfo.enableInterIntraCompound = false;
		outInfo.enableMaskedCompound = false;
		outInfo.enableWarpedMotion = false;
		outInfo.enableDualFilter = false;
		outInfo.enableOrderHint = false;
		outInfo.enableJntComp = false;
		outInfo.enableRefFrameMvs = false;
		outInfo.seqForceScreenContentTools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS;
		outInfo.seqForceIntegerMv = STD_VIDEO_AV1_SELECT_INTEGER_MV;
		outInfo.orderHintBits = 0;
	} else {
		if (!reader.readBit(outInfo.enableInterIntraCompound) ||
			!reader.readBit(outInfo.enableMaskedCompound) ||
			!reader.readBit(outInfo.enableWarpedMotion) ||
			!reader.readBit(outInfo.enableDualFilter) ||
			!reader.readBit(outInfo.enableOrderHint)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 inter tool flags");
		}
		if (outInfo.enableOrderHint) {
			if (!reader.readBit(outInfo.enableJntComp) || !reader.readBit(outInfo.enableRefFrameMvs)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 order-hint tool flags");
			}
		} else {
			outInfo.enableJntComp = false;
			outInfo.enableRefFrameMvs = false;
		}
		bool choose = false;
		if (!reader.readBit(choose)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 screen-content tool flag");
		}
		if (choose) {
			outInfo.seqForceScreenContentTools = 2;
		} else if (reader.readBit(flag)) {
			outInfo.seqForceScreenContentTools = flag ? 1 : 0;
		} else {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 screen-content tool value");
		}
		if (outInfo.seqForceScreenContentTools > 0) {
			if (!reader.readBit(choose)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 integer-mv tool flag");
			}
			if (choose) {
				outInfo.seqForceIntegerMv = 2;
			} else if (reader.readBit(flag)) {
				outInfo.seqForceIntegerMv = flag ? 1 : 0;
			} else {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 integer-mv tool value");
			}
		}
		if (outInfo.enableOrderHint && !reader.readBits(3, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 order hint bits");
		}
		outInfo.orderHintBits = outInfo.enableOrderHint ? value + 1u : 0u;
	}
	if (!reader.readBit(outInfo.enableSuperres) || !reader.readBit(outInfo.enableCdef) || !reader.readBit(outInfo.enableRestoration)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 post-filter flags");
	}
	OA_RETURN_IF_ERROR(parseAv1ColorConfig(reader, outInfo.seqProfile, outInfo.colorConfig));
	if (!reader.readBit(outInfo.filmGrainParamsPresent)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 film grain present flag");
	}
	return oa::Status::ok();
}

static oa::Status parseAv1FrameHeader(
	const oa::Av1Obu& inObu,
	const oa::Span<const oa::U8>& inFrame,
	const oa::Av1SequenceHeaderInfo& inSeq,
	const oa::Array<oa::U8, STD_VIDEO_AV1_NUM_REF_FRAMES>& inRefOrderHints,
	const oa::Array<bool, STD_VIDEO_AV1_NUM_REF_FRAMES>& inRefValid,
	oa::Av1FrameHeaderInfo& outInfo)
{
	if (inObu.payloadSize == 0 || inObu.payloadOffset + inObu.payloadSize > inFrame.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 frame payload");
	}

	Av1BitReader reader(inFrame.data() + inObu.payloadOffset, inObu.payloadSize);
	oa::Array<oa::U8, STD_VIDEO_AV1_NUM_REF_FRAMES> logicalOrderHints = {};
	for (oa::U32 ref = 0; ref < STD_VIDEO_AV1_NUM_REF_FRAMES; ++ref) {
		logicalOrderHints[ref] = inRefValid[ref] ? inRefOrderHints[ref] : 0;
	}
	bool flag = false;
	oa::U32 value = 0;
	if (!inSeq.reducedStillPictureHeader) {
		if (!reader.readBit(flag)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 show-existing-frame flag");
		}
		if (flag) {
			// show_existing_frame: this is a display-only operation. No new compressed
			// picture is decoded; we simply point at a previous reference picture
			// (identified by frame_to_show_map_idx into the 8-ref array).
			// The decoder layer will resolve it to a DPB slot and return the prior
			// frame's resources without submitting vkCmdDecodeVideoKHR work.
			oa::U32 idx = 0;
			if (!reader.readBits(3, idx)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame_to_show_map_idx");
			}
			outInfo.showExistingFrame = true;
			outInfo.frameToShowMapIdx = static_cast<oa::U8>(idx);
			outInfo.headerSize = reader.byteOffset();
			return oa::Status::ok();
		}
	}

	if (!reader.readBits(2, value)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame type");
	}
	outInfo.frameType = static_cast<StdVideoAV1FrameType>(value);
	outInfo.isKeyFrame = value == 0;
	if (!reader.readBit(outInfo.showFrame)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 show-frame flag");
	}
	if (!outInfo.showFrame && !reader.readBit(outInfo.showableFrame)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 showable-frame flag");
	}

	// AV1 §5.9.2 uncompressed_header(). The ordering below is exact: every field
	// must consume precisely the bits the spec says before tile_info(), or all
	// downstream syntax (tiles, quant, ...) misaligns. The previous code only
	// handled the intra layout and skipped error_resilient_mode, force_integer_mv,
	// frame_size_with_refs, the interpolation filter and motion fields — which
	// drifted every inter frame header by ~tens of bits.
	const bool frameIsIntra = (outInfo.frameType == STD_VIDEO_AV1_FRAME_TYPE_KEY)
		|| (outInfo.frameType == STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY);
	const bool frameTypeSwitch = (outInfo.frameType == STD_VIDEO_AV1_FRAME_TYPE_SWITCH);
	const bool keyShown = (outInfo.frameType == STD_VIDEO_AV1_FRAME_TYPE_KEY) && outInfo.showFrame;

	bool errorResilientMode = false;
	if (frameTypeSwitch || keyShown) {
		errorResilientMode = true;
	} else if (!reader.readBit(errorResilientMode)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 error-resilient flag");
	}
	outInfo.errorResilientMode = errorResilientMode || frameTypeSwitch || keyShown;

	if (!reader.readBit(outInfo.disableCdfUpdate)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 disable-cdf-update flag");
	}
	if (inSeq.seqForceScreenContentTools == 2) {
		if (!reader.readBit(outInfo.allowScreenContentTools)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 screen-content frame flag");
		}
	} else {
		outInfo.allowScreenContentTools = inSeq.seqForceScreenContentTools != 0;
	}
	bool forceIntegerMv = false;
	if (outInfo.allowScreenContentTools) {
		if (inSeq.seqForceIntegerMv == 2) {
			if (!reader.readBit(forceIntegerMv)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 force-integer-mv flag");
			}
		} else {
			forceIntegerMv = inSeq.seqForceIntegerMv != 0;
		}
	}
	if (frameIsIntra) {
		forceIntegerMv = true;
	}
	outInfo.forceIntegerMv = forceIntegerMv;

	// frame_id_numbers_present is assumed 0 (consistent with the sequence-header
	// parse, which never enables it); so no current_frame_id / delta_frame_id bits.

	bool frameSizeOverride = false;
	if (frameTypeSwitch) {
		frameSizeOverride = true;
	} else if (inSeq.reducedStillPictureHeader) {
		frameSizeOverride = false;
	} else if (!reader.readBit(frameSizeOverride)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame-size-override flag");
	}
	outInfo.frameSizeOverrideFlag = frameSizeOverride;

	if (inSeq.orderHintBits > 0) {
		if (!reader.readBits(inSeq.orderHintBits, outInfo.orderHint)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 order hint");
		}
	}

	if (frameIsIntra || errorResilientMode) {
		outInfo.primaryRefFrame = STD_VIDEO_AV1_PRIMARY_REF_NONE;
	} else {
		if (!reader.readBits(3, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 primary_ref_frame");
		}
		outInfo.primaryRefFrame = value;
	}

	// decoder_model_info_present assumed 0 → no buffer_removal_time fields.

	if (frameTypeSwitch || keyShown) {
		outInfo.refreshFrameFlags = 0xffu;
	} else {
		if (!reader.readBits(8, outInfo.refreshFrameFlags)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 refresh_frame_flags");
		}
	}

	// ref_order_hint[] is only coded for error-resilient frames with order hints.
	if ((!frameIsIntra || outInfo.refreshFrameFlags != 0xffu) && errorResilientMode && inSeq.enableOrderHint) {
		for (oa::U32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
			if (!reader.readBits(inSeq.orderHintBits, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 ref order hint");
			}
			logicalOrderHints[i] = static_cast<oa::U8>(value);
		}
	}

	// Shared size readers (AV1 §5.9.5 frame_size / §5.9.6 render_size /
	// §5.9.8 superres_params). superres uses SUPERRES_DENOM_BITS = 3; render
	// size uses two fixed 16-bit fields, not the frame-size bit widths.
	auto readSuperres = [&]() -> oa::Status {
		outInfo.useSuperres = false;
		outInfo.codedDenom = 0;
		if (inSeq.enableSuperres) {
			if (!reader.readBit(outInfo.useSuperres)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 superres flag");
			}
			if (outInfo.useSuperres) {
				oa::U32 denom = 0;
				if (!reader.readBits(3, denom)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 superres denominator");
				}
				outInfo.codedDenom = static_cast<oa::U8>(denom);
			}
		}
		return oa::Status::ok();
	};
	auto readFrameSize = [&]() -> oa::Status {
		if (frameSizeOverride) {
			if (!reader.skipBits((inSeq.frameWidthBitsMinus1 + 1u) + (inSeq.frameHeightBitsMinus1 + 1u))) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame size");
			}
		}
		return readSuperres();
	};
	auto readRenderSize = [&]() -> oa::Status {
		if (!reader.readBit(outInfo.renderAndFrameSizeDifferent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 render-size flag");
		}
		if (outInfo.renderAndFrameSizeDifferent && !reader.skipBits(16 + 16)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 render size");
		}
		return oa::Status::ok();
	};

	if (frameIsIntra) {
		OA_RETURN_IF_ERROR(readFrameSize());
		OA_RETURN_IF_ERROR(readRenderSize());
		if (outInfo.allowScreenContentTools) {
			// allow_intrabc (present when UpscaledWidth == frameWidth; true when
			// superres is inactive, the common case for non-screen content).
			if (!reader.readBit(outInfo.allowIntraBc)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 allow_intrabc");
			}
		}
	} else {
		if (inSeq.enableOrderHint) {
			if (!reader.readBit(outInfo.frameRefsShortSignaling)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame_refs_short_signaling");
			}
			if (outInfo.frameRefsShortSignaling) {
				// last_frame_idx f(3), gold_frame_idx f(3); set_frame_refs() derives the rest (no bits).
				if (!reader.skipBits(3 + 3)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 short ref signaling");
				}
			}
		}
		for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
			if (!outInfo.frameRefsShortSignaling) {
				if (!reader.readBits(3, value)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 ref_frame_idx");
				}
				outInfo.referenceNameSlotIndices[i] = static_cast<oa::I32>(value);
			}
			// frame_id_numbers_present assumed 0 → no delta_frame_id_minus_1.
		}
		// StdVideoDecodeAV1PictureInfo::orderHints is indexed by reference
		// *name* (INTRA_FRAME, LAST_FRAME ... ALTREF_FRAME), not by the eight
		// logical ref-frame-map slots addressed by ref_frame_idx[].
		outInfo.orderHints[0] = 0;
		for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
			const oa::I32 logical = outInfo.referenceNameSlotIndices[i];
			outInfo.orderHints[i + 1u] = logical >= 0
				&& static_cast<oa::U32>(logical) < logicalOrderHints.size()
				? logicalOrderHints[static_cast<oa::Usize>(logical)]
				: 0;
		}
		if (frameSizeOverride && !errorResilientMode) {
			// frame_size_with_refs(): a found_ref bit per reference; the first set
			// one adopts that ref's size, otherwise an explicit frame/render size.
			bool foundRef = false;
			for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
				if (!reader.readBit(foundRef)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 found_ref");
				}
				if (foundRef) {
					break;
				}
			}
			if (!foundRef) {
				OA_RETURN_IF_ERROR(readFrameSize());
				OA_RETURN_IF_ERROR(readRenderSize());
			} else {
				OA_RETURN_IF_ERROR(readSuperres());
			}
		} else {
			OA_RETURN_IF_ERROR(readFrameSize());
			OA_RETURN_IF_ERROR(readRenderSize());
		}
		if (forceIntegerMv) {
			outInfo.allowHighPrecisionMv = false;
		} else if (!reader.readBit(outInfo.allowHighPrecisionMv)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 allow_high_precision_mv");
		}
		// read_interpolation_filter(): is_filter_switchable f(1); if 0, interpolation_filter f(2).
		bool filterSwitchable = false;
		if (!reader.readBit(filterSwitchable)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 is_filter_switchable");
		}
		outInfo.isFilterSwitchable = filterSwitchable;
		if (filterSwitchable) {
			outInfo.interpolationFilter = STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE;
		} else if (!reader.readBits(2, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 interpolation_filter");
		} else {
			outInfo.interpolationFilter = static_cast<StdVideoAV1InterpolationFilter>(value);
		}
		if (!reader.readBit(outInfo.isMotionModeSwitchable)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 is_motion_mode_switchable");
		}
		if (errorResilientMode || !inSeq.enableRefFrameMvs) {
			outInfo.useRefFrameMvs = false;
		} else if (!reader.readBit(outInfo.useRefFrameMvs)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 use_ref_frame_mvs");
		}
	}

	if (frameIsIntra) {
		outInfo.useRefFrameMvs = false;
	}

	// disable_frame_end_update_cdf is the last field before tile_info().
	if (inSeq.reducedStillPictureHeader || outInfo.disableCdfUpdate) {
		outInfo.disableFrameEndUpdateCdf = true;
	} else if (!reader.readBit(outInfo.disableFrameEndUpdateCdf)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 frame-end-cdf flag");
	}

	const oa::U32 frameWidth = inSeq.maxFrameWidthMinus1 + 1u;
	const oa::U32 frameHeight = inSeq.maxFrameHeightMinus1 + 1u;
	const oa::U32 miCols = (frameWidth + 7u) >> 3u << 1u;
	const oa::U32 miRows = (frameHeight + 7u) >> 3u << 1u;
	const oa::U32 sbShift = inSeq.use128x128Superblock ? 5u : 4u;
	const oa::U32 sbCols = (miCols + (1u << sbShift) - 1u) >> sbShift;
	const oa::U32 sbRows = (miRows + (1u << sbShift) - 1u) >> sbShift;
	const oa::U32 maxTileWidthSb = inSeq.use128x128Superblock ? 32u : 64u;
	const oa::U32 maxTileAreaSb = 4096u >> (2u * (inSeq.use128x128Superblock ? 1u : 0u));
	const oa::U32 minLog2TileCols = av1FloorLog2((sbCols + maxTileWidthSb - 1u) / maxTileWidthSb);
	oa::U32 maxLog2TileCols = av1FloorLog2(sbCols == 0 ? 1u : sbCols);
	while (maxLog2TileCols > 0 && ((sbRows * (sbCols + (1u << maxLog2TileCols) - 1u)) >> maxLog2TileCols) > maxTileAreaSb) {
		--maxLog2TileCols;
	}
	oa::U32 minLog2Tiles = av1FloorLog2((sbRows * sbCols + maxTileAreaSb - 1u) / maxTileAreaSb);

	if (!reader.readBit(flag)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile spacing flag");
	}
	if (!flag) {
		return oa::Status::error(oa::StatusCode::Unavailable, "Only uniform AV1 tile spacing is implemented");
	}

	oa::U32 tileColsLog2 = minLog2TileCols;
	while (tileColsLog2 < maxLog2TileCols) {
		bool increment = false;
		if (!reader.readBit(increment)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile columns");
		}
		if (!increment) {
			break;
		}
		++tileColsLog2;
	}
	oa::U32 tileRowsLog2 = minLog2Tiles > tileColsLog2 ? minLog2Tiles - tileColsLog2 : 0u;
	while (tileRowsLog2 < av1FloorLog2(sbRows == 0 ? 1u : sbRows)) {
		bool increment = false;
		if (!reader.readBit(increment)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile rows");
		}
		if (!increment) {
			break;
		}
		++tileRowsLog2;
	}
	outInfo.tileCols = 1u << tileColsLog2;
	outInfo.tileRows = 1u << tileRowsLog2;
	outInfo.tileColsLog2 = tileColsLog2;
	outInfo.tileRowsLog2 = tileRowsLog2;
	if (outInfo.tileCols * outInfo.tileRows > 1u) {
		oa::U32 contextBits = outInfo.tileRowsLog2 + outInfo.tileColsLog2;
		if (!reader.readBits(contextBits, outInfo.contextUpdateTileId)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 context_update_tile_id");
		}
		if (!reader.readBits(2, outInfo.tileSizeBytesMinus1)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile size byte count");
		}
	}

	if (!reader.readBits(8, outInfo.baseQIdx)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 quantization header");
	}
	OA_RETURN_IF_ERROR(readAv1DeltaQ(reader, outInfo.deltaQYDc));
	if (!inSeq.colorConfig.flags.mono_chrome) {
		if (inSeq.colorConfig.flags.separate_uv_delta_q) {
			if (!reader.readBit(outInfo.diffUvDelta)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 diff_uv_delta flag");
			}
		}
		OA_RETURN_IF_ERROR(readAv1DeltaQ(reader, outInfo.deltaQUDc));
		OA_RETURN_IF_ERROR(readAv1DeltaQ(reader, outInfo.deltaQUAc));
		if (outInfo.diffUvDelta) {
			OA_RETURN_IF_ERROR(readAv1DeltaQ(reader, outInfo.deltaQVDc));
			OA_RETURN_IF_ERROR(readAv1DeltaQ(reader, outInfo.deltaQVAc));
		} else {
			outInfo.deltaQVDc = outInfo.deltaQUDc;
			outInfo.deltaQVAc = outInfo.deltaQUAc;
		}
	}
	if (!reader.readBit(outInfo.usingQMatrix)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 qmatrix flag");
	}
	if (outInfo.usingQMatrix) {
		if (!reader.readBits(4, outInfo.qmY) || !reader.readBits(4, outInfo.qmU)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 qmatrix header");
		}
		outInfo.qmV = outInfo.qmU;
	}
	if (!reader.readBit(outInfo.segmentationEnabled)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 segmentation flag");
	}
	if (outInfo.segmentationEnabled) {
		outInfo.segmentationUpdateMap = true;
		outInfo.segmentationTemporalUpdate = false;
		outInfo.segmentationUpdateData = true;
		static constexpr oa::U8 featureBits[STD_VIDEO_AV1_SEG_LVL_MAX] = {8, 6, 6, 6, 6, 3, 0, 0};
		static constexpr bool featureSigned[STD_VIDEO_AV1_SEG_LVL_MAX] = {true, true, true, true, true, false, false, false};
		for (oa::U32 segment = 0; segment < STD_VIDEO_AV1_MAX_SEGMENTS; ++segment) {
			for (oa::U32 feature = 0; feature < STD_VIDEO_AV1_SEG_LVL_MAX; ++feature) {
				bool enabled = false;
				if (!reader.readBit(enabled)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 segmentation feature flag");
				}
				outInfo.segmentFeatureEnabled[segment][feature] = enabled ? 1u : 0u;
				if (!enabled) {
					continue;
				}
				if (featureSigned[feature]) {
					OA_RETURN_IF_ERROR(readAv1SignedFeatureData(reader, featureBits[feature], outInfo.segmentFeatureData[segment][feature]));
				} else if (featureBits[feature] > 0) {
					if (!reader.readBits(featureBits[feature], value)) {
						return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 segmentation feature data");
					}
					outInfo.segmentFeatureData[segment][feature] = static_cast<oa::I16>(value);
				}
			}
		}
	}
	if (outInfo.baseQIdx > 0) {
		if (!reader.readBit(outInfo.deltaQPresent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 delta-q flag");
		}
		if (outInfo.deltaQPresent && !reader.readBits(2, outInfo.deltaQRes)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 delta-q resolution");
		}
	}
	if (outInfo.deltaQPresent) {
		if (!reader.readBit(outInfo.deltaLfPresent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 delta-lf flag");
		}
		if (outInfo.deltaLfPresent &&
			(!reader.readBits(2, outInfo.deltaLfRes) || !reader.readBit(outInfo.deltaLfMulti))) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 delta-lf header");
		}
	}
	outInfo.loopFilterLevels[2] = 0;
	outInfo.loopFilterLevels[3] = 0;
	for (oa::U32 i = 0; i < 2; ++i) {
		if (!reader.readBits(6, value)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 loop filter level");
		}
		outInfo.loopFilterLevels[i] = static_cast<oa::U8>(value);
	}
	if (!inSeq.colorConfig.flags.mono_chrome &&
		(outInfo.loopFilterLevels[0] != 0 || outInfo.loopFilterLevels[1] != 0)) {
		for (oa::U32 i = 2; i < 4; ++i) {
			if (!reader.readBits(6, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 chroma loop filter level");
			}
			outInfo.loopFilterLevels[i] = static_cast<oa::U8>(value);
		}
	}
	if (!reader.readBits(3, outInfo.loopFilterSharpness) || !reader.readBit(outInfo.loopFilterDeltaEnabled)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 loop filter");
	}
	if (outInfo.loopFilterDeltaEnabled) {
		if (!reader.readBit(outInfo.loopFilterDeltaUpdate)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 loop filter delta update flag");
		}
		if (outInfo.loopFilterDeltaUpdate) {
			for (oa::U32 i = 0; i < STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME; ++i) {
				bool updateRefDelta = false;
				if (!reader.readBit(updateRefDelta)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 loop filter ref delta flag");
				}
				outInfo.loopFilterUpdateRefDelta[i] = updateRefDelta ? 1u : 0u;
				if (updateRefDelta) {
					OA_RETURN_IF_ERROR(readAv1InverseSignedLiteral(reader, 6, outInfo.loopFilterRefDeltas[i]));
				}
			}
			for (oa::U32 i = 0; i < STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS; ++i) {
				bool updateModeDelta = false;
				if (!reader.readBit(updateModeDelta)) {
					return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 loop filter mode delta flag");
				}
				outInfo.loopFilterUpdateModeDelta[i] = updateModeDelta ? 1u : 0u;
				if (updateModeDelta) {
					OA_RETURN_IF_ERROR(readAv1InverseSignedLiteral(reader, 6, outInfo.loopFilterModeDeltas[i]));
				}
			}
		}
	}
	if (inSeq.enableCdef) {
		if (!reader.readBits(2, outInfo.cdefDampingMinus3) || !reader.readBits(2, outInfo.cdefBits)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 CDEF header");
		}
		const oa::U32 cdefStrengthCount = 1u << outInfo.cdefBits;
		for (oa::U32 i = 0; i < cdefStrengthCount; ++i) {
			if (!reader.readBits(4, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 CDEF luma primary strength");
			}
			outInfo.cdefYPriStrength[i] = static_cast<oa::U8>(value);
			if (!reader.readBits(2, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 CDEF luma secondary strength");
			}
			outInfo.cdefYSecStrength[i] = static_cast<oa::U8>(value);
			if (!reader.readBits(4, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 CDEF chroma primary strength");
			}
			outInfo.cdefUvPriStrength[i] = static_cast<oa::U8>(value);
			if (!reader.readBits(2, value)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 CDEF chroma secondary strength");
			}
			outInfo.cdefUvSecStrength[i] = static_cast<oa::U8>(value);
		}
	}
	if (inSeq.enableRestoration) {
		OA_RETURN_IF_ERROR(parseAv1RestorationHeader(reader, inSeq.use128x128Superblock, outInfo));
	}
	if (!reader.readBit(flag)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 transform header");
	}
	outInfo.txMode = flag ? STD_VIDEO_AV1_TX_MODE_SELECT : STD_VIDEO_AV1_TX_MODE_LARGEST;

	// AV1 uncompressed_header() tail. These fields are before byte_alignment()
	// and therefore before the tile payload. Omitting even one shifts every tile
	// offset and produces apparently successful but corrupt hardware decode.
	if (!frameIsIntra && !reader.readBit(outInfo.referenceSelect)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 reference_select");
	}

	auto relativeDist = [&inSeq](oa::U32 inA, oa::U32 inB) -> oa::I32 {
		if (!inSeq.enableOrderHint || inSeq.orderHintBits == 0) return 0;
		const oa::U32 modulus = 1u << inSeq.orderHintBits;
		const oa::U32 mask = modulus - 1u;
		const oa::U32 sign = modulus >> 1u;
		const oa::U32 diff = (inA - inB) & mask;
		return (diff & sign) != 0u
			? static_cast<oa::I32>(diff) - static_cast<oa::I32>(modulus)
			: static_cast<oa::I32>(diff);
	};
	// ref_frame_sign_bias is stored with the reconstructed picture and consumed
	// again when that picture is bound as a vulkan DPB reference. Bit zero is
	// INTRA_FRAME; bits 1..7 correspond to LAST..aLTREF.
	outInfo.refFrameSignBias = 0;
	if (!frameIsIntra && inSeq.enableOrderHint) {
		for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
			const oa::I32 logical = outInfo.referenceNameSlotIndices[i];
			if (logical < 0 || static_cast<oa::U32>(logical) >= inRefValid.size()
				|| !inRefValid[static_cast<oa::Usize>(logical)]) continue;
			const oa::U32 refHint = inRefOrderHints[static_cast<oa::Usize>(logical)];
			if (relativeDist(refHint, outInfo.orderHint) > 0) {
				outInfo.refFrameSignBias |= static_cast<oa::U8>(1u << (i + 1u));
			}
		}
	}
	oa::I32 nearestForward = -1;
	oa::I32 nearestBackward = -1;
	oa::I32 nearestForwardHint = 0;
	oa::I32 nearestBackwardHint = 0;
	if (!frameIsIntra && outInfo.referenceSelect && inSeq.enableOrderHint) {
		for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
			const oa::I32 logical = outInfo.referenceNameSlotIndices[i];
			if (logical < 0 || static_cast<oa::U32>(logical) >= inRefValid.size()
				|| !inRefValid[static_cast<oa::Usize>(logical)]) continue;
			const oa::I32 hint = inRefOrderHints[static_cast<oa::Usize>(logical)];
			const oa::I32 dist = relativeDist(static_cast<oa::U32>(hint), outInfo.orderHint);
			if (dist < 0 && (nearestForward < 0
				|| relativeDist(static_cast<oa::U32>(hint), static_cast<oa::U32>(nearestForwardHint)) > 0)) {
				nearestForward = static_cast<oa::I32>(i);
				nearestForwardHint = hint;
			}
			if (dist > 0 && (nearestBackward < 0
				|| relativeDist(static_cast<oa::U32>(hint), static_cast<oa::U32>(nearestBackwardHint)) < 0)) {
				nearestBackward = static_cast<oa::I32>(i);
				nearestBackwardHint = hint;
			}
		}
	}
	oa::I32 secondForward = -1;
	oa::I32 secondForwardHint = 0;
	if (nearestForward >= 0 && nearestBackward < 0) {
		for (oa::U32 i = 0; i < oa::Av1MaxReferencesPerFrame; ++i) {
			const oa::I32 logical = outInfo.referenceNameSlotIndices[i];
			if (logical < 0 || static_cast<oa::U32>(logical) >= inRefValid.size()
				|| !inRefValid[static_cast<oa::Usize>(logical)]) continue;
			const oa::I32 hint = inRefOrderHints[static_cast<oa::Usize>(logical)];
			if (relativeDist(static_cast<oa::U32>(hint), static_cast<oa::U32>(nearestForwardHint)) < 0
				&& (secondForward < 0 || relativeDist(static_cast<oa::U32>(hint),
					static_cast<oa::U32>(secondForwardHint)) > 0)) {
				secondForward = static_cast<oa::I32>(i);
				secondForwardHint = hint;
			}
		}
	}
	const oa::I32 skipOther = nearestBackward >= 0 ? nearestBackward : secondForward;
	if (nearestForward >= 0 && skipOther >= 0) {
		if (!reader.readBit(outInfo.skipModePresent)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 skip_mode_present");
		}
		const oa::U8 ref0 = static_cast<oa::U8>(nearestForward + STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME);
		const oa::U8 ref1 = static_cast<oa::U8>(skipOther + STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME);
		outInfo.skipModeFrame[0] = ref0 < ref1 ? ref0 : ref1;
		outInfo.skipModeFrame[1] = ref0 < ref1 ? ref1 : ref0;
	}

	if (!frameIsIntra && !outInfo.errorResilientMode && inSeq.enableWarpedMotion) {
		if (!reader.readBit(outInfo.allowWarpedMotion)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 allow_warped_motion");
		}
	}
	if (!reader.readBit(outInfo.reducedTxSet)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 reduced_tx_set");
	}

	if (!frameIsIntra) {
		for (oa::U32 ref = 0; ref < oa::Av1MaxReferencesPerFrame; ++ref) {
			bool isGlobal = false;
			if (!reader.readBit(isGlobal)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 global motion flag");
			}
			if (isGlobal) {
				return oa::Status::error(oa::StatusCode::Unimplemented,
					"AV1 non-identity global motion is not implemented");
			}
		}
	}
	if (inSeq.filmGrainParamsPresent && (outInfo.showFrame || outInfo.showableFrame)) {
		if (!reader.readBit(outInfo.applyGrain)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 apply_grain");
		}
		if (outInfo.applyGrain) {
			return oa::Status::error(oa::StatusCode::Unimplemented,
				"AV1 film-grain parameters are not implemented");
		}
	}
	reader.byteAlign();
	if (reader.byteOffset() >= inObu.payloadSize) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 frame header consumes the full frame payload");
	}
	outInfo.headerSize = reader.byteOffset();
	return oa::Status::ok();
}

static oa::Status parseAv1TileGroup(
	const oa::Av1Obu& inObu,
	const oa::Span<const oa::U8>& inFrame,
	const oa::Av1FrameHeaderInfo& inFrameHeader,
	bool inPayloadIncludesFrameHeader,
	oa::Av1TileGroupInfo& outTileGroup)
{
	outTileGroup.tileOffsets.clear();
	outTileGroup.tileSizes.clear();
	if (inObu.payloadOffset + inObu.payloadSize > inFrame.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}
	if (inPayloadIncludesFrameHeader && inObu.payloadSize <= inFrameHeader.headerSize) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}
	const oa::Usize tileGroupOffset = inPayloadIncludesFrameHeader
		? inObu.payloadOffset + inFrameHeader.headerSize
		: inObu.payloadOffset;
	const oa::Usize tileGroupSize = inPayloadIncludesFrameHeader
		? inObu.payloadSize - inFrameHeader.headerSize
		: inObu.payloadSize;
	if (tileGroupOffset > inFrame.size() || tileGroupSize == 0 || tileGroupOffset + tileGroupSize > inFrame.size()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}

	const oa::U32 tileCount = inFrameHeader.tileCols * inFrameHeader.tileRows;
	if (tileCount == 0 || tileCount > STD_VIDEO_AV1_MAX_TILE_COLS * STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile count");
	}
	if (tileCount == 1) {
		outTileGroup.tileOffsets.pushBack(static_cast<oa::U32>(tileGroupOffset));
		outTileGroup.tileSizes.pushBack(static_cast<oa::U32>(tileGroupSize));
		return oa::Status::ok();
	}

	Av1BitReader reader(inFrame.data() + tileGroupOffset, tileGroupSize);
	bool hasTileStartAndEnd = false;
	if (!reader.readBit(hasTileStartAndEnd)) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile group range flag");
	}

	const oa::U32 tileNumberBits = inFrameHeader.tileColsLog2 + inFrameHeader.tileRowsLog2;
	oa::U32 tileStart = 0;
	oa::U32 tileEnd = tileCount - 1u;
	if (hasTileStartAndEnd) {
		if (!reader.readBits(tileNumberBits, tileStart) || !reader.readBits(tileNumberBits, tileEnd)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile group range");
		}
		if (tileStart > tileEnd || tileEnd >= tileCount) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile group range");
		}
	}
	reader.byteAlign();

	const oa::U32 tileSizeBytes = inFrameHeader.tileSizeBytesMinus1 + 1u;
	if (tileSizeBytes == 0 || tileSizeBytes > 4) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 tile size byte count");
	}

	oa::Usize cursor = tileGroupOffset + reader.byteOffset();
	for (oa::U32 tile = tileStart; tile <= tileEnd; ++tile) {
		oa::U32 tileSize = 0;
		if (tile == tileEnd) {
			if (cursor > inObu.payloadOffset + inObu.payloadSize) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile cursor exceeds payload");
			}
			tileSize = static_cast<oa::U32>(inObu.payloadOffset + inObu.payloadSize - cursor);
		} else {
			if (cursor + tileSizeBytes > inObu.payloadOffset + inObu.payloadSize) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 tile size field");
			}
			tileSize = readLeBytes(inFrame.data() + cursor, tileSizeBytes) + 1u;
			cursor += tileSizeBytes;
			if (cursor + tileSize > inObu.payloadOffset + inObu.payloadSize) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile payload exceeds tile group");
			}
		}
		if (tileSize == 0) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 tile payload is empty");
		}
		outTileGroup.tileOffsets.pushBack(static_cast<oa::U32>(cursor));
		outTileGroup.tileSizes.pushBack(tileSize);
		cursor += tileSize;
	}
	return oa::Status::ok();
}

static oa::Status extractAv1FramePayload(const oa::Span<const oa::U8>& inBitstream, oa::Av1IvfFrame& outFrame)
{
	const oa::U8* data = inBitstream.data();
	const oa::Usize size = inBitstream.size();

	// Support both IVF-wrapped (our test assets) and raw OBU payload (real demuxer access units).
	if (size >= 44 &&
		data[0] == 'D' && data[1] == 'K' && data[2] == 'I' && data[3] == 'F' &&
		data[8] == 'A' && data[9] == 'V' && data[10] == '0' && data[11] == '1') {
		const oa::U32 headerSize = static_cast<oa::U32>(data[6]) | (static_cast<oa::U32>(data[7]) << 8u);
		if (headerSize < 32 || static_cast<oa::Usize>(headerSize) + 12u > size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 IVF header size");
		}
		const oa::Usize frameHeaderOffset = headerSize;
		const oa::U32 frameSize = readLe32(data + frameHeaderOffset);
		if (frameSize == 0 || frameHeaderOffset + 12u + frameSize > size) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 IVF frame size");
		}
		outFrame.offset = frameHeaderOffset + 12u;
		outFrame.size = frameSize;
		outFrame.timestamp = readLe64(data + frameHeaderOffset + 4u);
		return oa::Status::ok();
	}

	// Raw access unit (OBU sequence for one picture, as delivered by MP4/WebM demuxer etc.)
	outFrame.offset = 0;
	outFrame.size = size;
	outFrame.timestamp = 0;
	return oa::Status::ok();
}

static oa::Status findAv1Obus(const oa::Span<const oa::U8>& inFrame, oa::Vec<oa::Av1Obu>& outObus)
{
	outObus.clear();
	const oa::U8* data = inFrame.data();
	const oa::Usize size = inFrame.size();
	oa::Usize offset = 0;
	while (offset < size) {
		const oa::Usize headerOffset = offset;
		const oa::U8 header = data[offset++];
		if ((header & 0x80u) != 0 || (header & 0x01u) != 0) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 OBU header");
		}
		const oa::Av1ObuType type = static_cast<oa::Av1ObuType>((header >> 3u) & 0x0fu);
		const bool hasExtension = (header & 0x04u) != 0;
		const bool hasSize = (header & 0x02u) != 0;
		if (hasExtension) {
			if (offset >= size) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "truncated AV1 OBU extension header");
			}
			++offset;
		}
		oa::U64 payloadSize = 0;
		if (hasSize) {
			if (!readLeb128(data, size, offset, payloadSize)) {
				return oa::Status::error(oa::StatusCode::InvalidArgument, "Invalid AV1 OBU payload size");
			}
		} else {
			payloadSize = size - offset;
		}
		if (payloadSize > size - offset) {
			return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 OBU payload exceeds frame data");
		}
		outObus.pushBack({
			type,
			headerOffset,
			offset - headerOffset,
			offset,
			static_cast<oa::Usize>(payloadSize)});
		offset += static_cast<oa::Usize>(payloadSize);
	}
	if (outObus.empty()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "AV1 frame has no OBUs");
	}
	return oa::Status::ok();
}

// ============================================================================
// parser Implementation
// ============================================================================

oa::Status oa::VcpAv1::parseSps(const oa::Span<const oa::U8>& inObu)
{
	(void)inObu;
	return oa::Status::error(oa::StatusCode::Unimplemented, "AV1 sequence header parsing via parseSps is not implemented; use parseAccessUnit");
}

oa::Status oa::VcpAv1::parsePps(const oa::Span<const oa::U8>& inObu)
{
	(void)inObu;
	return oa::Status::error(oa::StatusCode::Unimplemented, "AV1 frame header parsing via ParsePps is not implemented; use parseAccessUnit");
}

void oa::VcpAv1::clearParameterSets()
{
	cachedSequenceHeader_ = {};
	hasCachedSequenceHeader_ = false;
	cachedRefOrderHints_.fill(0);
	cachedRefValid_.fill(false);
}

oa::Status oa::VcpAv1::inspectAccessUnit(
	const oa::Span<const oa::U8>& inBitstream,
	oa::Av1AccessUnitInfo& outInfo) const
{
	outInfo = {};
	oa::Av1IvfFrame frame = {};
	OA_RETURN_IF_ERROR(extractAv1FramePayload(inBitstream, frame));
	oa::Vec<oa::Av1Obu> obus;
	OA_RETURN_IF_ERROR(findAv1Obus(
		oa::Span<const oa::U8>(inBitstream.data() + frame.offset, frame.size), obus));
	for (const oa::Av1Obu& obu : obus) {
		switch (obu.type) {
		case oa::Av1ObuType::SequenceHeader: ++outInfo.sequenceHeaderCount; break;
		case oa::Av1ObuType::Frame: ++outInfo.frameCount; break;
		case oa::Av1ObuType::FrameHeader: ++outInfo.frameHeaderCount; break;
		case oa::Av1ObuType::TileGroup: ++outInfo.tileGroupCount; break;
		default: break;
		}
	}
	return oa::Status::ok();
}

oa::Status oa::VcpAv1::parseAccessUnit(const oa::Span<const oa::U8>& inBitstream, oa::Av1PictureDesc& outDesc)
{
	outDesc = {};
	oa::Vec<oa::Av1PictureDesc> pictures;
	OA_RETURN_IF_ERROR(parseAccessUnitPictures(inBitstream, pictures));
	if (!pictures.empty()) outDesc = pictures[0];
	return oa::Status::ok();
}

oa::Status oa::VcpAv1::parseAccessUnitPictures(
	const oa::Span<const oa::U8>& inBitstream,
	oa::Vec<oa::Av1PictureDesc>& outDescs)
{
	outDescs.clear();

	oa::Av1IvfFrame frame = {};
	OA_RETURN_IF_ERROR(extractAv1FramePayload(inBitstream, frame));

	oa::Vec<oa::Av1Obu> obus;
	OA_RETURN_IF_ERROR(findAv1Obus(
		oa::Span<const oa::U8>(inBitstream.data() + frame.offset, frame.size),
		obus));
	oa::Span<const oa::U8> av1FrameData(inBitstream.data() + frame.offset, frame.size);

	auto parsePicture = [&](const oa::Av1Obu& inHeader, oa::Usize inFirstTile,
		oa::Usize inTileEnd, bool inCombinedFrame) -> oa::Status {
		if (!hasCachedSequenceHeader_) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"AV1 picture has no cached sequence header");
		}
		oa::Av1PictureDesc desc = {};
		desc.frame = frame;
		desc.sequenceHeader = cachedSequenceHeader_;
		OA_RETURN_IF_ERROR(parseAv1FrameHeader(inHeader, av1FrameData,
			desc.sequenceHeader, cachedRefOrderHints_, cachedRefValid_, desc.frameHeader));
		desc.showExistingFrame = desc.frameHeader.showExistingFrame;
		desc.frameToShowMapIdx = desc.frameHeader.frameToShowMapIdx;
		desc.frameHeaderOffset = inHeader.payloadOffset;
		desc.decodeObuOffset = inHeader.headerOffset;
		if (desc.showExistingFrame) {
			desc.decodeObuSize = inHeader.headerSize + inHeader.payloadSize;
			outDescs.pushBack(desc);
			return oa::Status::ok();
		}
		if (inFirstTile >= inTileEnd || inFirstTile >= obus.size()) {
			return oa::Status::error(oa::StatusCode::Unavailable,
				"AV1 frame header has no tile payload");
		}
		const oa::Av1Obu& lastTile = obus[inTileEnd - 1u];
		desc.decodeObuSize = lastTile.headerOffset + lastTile.headerSize
			+ lastTile.payloadSize - inHeader.headerOffset;
		for (oa::Usize tileIndex = inFirstTile; tileIndex < inTileEnd; ++tileIndex) {
			oa::Av1TileGroupInfo group = {};
			OA_RETURN_IF_ERROR(parseAv1TileGroup(obus[tileIndex], av1FrameData,
				desc.frameHeader, inCombinedFrame, group));
			for (oa::U32 offset : group.tileOffsets) desc.tileOffsets.pushBack(offset);
			for (oa::U32 size : group.tileSizes) desc.tileSizes.pushBack(size);
		}
		desc.tileGroup.tileOffsets = desc.tileOffsets;
		desc.tileGroup.tileSizes = desc.tileSizes;
		desc.hasPicture = true;
		for (oa::U32 mask = desc.frameHeader.refreshFrameFlags, ref = 0;
			mask != 0u && ref < STD_VIDEO_AV1_NUM_REF_FRAMES; mask >>= 1u, ++ref) {
			if ((mask & 1u) != 0u) {
				cachedRefOrderHints_[ref] = static_cast<oa::U8>(desc.frameHeader.orderHint);
				cachedRefValid_[ref] = true;
			}
		}
		outDescs.pushBack(desc);
		return oa::Status::ok();
	};

	for (oa::Usize i = 0; i < obus.size();) {
		const oa::Av1Obu& obu = obus[i];
		if (obu.type == oa::Av1ObuType::SequenceHeader) {
			OA_RETURN_IF_ERROR(parseAv1SequenceHeader(obu, av1FrameData,
				cachedSequenceHeader_));
			hasCachedSequenceHeader_ = true;
			++i;
			continue;
		}
		if (obu.type == oa::Av1ObuType::Frame) {
			OA_RETURN_IF_ERROR(parsePicture(obu, i, i + 1u, true));
			++i;
			continue;
		}
		if (obu.type == oa::Av1ObuType::FrameHeader) {
			oa::Usize end = i + 1u;
			while (end < obus.size() && obus[end].type == oa::Av1ObuType::TileGroup) ++end;
			OA_RETURN_IF_ERROR(parsePicture(obu, i + 1u, end, false));
			i = end;
			continue;
		}
		++i;
	}
	if (!hasCachedSequenceHeader_) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"AV1 access unit is missing a sequence header and none is cached");
	}
	return oa::Status::ok();
}

namespace {
struct Av1CodecRegistrar {
	Av1CodecRegistrar() {
		auto parser = oa::makeUnique<oa::VcpAv1>();
		oa::VideoCodecRegistry::getInstance().registerParser(
			oa::VideoCodec::AV1,
			oa::move(parser));
	}
};
static Av1CodecRegistrar g_Av1Registrar __attribute__((used));
} // namespace
