// OA Vision — AV1 Codec Parser Implementation
// Extracts and converts AV1 OBUs (Open Bitstream Units)

#include "VcpAv1.h"
#include "CodecRegistry.h"
#include <Oa/Vision/VideoDecoder.h>

class OaAv1BitReader {
public:
	OaAv1BitReader(const OaU8* InData, OaUsize InSize)
		: Data_(InData), Size_(InSize)
	{}

	bool ReadBits(OaU32 InCount, OaU32& OutValue)
	{
		if (InCount > 32 || BitOffset_ + InCount > Size_ * 8u) {
			return false;
		}
		OaU32 value = 0;
		for (OaU32 i = 0; i < InCount; ++i) {
			const OaUsize bitIndex = BitOffset_++;
			const OaU8 byte = Data_[bitIndex >> 3u];
			value = (value << 1u) | ((byte >> (7u - (bitIndex & 7u))) & 1u);
		}
		OutValue = value;
		return true;
	}

	bool ReadBit(bool& OutValue)
	{
		OaU32 value = 0;
		if (!ReadBits(1, value)) {
			return false;
		}
		OutValue = value != 0;
		return true;
	}

	bool SkipBits(OaU32 InCount)
	{
		OaU32 ignored = 0;
		while (InCount > 0) {
			const OaU32 chunk = InCount > 32 ? 32 : InCount;
			if (!ReadBits(chunk, ignored)) {
				return false;
			}
			InCount -= chunk;
		}
		return true;
	}

	bool ReadUvlc(OaU32& OutValue)
	{
		OaU32 leadingZeros = 0;
		bool bit = false;
		while (leadingZeros < 32u) {
			if (!ReadBit(bit)) return false;
			if (bit) break;
			++leadingZeros;
		}
		if (leadingZeros == 32u) {
			OutValue = 0xffffffffu;
			return true;
		}
		OaU32 suffix = 0;
		if (leadingZeros > 0u && !ReadBits(leadingZeros, suffix)) return false;
		OutValue = ((1u << leadingZeros) - 1u) + suffix;
		return true;
	}

	void ByteAlign() {
		BitOffset_ = OaAlignUp(BitOffset_, static_cast<OaUsize>(8));
	}

	OaUsize ByteOffset() const {
		return (BitOffset_ + 7u) >> 3u;
	}

	OaUsize DebugBitPos() const { return BitOffset_; }

private:
	const OaU8* Data_ = nullptr;
	OaUsize Size_ = 0;
	OaUsize BitOffset_ = 0;
};

static OaU32 ReadLe32(const OaU8* InData) {
	return static_cast<OaU32>(InData[0]) |
		(static_cast<OaU32>(InData[1]) << 8u) |
		(static_cast<OaU32>(InData[2]) << 16u) |
		(static_cast<OaU32>(InData[3]) << 24u);
}

static OaU64 ReadLe64(const OaU8* InData) {
	OaU64 value = 0;
	for (OaU32 i = 0; i < 8; ++i) {
		value |= static_cast<OaU64>(InData[i]) << (i * 8u);
	}
	return value;
}

static bool ReadLeb128(const OaU8* InData, OaUsize InSize, OaUsize& InOutOffset, OaU64& OutValue)
{
	OutValue = 0;
	for (OaU32 i = 0; i < 8 && InOutOffset < InSize; ++i) {
		const OaU8 byte = InData[InOutOffset++];
		OutValue |= static_cast<OaU64>(byte & 0x7fu) << (i * 7u);
		if ((byte & 0x80u) == 0) {
			return true;
		}
	}
	return false;
}

static OaU32 Av1FloorLog2(OaU32 InValue)
{
	OaU32 result = 0;
	while (InValue > 1) {
		InValue >>= 1u;
		++result;
	}
	return result;
}

static OaStatus ReadAv1DeltaQ(OaAv1BitReader& InReader, OaI32& OutDelta)
{
	bool deltaCoded = false;
	if (!InReader.ReadBit(deltaCoded)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 quantization delta flag");
	}
	OutDelta = 0;
	if (!deltaCoded) {
		return OaStatus::Ok();
	}
	OaU32 value = 0;
	if (!InReader.ReadBits(7, value)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 quantization delta");
	}
	OutDelta = (value & 1u) != 0
		? -static_cast<OaI32>((value + 1u) >> 1u)
		: static_cast<OaI32>(value >> 1u);
	return OaStatus::Ok();
}

static OaStatus ReadAv1SignedFeatureData(OaAv1BitReader& InReader, OaU32 InBits, OaI16& OutValue)
{
	if (InBits == 0) {
		OutValue = 0;
		return OaStatus::Ok();
	}
	OaU32 value = 0;
	bool sign = false;
	if (!InReader.ReadBits(InBits, value) || !InReader.ReadBit(sign)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 segmentation feature data");
	}
	OutValue = static_cast<OaI16>(sign ? -static_cast<OaI32>(value) : static_cast<OaI32>(value));
	return OaStatus::Ok();
}

static OaStatus ReadAv1InverseSignedLiteral(OaAv1BitReader& InReader, OaU32 InBits, OaI8& OutValue)
{
	OaU32 value = 0;
	if (!InReader.ReadBits(InBits, value)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 signed literal");
	}
	const OaI32 decoded = (value & 1u) != 0
		? -static_cast<OaI32>((value >> 1u) + 1u)
		: static_cast<OaI32>(value >> 1u);
	OutValue = static_cast<OaI8>(decoded);
	return OaStatus::Ok();
}

static OaStatus ReadAv1RestorationType(OaAv1BitReader& InReader, StdVideoAV1FrameRestorationType& OutType)
{
	bool useWiener = false;
	if (!InReader.ReadBit(useWiener)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 restoration type");
	}
	bool useSgrproj = false;
	if (!InReader.ReadBit(useSgrproj)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 restoration type");
	}
	if (useWiener && useSgrproj) {
		OutType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE;
	} else if (useWiener) {
		OutType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER;
	} else if (useSgrproj) {
		OutType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ;
	} else {
		OutType = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE;
	}
	return OaStatus::Ok();
}

static OaStatus ParseAv1RestorationHeader(
	OaAv1BitReader& InReader,
	bool InUse128x128Superblock,
	OaAv1FrameHeaderInfo& InOutInfo)
{
	for (OaU32 plane = 0; plane < STD_VIDEO_AV1_MAX_NUM_PLANES; ++plane) {
		OA_RETURN_IF_ERROR(ReadAv1RestorationType(InReader, InOutInfo.RestorationTypes[plane]));
	}

	InOutInfo.UsesLr = false;
	InOutInfo.UsesChromaLr = false;
	for (OaU32 plane = 0; plane < STD_VIDEO_AV1_MAX_NUM_PLANES; ++plane) {
		if (InOutInfo.RestorationTypes[plane] != STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE) {
			InOutInfo.UsesLr = true;
			InOutInfo.UsesChromaLr = InOutInfo.UsesChromaLr || plane > 0;
		}
	}
	if (!InOutInfo.UsesLr) {
		return OaStatus::Ok();
	}

	bool flag = false;
	OaU32 lrUnitShift = 0;
	if (InUse128x128Superblock) {
		if (!InReader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 restoration unit shift");
		}
		lrUnitShift = flag ? 2u : 1u;
	} else {
		if (!InReader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 restoration unit shift");
		}
		lrUnitShift = flag ? 1u : 0u;
		if (lrUnitShift > 0) {
			if (!InReader.ReadBit(flag)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 restoration unit extra shift");
			}
			lrUnitShift += flag ? 1u : 0u;
		}
	}
	// Vulkan std-video / NVIDIA vk_video_samples store log2(restoration_unit_size)-5,
	// which for AV1 equals 1 + lr_unit_shift (not the pixel width).
	const OaU16 lumaRestorationSize = static_cast<OaU16>(1u + lrUnitShift);
	InOutInfo.RestorationSizes[0] = lumaRestorationSize;

	OaU32 lrUvShift = 0;
	if (InOutInfo.UsesChromaLr) {
		if (!InReader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 chroma restoration unit shift");
		}
		lrUvShift = flag ? 1u : 0u;
	}
	InOutInfo.RestorationSizes[1] = static_cast<OaU16>(lumaRestorationSize - lrUvShift);
	InOutInfo.RestorationSizes[2] = InOutInfo.RestorationSizes[1];
	return OaStatus::Ok();
}

static OaU32 ReadLeBytes(const OaU8* InData, OaU32 InByteCount)
{
	OaU32 value = 0;
	for (OaU32 i = 0; i < InByteCount; ++i) {
		value |= static_cast<OaU32>(InData[i]) << (i * 8u);
	}
	return value;
}

static OaStatus ParseAv1ColorConfig(
	OaAv1BitReader& InReader,
	StdVideoAV1Profile InProfile,
	StdVideoAV1ColorConfig& OutConfig)
{
	OaU32 value = 0;
	bool flag = false;
	bool highBitDepth = false;
	if (!InReader.ReadBit(highBitDepth)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 high bit depth flag");
	}
	if (InProfile == STD_VIDEO_AV1_PROFILE_PROFESSIONAL && highBitDepth) {
		bool twelveBit = false;
		if (!InReader.ReadBit(twelveBit)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 twelve bit flag");
		}
		OutConfig.BitDepth = twelveBit ? 12u : 10u;
	} else {
		OutConfig.BitDepth = highBitDepth ? 10u : 8u;
	}

	bool monoChrome = false;
	if (InProfile != STD_VIDEO_AV1_PROFILE_HIGH) {
		if (!InReader.ReadBit(monoChrome)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 mono chrome flag");
		}
	}
	OutConfig.flags.mono_chrome = monoChrome ? 1u : 0u;

	bool colorDescriptionPresent = false;
	if (!InReader.ReadBit(colorDescriptionPresent)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 color description flag");
	}
	OutConfig.flags.color_description_present_flag = colorDescriptionPresent ? 1u : 0u;
	if (colorDescriptionPresent) {
		if (!InReader.ReadBits(8, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 color primaries");
		}
		OutConfig.color_primaries = static_cast<StdVideoAV1ColorPrimaries>(value);
		if (!InReader.ReadBits(8, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 transfer characteristics");
		}
		OutConfig.transfer_characteristics = static_cast<StdVideoAV1TransferCharacteristics>(value);
		if (!InReader.ReadBits(8, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 matrix coefficients");
		}
		OutConfig.matrix_coefficients = static_cast<StdVideoAV1MatrixCoefficients>(value);
	} else {
		OutConfig.color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED;
		OutConfig.transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED;
		OutConfig.matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED;
	}

	if (monoChrome) {
		if (!InReader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 mono chrome color range");
		}
		OutConfig.flags.color_range = flag ? 1u : 0u;
		OutConfig.subsampling_x = 1u;
		OutConfig.subsampling_y = 1u;
		OutConfig.flags.separate_uv_delta_q = 0u;
	} else {
		if (OutConfig.color_primaries == STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709 &&
			OutConfig.transfer_characteristics == STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SRGB &&
			OutConfig.matrix_coefficients == STD_VIDEO_AV1_MATRIX_COEFFICIENTS_IDENTITY) {
			OutConfig.subsampling_x = 0u;
			OutConfig.subsampling_y = 0u;
			OutConfig.flags.color_range = 1u;
		} else {
			bool colorRange = false;
			if (!InReader.ReadBit(colorRange)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 color range");
			}
			OutConfig.flags.color_range = colorRange ? 1u : 0u;
			if (InProfile == STD_VIDEO_AV1_PROFILE_MAIN) {
				OutConfig.subsampling_x = 1u;
				OutConfig.subsampling_y = 1u;
			} else if (InProfile == STD_VIDEO_AV1_PROFILE_HIGH) {
				OutConfig.subsampling_x = 0u;
				OutConfig.subsampling_y = 0u;
			} else if (OutConfig.BitDepth == 12u) {
				bool subX = false;
				if (!InReader.ReadBit(subX)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 chroma subsampling X");
				}
				OutConfig.subsampling_x = subX ? 1u : 0u;
				if (subX) {
					bool subY = false;
					if (!InReader.ReadBit(subY)) {
						return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 chroma subsampling Y");
					}
					OutConfig.subsampling_y = subY ? 1u : 0u;
				} else {
					OutConfig.subsampling_y = 0u;
				}
			} else {
				OutConfig.subsampling_x = 1u;
				OutConfig.subsampling_y = 0u;
			}
		}
		if (OutConfig.subsampling_x != 0u && OutConfig.subsampling_y != 0u) {
			if (!InReader.ReadBits(2, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 chroma sample position");
			}
			OutConfig.chroma_sample_position = static_cast<StdVideoAV1ChromaSamplePosition>(value);
		}
		if (!InReader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 separate uv delta q");
		}
		OutConfig.flags.separate_uv_delta_q = flag ? 1u : 0u;
	}
	return OaStatus::Ok();
}

static OaStatus ParseAv1SequenceHeader(const OaAv1Obu& InObu, const OaSpan<const OaU8>& InFrame, OaAv1SequenceHeaderInfo& OutInfo)
{
	if (InObu.PayloadSize == 0 || InObu.PayloadOffset + InObu.PayloadSize > InFrame.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 sequence header payload");
	}

	OaAv1BitReader reader(InFrame.Data() + InObu.PayloadOffset, InObu.PayloadSize);
	OaU32 value = 0;
	bool flag = false;
	if (!reader.ReadBits(3, value) || !reader.ReadBit(OutInfo.StillPicture) || !reader.ReadBit(OutInfo.ReducedStillPictureHeader)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 sequence header");
	}
	OutInfo.SeqProfile = static_cast<StdVideoAV1Profile>(value);

	bool decoderModelInfoPresent = false;
	OaU32 bufferDelayLengthMinus1 = 0;
	if (OutInfo.ReducedStillPictureHeader) {
		value = 0;
	} else {
		if (!reader.ReadBit(OutInfo.TimingInfoPresent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 sequence timing flag");
		}
		if (OutInfo.TimingInfoPresent) {
			if (!reader.ReadBits(32, OutInfo.TimingInfo.num_units_in_display_tick)
				|| !reader.ReadBits(32, OutInfo.TimingInfo.time_scale)
				|| !reader.ReadBit(flag)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 timing info");
			}
			OutInfo.TimingInfo.flags.equal_picture_interval = flag ? 1u : 0u;
			if (flag && !reader.ReadUvlc(OutInfo.TimingInfo.num_ticks_per_picture_minus_1)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument,
					"Truncated AV1 equal picture interval");
			}
			if (!reader.ReadBit(decoderModelInfoPresent)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument,
					"Truncated AV1 decoder model flag");
			}
			if (decoderModelInfoPresent
				&& (!reader.ReadBits(5, bufferDelayLengthMinus1)
					|| !reader.SkipBits(32 + 5 + 5))) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 decoder model info");
			}
		}
		if (!reader.ReadBit(OutInfo.InitialDisplayDelayPresent) || !reader.ReadBits(5, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 operating point header");
		}
	}

	const OaU32 operatingPoints = value + 1u;
	for (OaU32 i = 0; i < operatingPoints; ++i) {
		if (!reader.SkipBits(12) || !reader.ReadBits(5, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 operating point");
		}
		if (value > 7 && !reader.SkipBits(1)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 operating point tier");
		}
		if (decoderModelInfoPresent) {
			if (!reader.ReadBit(flag)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 decoder model flag");
			}
			const OaU32 delayBits = bufferDelayLengthMinus1 + 1u;
			if (flag && !reader.SkipBits(delayBits + delayBits + 1u)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 operating decoder model");
			}
		}
		if (OutInfo.InitialDisplayDelayPresent) {
			if (!reader.ReadBit(flag)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 initial display delay flag");
			}
			if (flag && !reader.SkipBits(4)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 initial display delay");
			}
		}
	}

	if (!reader.ReadBits(4, OutInfo.FrameWidthBitsMinus1) ||
		!reader.ReadBits(4, OutInfo.FrameHeightBitsMinus1) ||
		!reader.ReadBits(OutInfo.FrameWidthBitsMinus1 + 1u, OutInfo.MaxFrameWidthMinus1) ||
		!reader.ReadBits(OutInfo.FrameHeightBitsMinus1 + 1u, OutInfo.MaxFrameHeightMinus1)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame size bits");
	}

	if (OutInfo.ReducedStillPictureHeader) {
		OutInfo.FrameIdNumbersPresent = false;
	} else {
		if (!reader.ReadBit(OutInfo.FrameIdNumbersPresent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame id flag");
		}
		if (OutInfo.FrameIdNumbersPresent) {
			OaU32 deltaLen = 0;
			OaU32 addLen = 0;
			if (!reader.ReadBits(4, deltaLen) || !reader.ReadBits(3, addLen)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame id fields");
			}
			OutInfo.DeltaFrameIdLengthMinus2 = static_cast<OaU8>(deltaLen);
			OutInfo.AdditionalFrameIdLengthMinus1 = static_cast<OaU8>(addLen);
			const OaU32 frameIdLength = 3u + (deltaLen + 2u) + (addLen + 1u);
			if (frameIdLength > 16u) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame id length");
			}
		}
	}

	if (!reader.ReadBit(OutInfo.Use128x128Superblock) ||
		!reader.ReadBit(OutInfo.EnableFilterIntra) ||
		!reader.ReadBit(OutInfo.EnableIntraEdgeFilter)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 coding tool flags");
	}
	if (OutInfo.ReducedStillPictureHeader) {
		OutInfo.EnableInterIntraCompound = false;
		OutInfo.EnableMaskedCompound = false;
		OutInfo.EnableWarpedMotion = false;
		OutInfo.EnableDualFilter = false;
		OutInfo.EnableOrderHint = false;
		OutInfo.EnableJntComp = false;
		OutInfo.EnableRefFrameMvs = false;
		OutInfo.SeqForceScreenContentTools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS;
		OutInfo.SeqForceIntegerMv = STD_VIDEO_AV1_SELECT_INTEGER_MV;
		OutInfo.OrderHintBits = 0;
	} else {
		if (!reader.ReadBit(OutInfo.EnableInterIntraCompound) ||
			!reader.ReadBit(OutInfo.EnableMaskedCompound) ||
			!reader.ReadBit(OutInfo.EnableWarpedMotion) ||
			!reader.ReadBit(OutInfo.EnableDualFilter) ||
			!reader.ReadBit(OutInfo.EnableOrderHint)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 inter tool flags");
		}
		if (OutInfo.EnableOrderHint) {
			if (!reader.ReadBit(OutInfo.EnableJntComp) || !reader.ReadBit(OutInfo.EnableRefFrameMvs)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 order-hint tool flags");
			}
		} else {
			OutInfo.EnableJntComp = false;
			OutInfo.EnableRefFrameMvs = false;
		}
		bool choose = false;
		if (!reader.ReadBit(choose)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 screen-content tool flag");
		}
		if (choose) {
			OutInfo.SeqForceScreenContentTools = 2;
		} else if (reader.ReadBit(flag)) {
			OutInfo.SeqForceScreenContentTools = flag ? 1 : 0;
		} else {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 screen-content tool value");
		}
		if (OutInfo.SeqForceScreenContentTools > 0) {
			if (!reader.ReadBit(choose)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 integer-mv tool flag");
			}
			if (choose) {
				OutInfo.SeqForceIntegerMv = 2;
			} else if (reader.ReadBit(flag)) {
				OutInfo.SeqForceIntegerMv = flag ? 1 : 0;
			} else {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 integer-mv tool value");
			}
		}
		if (OutInfo.EnableOrderHint && !reader.ReadBits(3, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 order hint bits");
		}
		OutInfo.OrderHintBits = OutInfo.EnableOrderHint ? value + 1u : 0u;
	}
	if (!reader.ReadBit(OutInfo.EnableSuperres) || !reader.ReadBit(OutInfo.EnableCdef) || !reader.ReadBit(OutInfo.EnableRestoration)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 post-filter flags");
	}
	OA_RETURN_IF_ERROR(ParseAv1ColorConfig(reader, OutInfo.SeqProfile, OutInfo.ColorConfig));
	if (!reader.ReadBit(OutInfo.FilmGrainParamsPresent)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 film grain present flag");
	}
	return OaStatus::Ok();
}

static OaStatus ParseAv1FrameHeader(
	const OaAv1Obu& InObu,
	const OaSpan<const OaU8>& InFrame,
	const OaAv1SequenceHeaderInfo& InSeq,
	const OaArray<OaU8, STD_VIDEO_AV1_NUM_REF_FRAMES>& InRefOrderHints,
	const OaArray<bool, STD_VIDEO_AV1_NUM_REF_FRAMES>& InRefValid,
	OaAv1FrameHeaderInfo& OutInfo)
{
	if (InObu.PayloadSize == 0 || InObu.PayloadOffset + InObu.PayloadSize > InFrame.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 frame payload");
	}

	OaAv1BitReader reader(InFrame.Data() + InObu.PayloadOffset, InObu.PayloadSize);
	OaArray<OaU8, STD_VIDEO_AV1_NUM_REF_FRAMES> logicalOrderHints = {};
	for (OaU32 ref = 0; ref < STD_VIDEO_AV1_NUM_REF_FRAMES; ++ref) {
		logicalOrderHints[ref] = InRefValid[ref] ? InRefOrderHints[ref] : 0;
	}
	bool flag = false;
	OaU32 value = 0;
	if (!InSeq.ReducedStillPictureHeader) {
		if (!reader.ReadBit(flag)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 show-existing-frame flag");
		}
		if (flag) {
			// show_existing_frame: this is a display-only operation. No new compressed
			// picture is decoded; we simply point at a previous reference picture
			// (identified by frame_to_show_map_idx into the 8-ref array).
			// The decoder layer will resolve it to a DPB slot and return the prior
			// frame's resources without submitting vkCmdDecodeVideoKHR work.
			OaU32 idx = 0;
			if (!reader.ReadBits(3, idx)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame_to_show_map_idx");
			}
			OutInfo.ShowExistingFrame = true;
			OutInfo.FrameToShowMapIdx = static_cast<OaU8>(idx);
			OutInfo.HeaderSize = reader.ByteOffset();
			return OaStatus::Ok();
		}
	}

	if (!reader.ReadBits(2, value)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame type");
	}
	OutInfo.FrameType = static_cast<StdVideoAV1FrameType>(value);
	OutInfo.IsKeyFrame = value == 0;
	if (!reader.ReadBit(OutInfo.ShowFrame)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 show-frame flag");
	}
	if (!OutInfo.ShowFrame && !reader.ReadBit(OutInfo.ShowableFrame)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 showable-frame flag");
	}

	// AV1 §5.9.2 uncompressed_header(). The ordering below is exact: every field
	// must consume precisely the bits the spec says before tile_info(), or all
	// downstream syntax (tiles, quant, ...) misaligns. The previous code only
	// handled the intra layout and skipped error_resilient_mode, force_integer_mv,
	// frame_size_with_refs, the interpolation filter and motion fields — which
	// drifted every inter frame header by ~tens of bits.
	const bool frameIsIntra = (OutInfo.FrameType == STD_VIDEO_AV1_FRAME_TYPE_KEY)
		|| (OutInfo.FrameType == STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY);
	const bool frameTypeSwitch = (OutInfo.FrameType == STD_VIDEO_AV1_FRAME_TYPE_SWITCH);
	const bool keyShown = (OutInfo.FrameType == STD_VIDEO_AV1_FRAME_TYPE_KEY) && OutInfo.ShowFrame;

	bool errorResilientMode = false;
	if (frameTypeSwitch || keyShown) {
		errorResilientMode = true;
	} else if (!reader.ReadBit(errorResilientMode)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 error-resilient flag");
	}
	OutInfo.ErrorResilientMode = errorResilientMode || frameTypeSwitch || keyShown;

	if (!reader.ReadBit(OutInfo.DisableCdfUpdate)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 disable-cdf-update flag");
	}
	if (InSeq.SeqForceScreenContentTools == 2) {
		if (!reader.ReadBit(OutInfo.AllowScreenContentTools)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 screen-content frame flag");
		}
	} else {
		OutInfo.AllowScreenContentTools = InSeq.SeqForceScreenContentTools != 0;
	}
	bool forceIntegerMv = false;
	if (OutInfo.AllowScreenContentTools) {
		if (InSeq.SeqForceIntegerMv == 2) {
			if (!reader.ReadBit(forceIntegerMv)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 force-integer-mv flag");
			}
		} else {
			forceIntegerMv = InSeq.SeqForceIntegerMv != 0;
		}
	}
	if (frameIsIntra) {
		forceIntegerMv = true;
	}
	OutInfo.ForceIntegerMv = forceIntegerMv;

	// frame_id_numbers_present is assumed 0 (consistent with the sequence-header
	// parse, which never enables it); so no current_frame_id / delta_frame_id bits.

	bool frameSizeOverride = false;
	if (frameTypeSwitch) {
		frameSizeOverride = true;
	} else if (InSeq.ReducedStillPictureHeader) {
		frameSizeOverride = false;
	} else if (!reader.ReadBit(frameSizeOverride)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame-size-override flag");
	}
	OutInfo.FrameSizeOverrideFlag = frameSizeOverride;

	if (InSeq.OrderHintBits > 0) {
		if (!reader.ReadBits(InSeq.OrderHintBits, OutInfo.OrderHint)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 order hint");
		}
	}

	if (frameIsIntra || errorResilientMode) {
		OutInfo.PrimaryRefFrame = STD_VIDEO_AV1_PRIMARY_REF_NONE;
	} else {
		if (!reader.ReadBits(3, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 primary_ref_frame");
		}
		OutInfo.PrimaryRefFrame = value;
	}

	// decoder_model_info_present assumed 0 → no buffer_removal_time fields.

	if (frameTypeSwitch || keyShown) {
		OutInfo.RefreshFrameFlags = 0xffu;
	} else {
		if (!reader.ReadBits(8, OutInfo.RefreshFrameFlags)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 refresh_frame_flags");
		}
	}

	// ref_order_hint[] is only coded for error-resilient frames with order hints.
	if ((!frameIsIntra || OutInfo.RefreshFrameFlags != 0xffu) && errorResilientMode && InSeq.EnableOrderHint) {
		for (OaU32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
			if (!reader.ReadBits(InSeq.OrderHintBits, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 ref order hint");
			}
			logicalOrderHints[i] = static_cast<OaU8>(value);
		}
	}

	// Shared size readers (AV1 §5.9.5 frame_size / §5.9.6 render_size /
	// §5.9.8 superres_params). superres uses SUPERRES_DENOM_BITS = 3; render
	// size uses two fixed 16-bit fields, not the frame-size bit widths.
	auto readSuperres = [&]() -> OaStatus {
		OutInfo.UseSuperres = false;
		OutInfo.CodedDenom = 0;
		if (InSeq.EnableSuperres) {
			if (!reader.ReadBit(OutInfo.UseSuperres)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 superres flag");
			}
			if (OutInfo.UseSuperres) {
				OaU32 denom = 0;
				if (!reader.ReadBits(3, denom)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 superres denominator");
				}
				OutInfo.CodedDenom = static_cast<OaU8>(denom);
			}
		}
		return OaStatus::Ok();
	};
	auto readFrameSize = [&]() -> OaStatus {
		if (frameSizeOverride) {
			if (!reader.SkipBits((InSeq.FrameWidthBitsMinus1 + 1u) + (InSeq.FrameHeightBitsMinus1 + 1u))) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame size");
			}
		}
		return readSuperres();
	};
	auto readRenderSize = [&]() -> OaStatus {
		if (!reader.ReadBit(OutInfo.RenderAndFrameSizeDifferent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 render-size flag");
		}
		if (OutInfo.RenderAndFrameSizeDifferent && !reader.SkipBits(16 + 16)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 render size");
		}
		return OaStatus::Ok();
	};

	if (frameIsIntra) {
		OA_RETURN_IF_ERROR(readFrameSize());
		OA_RETURN_IF_ERROR(readRenderSize());
		if (OutInfo.AllowScreenContentTools) {
			// allow_intrabc (present when UpscaledWidth == FrameWidth; true when
			// superres is inactive, the common case for non-screen content).
			if (!reader.ReadBit(OutInfo.AllowIntraBc)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 allow_intrabc");
			}
		}
	} else {
		if (InSeq.EnableOrderHint) {
			if (!reader.ReadBit(OutInfo.FrameRefsShortSignaling)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame_refs_short_signaling");
			}
			if (OutInfo.FrameRefsShortSignaling) {
				// last_frame_idx f(3), gold_frame_idx f(3); set_frame_refs() derives the rest (no bits).
				if (!reader.SkipBits(3 + 3)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 short ref signaling");
				}
			}
		}
		for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
			if (!OutInfo.FrameRefsShortSignaling) {
				if (!reader.ReadBits(3, value)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 ref_frame_idx");
				}
				OutInfo.ReferenceNameSlotIndices[i] = static_cast<OaI32>(value);
			}
			// frame_id_numbers_present assumed 0 → no delta_frame_id_minus_1.
		}
		// StdVideoDecodeAV1PictureInfo::OrderHints is indexed by reference
		// *name* (INTRA_FRAME, LAST_FRAME ... ALTREF_FRAME), not by the eight
		// logical ref-frame-map slots addressed by ref_frame_idx[].
		OutInfo.OrderHints[0] = 0;
		for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
			const OaI32 logical = OutInfo.ReferenceNameSlotIndices[i];
			OutInfo.OrderHints[i + 1u] = logical >= 0
				&& static_cast<OaU32>(logical) < logicalOrderHints.Size()
				? logicalOrderHints[static_cast<OaUsize>(logical)]
				: 0;
		}
		if (frameSizeOverride && !errorResilientMode) {
			// frame_size_with_refs(): a found_ref bit per reference; the first set
			// one adopts that ref's size, otherwise an explicit frame/render size.
			bool foundRef = false;
			for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
				if (!reader.ReadBit(foundRef)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 found_ref");
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
			OutInfo.AllowHighPrecisionMv = false;
		} else if (!reader.ReadBit(OutInfo.AllowHighPrecisionMv)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 allow_high_precision_mv");
		}
		// read_interpolation_filter(): is_filter_switchable f(1); if 0, interpolation_filter f(2).
		bool filterSwitchable = false;
		if (!reader.ReadBit(filterSwitchable)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 is_filter_switchable");
		}
		OutInfo.IsFilterSwitchable = filterSwitchable;
		if (filterSwitchable) {
			OutInfo.InterpolationFilter = STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE;
		} else if (!reader.ReadBits(2, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 interpolation_filter");
		} else {
			OutInfo.InterpolationFilter = static_cast<StdVideoAV1InterpolationFilter>(value);
		}
		if (!reader.ReadBit(OutInfo.IsMotionModeSwitchable)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 is_motion_mode_switchable");
		}
		if (errorResilientMode || !InSeq.EnableRefFrameMvs) {
			OutInfo.UseRefFrameMvs = false;
		} else if (!reader.ReadBit(OutInfo.UseRefFrameMvs)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 use_ref_frame_mvs");
		}
	}

	if (frameIsIntra) {
		OutInfo.UseRefFrameMvs = false;
	}

	// disable_frame_end_update_cdf is the last field before tile_info().
	if (InSeq.ReducedStillPictureHeader || OutInfo.DisableCdfUpdate) {
		OutInfo.DisableFrameEndUpdateCdf = true;
	} else if (!reader.ReadBit(OutInfo.DisableFrameEndUpdateCdf)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 frame-end-cdf flag");
	}

	const OaU32 frameWidth = InSeq.MaxFrameWidthMinus1 + 1u;
	const OaU32 frameHeight = InSeq.MaxFrameHeightMinus1 + 1u;
	const OaU32 miCols = (frameWidth + 7u) >> 3u << 1u;
	const OaU32 miRows = (frameHeight + 7u) >> 3u << 1u;
	const OaU32 sbShift = InSeq.Use128x128Superblock ? 5u : 4u;
	const OaU32 sbCols = (miCols + (1u << sbShift) - 1u) >> sbShift;
	const OaU32 sbRows = (miRows + (1u << sbShift) - 1u) >> sbShift;
	const OaU32 maxTileWidthSb = InSeq.Use128x128Superblock ? 32u : 64u;
	const OaU32 maxTileAreaSb = 4096u >> (2u * (InSeq.Use128x128Superblock ? 1u : 0u));
	const OaU32 minLog2TileCols = Av1FloorLog2((sbCols + maxTileWidthSb - 1u) / maxTileWidthSb);
	OaU32 maxLog2TileCols = Av1FloorLog2(sbCols == 0 ? 1u : sbCols);
	while (maxLog2TileCols > 0 && ((sbRows * (sbCols + (1u << maxLog2TileCols) - 1u)) >> maxLog2TileCols) > maxTileAreaSb) {
		--maxLog2TileCols;
	}
	OaU32 minLog2Tiles = Av1FloorLog2((sbRows * sbCols + maxTileAreaSb - 1u) / maxTileAreaSb);

	if (!reader.ReadBit(flag)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile spacing flag");
	}
	if (!flag) {
		return OaStatus::Error(OaStatusCode::Unavailable, "Only uniform AV1 tile spacing is implemented");
	}

	OaU32 tileColsLog2 = minLog2TileCols;
	while (tileColsLog2 < maxLog2TileCols) {
		bool increment = false;
		if (!reader.ReadBit(increment)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile columns");
		}
		if (!increment) {
			break;
		}
		++tileColsLog2;
	}
	OaU32 tileRowsLog2 = minLog2Tiles > tileColsLog2 ? minLog2Tiles - tileColsLog2 : 0u;
	while (tileRowsLog2 < Av1FloorLog2(sbRows == 0 ? 1u : sbRows)) {
		bool increment = false;
		if (!reader.ReadBit(increment)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile rows");
		}
		if (!increment) {
			break;
		}
		++tileRowsLog2;
	}
	OutInfo.TileCols = 1u << tileColsLog2;
	OutInfo.TileRows = 1u << tileRowsLog2;
	OutInfo.TileColsLog2 = tileColsLog2;
	OutInfo.TileRowsLog2 = tileRowsLog2;
	if (OutInfo.TileCols * OutInfo.TileRows > 1u) {
		OaU32 contextBits = OutInfo.TileRowsLog2 + OutInfo.TileColsLog2;
		if (!reader.ReadBits(contextBits, OutInfo.ContextUpdateTileId)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 context_update_tile_id");
		}
		if (!reader.ReadBits(2, OutInfo.TileSizeBytesMinus1)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile size byte count");
		}
	}

	if (!reader.ReadBits(8, OutInfo.BaseQIdx)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 quantization header");
	}
	OA_RETURN_IF_ERROR(ReadAv1DeltaQ(reader, OutInfo.DeltaQYDc));
	if (!InSeq.ColorConfig.flags.mono_chrome) {
		if (InSeq.ColorConfig.flags.separate_uv_delta_q) {
			if (!reader.ReadBit(OutInfo.DiffUvDelta)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 diff_uv_delta flag");
			}
		}
		OA_RETURN_IF_ERROR(ReadAv1DeltaQ(reader, OutInfo.DeltaQUDc));
		OA_RETURN_IF_ERROR(ReadAv1DeltaQ(reader, OutInfo.DeltaQUAc));
		if (OutInfo.DiffUvDelta) {
			OA_RETURN_IF_ERROR(ReadAv1DeltaQ(reader, OutInfo.DeltaQVDc));
			OA_RETURN_IF_ERROR(ReadAv1DeltaQ(reader, OutInfo.DeltaQVAc));
		} else {
			OutInfo.DeltaQVDc = OutInfo.DeltaQUDc;
			OutInfo.DeltaQVAc = OutInfo.DeltaQUAc;
		}
	}
	if (!reader.ReadBit(OutInfo.UsingQMatrix)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 qmatrix flag");
	}
	if (OutInfo.UsingQMatrix) {
		if (!reader.ReadBits(4, OutInfo.QmY) || !reader.ReadBits(4, OutInfo.QmU)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 qmatrix header");
		}
		OutInfo.QmV = OutInfo.QmU;
	}
	if (!reader.ReadBit(OutInfo.SegmentationEnabled)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 segmentation flag");
	}
	if (OutInfo.SegmentationEnabled) {
		OutInfo.SegmentationUpdateMap = true;
		OutInfo.SegmentationTemporalUpdate = false;
		OutInfo.SegmentationUpdateData = true;
		static constexpr OaU8 featureBits[STD_VIDEO_AV1_SEG_LVL_MAX] = {8, 6, 6, 6, 6, 3, 0, 0};
		static constexpr bool featureSigned[STD_VIDEO_AV1_SEG_LVL_MAX] = {true, true, true, true, true, false, false, false};
		for (OaU32 segment = 0; segment < STD_VIDEO_AV1_MAX_SEGMENTS; ++segment) {
			for (OaU32 feature = 0; feature < STD_VIDEO_AV1_SEG_LVL_MAX; ++feature) {
				bool enabled = false;
				if (!reader.ReadBit(enabled)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 segmentation feature flag");
				}
				OutInfo.SegmentFeatureEnabled[segment][feature] = enabled ? 1u : 0u;
				if (!enabled) {
					continue;
				}
				if (featureSigned[feature]) {
					OA_RETURN_IF_ERROR(ReadAv1SignedFeatureData(reader, featureBits[feature], OutInfo.SegmentFeatureData[segment][feature]));
				} else if (featureBits[feature] > 0) {
					if (!reader.ReadBits(featureBits[feature], value)) {
						return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 segmentation feature data");
					}
					OutInfo.SegmentFeatureData[segment][feature] = static_cast<OaI16>(value);
				}
			}
		}
	}
	if (OutInfo.BaseQIdx > 0) {
		if (!reader.ReadBit(OutInfo.DeltaQPresent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 delta-q flag");
		}
		if (OutInfo.DeltaQPresent && !reader.ReadBits(2, OutInfo.DeltaQRes)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 delta-q resolution");
		}
	}
	if (OutInfo.DeltaQPresent) {
		if (!reader.ReadBit(OutInfo.DeltaLfPresent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 delta-lf flag");
		}
		if (OutInfo.DeltaLfPresent &&
			(!reader.ReadBits(2, OutInfo.DeltaLfRes) || !reader.ReadBit(OutInfo.DeltaLfMulti))) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 delta-lf header");
		}
	}
	OutInfo.LoopFilterLevels[2] = 0;
	OutInfo.LoopFilterLevels[3] = 0;
	for (OaU32 i = 0; i < 2; ++i) {
		if (!reader.ReadBits(6, value)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 loop filter level");
		}
		OutInfo.LoopFilterLevels[i] = static_cast<OaU8>(value);
	}
	if (!InSeq.ColorConfig.flags.mono_chrome &&
		(OutInfo.LoopFilterLevels[0] != 0 || OutInfo.LoopFilterLevels[1] != 0)) {
		for (OaU32 i = 2; i < 4; ++i) {
			if (!reader.ReadBits(6, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 chroma loop filter level");
			}
			OutInfo.LoopFilterLevels[i] = static_cast<OaU8>(value);
		}
	}
	if (!reader.ReadBits(3, OutInfo.LoopFilterSharpness) || !reader.ReadBit(OutInfo.LoopFilterDeltaEnabled)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 loop filter");
	}
	if (OutInfo.LoopFilterDeltaEnabled) {
		if (!reader.ReadBit(OutInfo.LoopFilterDeltaUpdate)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 loop filter delta update flag");
		}
		if (OutInfo.LoopFilterDeltaUpdate) {
			for (OaU32 i = 0; i < STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME; ++i) {
				bool updateRefDelta = false;
				if (!reader.ReadBit(updateRefDelta)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 loop filter ref delta flag");
				}
				OutInfo.LoopFilterUpdateRefDelta[i] = updateRefDelta ? 1u : 0u;
				if (updateRefDelta) {
					OA_RETURN_IF_ERROR(ReadAv1InverseSignedLiteral(reader, 6, OutInfo.LoopFilterRefDeltas[i]));
				}
			}
			for (OaU32 i = 0; i < STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS; ++i) {
				bool updateModeDelta = false;
				if (!reader.ReadBit(updateModeDelta)) {
					return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 loop filter mode delta flag");
				}
				OutInfo.LoopFilterUpdateModeDelta[i] = updateModeDelta ? 1u : 0u;
				if (updateModeDelta) {
					OA_RETURN_IF_ERROR(ReadAv1InverseSignedLiteral(reader, 6, OutInfo.LoopFilterModeDeltas[i]));
				}
			}
		}
	}
	if (InSeq.EnableCdef) {
		if (!reader.ReadBits(2, OutInfo.CdefDampingMinus3) || !reader.ReadBits(2, OutInfo.CdefBits)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 CDEF header");
		}
		const OaU32 cdefStrengthCount = 1u << OutInfo.CdefBits;
		for (OaU32 i = 0; i < cdefStrengthCount; ++i) {
			if (!reader.ReadBits(4, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 CDEF luma primary strength");
			}
			OutInfo.CdefYPriStrength[i] = static_cast<OaU8>(value);
			if (!reader.ReadBits(2, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 CDEF luma secondary strength");
			}
			OutInfo.CdefYSecStrength[i] = static_cast<OaU8>(value);
			if (!reader.ReadBits(4, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 CDEF chroma primary strength");
			}
			OutInfo.CdefUvPriStrength[i] = static_cast<OaU8>(value);
			if (!reader.ReadBits(2, value)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 CDEF chroma secondary strength");
			}
			OutInfo.CdefUvSecStrength[i] = static_cast<OaU8>(value);
		}
	}
	if (InSeq.EnableRestoration) {
		OA_RETURN_IF_ERROR(ParseAv1RestorationHeader(reader, InSeq.Use128x128Superblock, OutInfo));
	}
	if (!reader.ReadBit(flag)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 transform header");
	}
	OutInfo.TxMode = flag ? STD_VIDEO_AV1_TX_MODE_SELECT : STD_VIDEO_AV1_TX_MODE_LARGEST;

	// AV1 uncompressed_header() tail. These fields are before byte_alignment()
	// and therefore before the tile payload. Omitting even one shifts every tile
	// offset and produces apparently successful but corrupt hardware decode.
	if (!frameIsIntra && !reader.ReadBit(OutInfo.ReferenceSelect)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 reference_select");
	}

	auto relativeDist = [&InSeq](OaU32 InA, OaU32 InB) -> OaI32 {
		if (!InSeq.EnableOrderHint || InSeq.OrderHintBits == 0) return 0;
		const OaU32 modulus = 1u << InSeq.OrderHintBits;
		const OaU32 mask = modulus - 1u;
		const OaU32 sign = modulus >> 1u;
		const OaU32 diff = (InA - InB) & mask;
		return (diff & sign) != 0u
			? static_cast<OaI32>(diff) - static_cast<OaI32>(modulus)
			: static_cast<OaI32>(diff);
	};
	// ref_frame_sign_bias is stored with the reconstructed picture and consumed
	// again when that picture is bound as a Vulkan DPB reference. Bit zero is
	// INTRA_FRAME; bits 1..7 correspond to LAST..ALTREF.
	OutInfo.RefFrameSignBias = 0;
	if (!frameIsIntra && InSeq.EnableOrderHint) {
		for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
			const OaI32 logical = OutInfo.ReferenceNameSlotIndices[i];
			if (logical < 0 || static_cast<OaU32>(logical) >= InRefValid.Size()
				|| !InRefValid[static_cast<OaUsize>(logical)]) continue;
			const OaU32 refHint = InRefOrderHints[static_cast<OaUsize>(logical)];
			if (relativeDist(refHint, OutInfo.OrderHint) > 0) {
				OutInfo.RefFrameSignBias |= static_cast<OaU8>(1u << (i + 1u));
			}
		}
	}
	OaI32 nearestForward = -1;
	OaI32 nearestBackward = -1;
	OaI32 nearestForwardHint = 0;
	OaI32 nearestBackwardHint = 0;
	if (!frameIsIntra && OutInfo.ReferenceSelect && InSeq.EnableOrderHint) {
		for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
			const OaI32 logical = OutInfo.ReferenceNameSlotIndices[i];
			if (logical < 0 || static_cast<OaU32>(logical) >= InRefValid.Size()
				|| !InRefValid[static_cast<OaUsize>(logical)]) continue;
			const OaI32 hint = InRefOrderHints[static_cast<OaUsize>(logical)];
			const OaI32 dist = relativeDist(static_cast<OaU32>(hint), OutInfo.OrderHint);
			if (dist < 0 && (nearestForward < 0
				|| relativeDist(static_cast<OaU32>(hint), static_cast<OaU32>(nearestForwardHint)) > 0)) {
				nearestForward = static_cast<OaI32>(i);
				nearestForwardHint = hint;
			}
			if (dist > 0 && (nearestBackward < 0
				|| relativeDist(static_cast<OaU32>(hint), static_cast<OaU32>(nearestBackwardHint)) < 0)) {
				nearestBackward = static_cast<OaI32>(i);
				nearestBackwardHint = hint;
			}
		}
	}
	OaI32 secondForward = -1;
	OaI32 secondForwardHint = 0;
	if (nearestForward >= 0 && nearestBackward < 0) {
		for (OaU32 i = 0; i < OaAv1MaxReferencesPerFrame; ++i) {
			const OaI32 logical = OutInfo.ReferenceNameSlotIndices[i];
			if (logical < 0 || static_cast<OaU32>(logical) >= InRefValid.Size()
				|| !InRefValid[static_cast<OaUsize>(logical)]) continue;
			const OaI32 hint = InRefOrderHints[static_cast<OaUsize>(logical)];
			if (relativeDist(static_cast<OaU32>(hint), static_cast<OaU32>(nearestForwardHint)) < 0
				&& (secondForward < 0 || relativeDist(static_cast<OaU32>(hint),
					static_cast<OaU32>(secondForwardHint)) > 0)) {
				secondForward = static_cast<OaI32>(i);
				secondForwardHint = hint;
			}
		}
	}
	const OaI32 skipOther = nearestBackward >= 0 ? nearestBackward : secondForward;
	if (nearestForward >= 0 && skipOther >= 0) {
		if (!reader.ReadBit(OutInfo.SkipModePresent)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 skip_mode_present");
		}
		const OaU8 ref0 = static_cast<OaU8>(nearestForward + STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME);
		const OaU8 ref1 = static_cast<OaU8>(skipOther + STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME);
		OutInfo.SkipModeFrame[0] = ref0 < ref1 ? ref0 : ref1;
		OutInfo.SkipModeFrame[1] = ref0 < ref1 ? ref1 : ref0;
	}

	if (!frameIsIntra && !OutInfo.ErrorResilientMode && InSeq.EnableWarpedMotion) {
		if (!reader.ReadBit(OutInfo.AllowWarpedMotion)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 allow_warped_motion");
		}
	}
	if (!reader.ReadBit(OutInfo.ReducedTxSet)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 reduced_tx_set");
	}

	if (!frameIsIntra) {
		for (OaU32 ref = 0; ref < OaAv1MaxReferencesPerFrame; ++ref) {
			bool isGlobal = false;
			if (!reader.ReadBit(isGlobal)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 global motion flag");
			}
			if (isGlobal) {
				return OaStatus::Error(OaStatusCode::Unimplemented,
					"AV1 non-identity global motion is not implemented");
			}
		}
	}
	if (InSeq.FilmGrainParamsPresent && (OutInfo.ShowFrame || OutInfo.ShowableFrame)) {
		if (!reader.ReadBit(OutInfo.ApplyGrain)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 apply_grain");
		}
		if (OutInfo.ApplyGrain) {
			return OaStatus::Error(OaStatusCode::Unimplemented,
				"AV1 film-grain parameters are not implemented");
		}
	}
	reader.ByteAlign();
	if (reader.ByteOffset() >= InObu.PayloadSize) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 frame header consumes the full frame payload");
	}
	OutInfo.HeaderSize = reader.ByteOffset();
	return OaStatus::Ok();
}

static OaStatus ParseAv1TileGroup(
	const OaAv1Obu& InObu,
	const OaSpan<const OaU8>& InFrame,
	const OaAv1FrameHeaderInfo& InFrameHeader,
	bool InPayloadIncludesFrameHeader,
	OaAv1TileGroupInfo& OutTileGroup)
{
	OutTileGroup.TileOffsets.Clear();
	OutTileGroup.TileSizes.Clear();
	if (InObu.PayloadOffset + InObu.PayloadSize > InFrame.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}
	if (InPayloadIncludesFrameHeader && InObu.PayloadSize <= InFrameHeader.HeaderSize) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}
	const OaUsize tileGroupOffset = InPayloadIncludesFrameHeader
		? InObu.PayloadOffset + InFrameHeader.HeaderSize
		: InObu.PayloadOffset;
	const OaUsize tileGroupSize = InPayloadIncludesFrameHeader
		? InObu.PayloadSize - InFrameHeader.HeaderSize
		: InObu.PayloadSize;
	if (tileGroupOffset > InFrame.Size() || tileGroupSize == 0 || tileGroupOffset + tileGroupSize > InFrame.Size()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile group has no tile payload");
	}

	const OaU32 tileCount = InFrameHeader.TileCols * InFrameHeader.TileRows;
	if (tileCount == 0 || tileCount > STD_VIDEO_AV1_MAX_TILE_COLS * STD_VIDEO_AV1_MAX_TILE_ROWS) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile count");
	}
	if (tileCount == 1) {
		OutTileGroup.TileOffsets.PushBack(static_cast<OaU32>(tileGroupOffset));
		OutTileGroup.TileSizes.PushBack(static_cast<OaU32>(tileGroupSize));
		return OaStatus::Ok();
	}

	OaAv1BitReader reader(InFrame.Data() + tileGroupOffset, tileGroupSize);
	bool hasTileStartAndEnd = false;
	if (!reader.ReadBit(hasTileStartAndEnd)) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile group range flag");
	}

	const OaU32 tileNumberBits = InFrameHeader.TileColsLog2 + InFrameHeader.TileRowsLog2;
	OaU32 tileStart = 0;
	OaU32 tileEnd = tileCount - 1u;
	if (hasTileStartAndEnd) {
		if (!reader.ReadBits(tileNumberBits, tileStart) || !reader.ReadBits(tileNumberBits, tileEnd)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile group range");
		}
		if (tileStart > tileEnd || tileEnd >= tileCount) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile group range");
		}
	}
	reader.ByteAlign();

	const OaU32 tileSizeBytes = InFrameHeader.TileSizeBytesMinus1 + 1u;
	if (tileSizeBytes == 0 || tileSizeBytes > 4) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 tile size byte count");
	}

	OaUsize cursor = tileGroupOffset + reader.ByteOffset();
	for (OaU32 tile = tileStart; tile <= tileEnd; ++tile) {
		OaU32 tileSize = 0;
		if (tile == tileEnd) {
			if (cursor > InObu.PayloadOffset + InObu.PayloadSize) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile cursor exceeds payload");
			}
			tileSize = static_cast<OaU32>(InObu.PayloadOffset + InObu.PayloadSize - cursor);
		} else {
			if (cursor + tileSizeBytes > InObu.PayloadOffset + InObu.PayloadSize) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 tile size field");
			}
			tileSize = ReadLeBytes(InFrame.Data() + cursor, tileSizeBytes) + 1u;
			cursor += tileSizeBytes;
			if (cursor + tileSize > InObu.PayloadOffset + InObu.PayloadSize) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile payload exceeds tile group");
			}
		}
		if (tileSize == 0) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 tile payload is empty");
		}
		OutTileGroup.TileOffsets.PushBack(static_cast<OaU32>(cursor));
		OutTileGroup.TileSizes.PushBack(tileSize);
		cursor += tileSize;
	}
	return OaStatus::Ok();
}

static OaStatus ExtractAv1FramePayload(const OaSpan<const OaU8>& InBitstream, OaIvfFrame& OutFrame)
{
	const OaU8* data = InBitstream.Data();
	const OaUsize size = InBitstream.Size();

	// Support both IVF-wrapped (our test assets) and raw OBU payload (real demuxer access units).
	if (size >= 44 &&
		data[0] == 'D' && data[1] == 'K' && data[2] == 'I' && data[3] == 'F' &&
		data[8] == 'A' && data[9] == 'V' && data[10] == '0' && data[11] == '1') {
		const OaU32 headerSize = static_cast<OaU32>(data[6]) | (static_cast<OaU32>(data[7]) << 8u);
		if (headerSize < 32 || static_cast<OaUsize>(headerSize) + 12u > size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 IVF header size");
		}
		const OaUsize frameHeaderOffset = headerSize;
		const OaU32 frameSize = ReadLe32(data + frameHeaderOffset);
		if (frameSize == 0 || frameHeaderOffset + 12u + frameSize > size) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 IVF frame size");
		}
		OutFrame.Offset = frameHeaderOffset + 12u;
		OutFrame.Size = frameSize;
		OutFrame.Timestamp = ReadLe64(data + frameHeaderOffset + 4u);
		return OaStatus::Ok();
	}

	// Raw access unit (OBU sequence for one picture, as delivered by MP4/WebM demuxer etc.)
	OutFrame.Offset = 0;
	OutFrame.Size = size;
	OutFrame.Timestamp = 0;
	return OaStatus::Ok();
}

static OaStatus FindAv1Obus(const OaSpan<const OaU8>& InFrame, OaVec<OaAv1Obu>& OutObus)
{
	OutObus.Clear();
	const OaU8* data = InFrame.Data();
	const OaUsize size = InFrame.Size();
	OaUsize offset = 0;
	while (offset < size) {
		const OaUsize headerOffset = offset;
		const OaU8 header = data[offset++];
		if ((header & 0x80u) != 0 || (header & 0x01u) != 0) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 OBU header");
		}
		const OaAv1ObuType type = static_cast<OaAv1ObuType>((header >> 3u) & 0x0fu);
		const bool hasExtension = (header & 0x04u) != 0;
		const bool hasSize = (header & 0x02u) != 0;
		if (hasExtension) {
			if (offset >= size) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Truncated AV1 OBU extension header");
			}
			++offset;
		}
		OaU64 payloadSize = 0;
		if (hasSize) {
			if (!ReadLeb128(data, size, offset, payloadSize)) {
				return OaStatus::Error(OaStatusCode::InvalidArgument, "Invalid AV1 OBU payload size");
			}
		} else {
			payloadSize = size - offset;
		}
		if (payloadSize > size - offset) {
			return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 OBU payload exceeds frame data");
		}
		OutObus.PushBack({
			type,
			headerOffset,
			offset - headerOffset,
			offset,
			static_cast<OaUsize>(payloadSize)});
		offset += static_cast<OaUsize>(payloadSize);
	}
	if (OutObus.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AV1 frame has no OBUs");
	}
	return OaStatus::Ok();
}

// ============================================================================
// Parser Implementation
// ============================================================================

OaStatus OaVcpAv1::ParseSps(const OaSpan<const OaU8>& InObu)
{
	(void)InObu;
	return OaStatus::Error(OaStatusCode::Unimplemented, "AV1 sequence header parsing via ParseSps is not implemented; use ParseAccessUnit");
}

OaStatus OaVcpAv1::ParsePps(const OaSpan<const OaU8>& InObu)
{
	(void)InObu;
	return OaStatus::Error(OaStatusCode::Unimplemented, "AV1 frame header parsing via ParsePps is not implemented; use ParseAccessUnit");
}

void OaVcpAv1::ClearParameterSets()
{
	CachedSequenceHeader_ = {};
	HasCachedSequenceHeader_ = false;
	CachedRefOrderHints_.Fill(0);
	CachedRefValid_.Fill(false);
}

OaStatus OaVcpAv1::InspectAccessUnit(
	const OaSpan<const OaU8>& InBitstream,
	OaAv1AccessUnitInfo& OutInfo) const
{
	OutInfo = {};
	OaIvfFrame frame = {};
	OA_RETURN_IF_ERROR(ExtractAv1FramePayload(InBitstream, frame));
	OaVec<OaAv1Obu> obus;
	OA_RETURN_IF_ERROR(FindAv1Obus(
		OaSpan<const OaU8>(InBitstream.Data() + frame.Offset, frame.Size), obus));
	for (const OaAv1Obu& obu : obus) {
		switch (obu.Type) {
		case OaAv1ObuType::SequenceHeader: ++OutInfo.SequenceHeaderCount; break;
		case OaAv1ObuType::Frame: ++OutInfo.FrameCount; break;
		case OaAv1ObuType::FrameHeader: ++OutInfo.FrameHeaderCount; break;
		case OaAv1ObuType::TileGroup: ++OutInfo.TileGroupCount; break;
		default: break;
		}
	}
	return OaStatus::Ok();
}

OaStatus OaVcpAv1::ParseAccessUnit(const OaSpan<const OaU8>& InBitstream, OaAv1PictureDesc& OutDesc)
{
	OutDesc = {};
	OaVec<OaAv1PictureDesc> pictures;
	OA_RETURN_IF_ERROR(ParseAccessUnitPictures(InBitstream, pictures));
	if (!pictures.Empty()) OutDesc = pictures[0];
	return OaStatus::Ok();
}

OaStatus OaVcpAv1::ParseAccessUnitPictures(
	const OaSpan<const OaU8>& InBitstream,
	OaVec<OaAv1PictureDesc>& OutDescs)
{
	OutDescs.Clear();

	OaIvfFrame frame = {};
	OA_RETURN_IF_ERROR(ExtractAv1FramePayload(InBitstream, frame));

	OaVec<OaAv1Obu> obus;
	OA_RETURN_IF_ERROR(FindAv1Obus(
		OaSpan<const OaU8>(InBitstream.Data() + frame.Offset, frame.Size),
		obus));
	OaSpan<const OaU8> av1FrameData(InBitstream.Data() + frame.Offset, frame.Size);

	auto parsePicture = [&](const OaAv1Obu& InHeader, OaUsize InFirstTile,
		OaUsize InTileEnd, bool InCombinedFrame) -> OaStatus {
		if (!HasCachedSequenceHeader_) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"AV1 picture has no cached sequence header");
		}
		OaAv1PictureDesc desc = {};
		desc.Frame = frame;
		desc.SequenceHeader = CachedSequenceHeader_;
		OA_RETURN_IF_ERROR(ParseAv1FrameHeader(InHeader, av1FrameData,
			desc.SequenceHeader, CachedRefOrderHints_, CachedRefValid_, desc.FrameHeader));
		desc.ShowExistingFrame = desc.FrameHeader.ShowExistingFrame;
		desc.FrameToShowMapIdx = desc.FrameHeader.FrameToShowMapIdx;
		desc.FrameHeaderOffset = InHeader.PayloadOffset;
		desc.DecodeObuOffset = InHeader.HeaderOffset;
		if (desc.ShowExistingFrame) {
			desc.DecodeObuSize = InHeader.HeaderSize + InHeader.PayloadSize;
			OutDescs.PushBack(desc);
			return OaStatus::Ok();
		}
		if (InFirstTile >= InTileEnd || InFirstTile >= obus.Size()) {
			return OaStatus::Error(OaStatusCode::Unavailable,
				"AV1 frame header has no tile payload");
		}
		const OaAv1Obu& lastTile = obus[InTileEnd - 1u];
		desc.DecodeObuSize = lastTile.HeaderOffset + lastTile.HeaderSize
			+ lastTile.PayloadSize - InHeader.HeaderOffset;
		for (OaUsize tileIndex = InFirstTile; tileIndex < InTileEnd; ++tileIndex) {
			OaAv1TileGroupInfo group = {};
			OA_RETURN_IF_ERROR(ParseAv1TileGroup(obus[tileIndex], av1FrameData,
				desc.FrameHeader, InCombinedFrame, group));
			for (OaU32 offset : group.TileOffsets) desc.TileOffsets.PushBack(offset);
			for (OaU32 size : group.TileSizes) desc.TileSizes.PushBack(size);
		}
		desc.TileGroup.TileOffsets = desc.TileOffsets;
		desc.TileGroup.TileSizes = desc.TileSizes;
		desc.HasPicture = true;
		for (OaU32 mask = desc.FrameHeader.RefreshFrameFlags, ref = 0;
			mask != 0u && ref < STD_VIDEO_AV1_NUM_REF_FRAMES; mask >>= 1u, ++ref) {
			if ((mask & 1u) != 0u) {
				CachedRefOrderHints_[ref] = static_cast<OaU8>(desc.FrameHeader.OrderHint);
				CachedRefValid_[ref] = true;
			}
		}
		OutDescs.PushBack(desc);
		return OaStatus::Ok();
	};

	for (OaUsize i = 0; i < obus.Size();) {
		const OaAv1Obu& obu = obus[i];
		if (obu.Type == OaAv1ObuType::SequenceHeader) {
			OA_RETURN_IF_ERROR(ParseAv1SequenceHeader(obu, av1FrameData,
				CachedSequenceHeader_));
			HasCachedSequenceHeader_ = true;
			++i;
			continue;
		}
		if (obu.Type == OaAv1ObuType::Frame) {
			OA_RETURN_IF_ERROR(parsePicture(obu, i, i + 1u, true));
			++i;
			continue;
		}
		if (obu.Type == OaAv1ObuType::FrameHeader) {
			OaUsize end = i + 1u;
			while (end < obus.Size() && obus[end].Type == OaAv1ObuType::TileGroup) ++end;
			OA_RETURN_IF_ERROR(parsePicture(obu, i + 1u, end, false));
			i = end;
			continue;
		}
		++i;
	}
	if (!HasCachedSequenceHeader_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"AV1 access unit is missing a sequence header and none is cached");
	}
	return OaStatus::Ok();
}

namespace {
struct OaAv1CodecRegistrar {
	OaAv1CodecRegistrar() {
		auto parser = OaStdMakeUnique<OaVcpAv1>();
		OaVideoCodecRegistry::GetInstance().RegisterParser(
			OaVideoCodec::AV1,
			OaStdMove(parser));
	}
};
static OaAv1CodecRegistrar g_Av1Registrar __attribute__((used));
} // namespace
