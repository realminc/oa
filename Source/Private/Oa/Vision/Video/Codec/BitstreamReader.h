// OA Vision — H.264/H.265 Bitstream Reader
// Exp-Golomb decoding for NAL unit parsing

#pragma once

#include <Oa/Core/Types.h>

// Bitstream reader for H.264/H.265 NAL units
// Handles Exp-Golomb variable-length codes and bit-level reading
class OaBitstreamReader
{
public:
	// Initialize reader with NAL unit data (after start code)
	OaBitstreamReader(const OaU8* InData, OaUsize InSize)
		: Data_(InData), Size_(InSize), BytePos_(0), BitPos_(0) {}

	// Read single bit
	OaU32 ReadBit()
	{
		if (BytePos_ >= Size_)
		{
			return 0; // EOF
		}

		OaU32 bit = (Data_[BytePos_] >> (7 - BitPos_)) & 1;
		++BitPos_;
		if (BitPos_ == 8)
		{
			BitPos_ = 0;
			++BytePos_;
		}
		return bit;
	}

	// Read N bits (up to 32)
	OaU32 ReadBits(OaU32 InN)
	{
		OaU32 value = 0;
		for (OaU32 i = 0; i < InN; ++i)
		{
			value = (value << 1) | ReadBit();
		}
		return value;
	}

	// Read unsigned Exp-Golomb code (ue(v))
	OaU32 ReadUE()
	{
		// Count leading zeros
		OaU32 leadingZeros = 0;
		while (ReadBit() == 0 && leadingZeros < 32)
		{
			++leadingZeros;
		}

		if (leadingZeros == 0)
		{
			return 0;
		}

		// Read remaining bits
		OaU32 value = (1 << leadingZeros) - 1;
		value += ReadBits(leadingZeros);
		return value;
	}

	// Read signed Exp-Golomb code (se(v))
	OaI32 ReadSE()
	{
		OaU32 codeNum = ReadUE();
		if (codeNum == 0)
		{
			return 0;
		}

		// Map to signed: 1→+1, 2→-1, 3→+2, 4→-2, ...
		OaI32 sign = (codeNum & 1) ? 1 : -1;
		return sign * static_cast<OaI32>((codeNum + 1) >> 1);
	}

	// Skip N bits
	void SkipBits(OaU32 InN)
	{
		for (OaU32 i = 0; i < InN; ++i)
		{
			ReadBit();
		}
	}

	// Byte-align (skip to next byte boundary)
	void ByteAlign()
	{
		if (BitPos_ != 0)
		{
			BitPos_ = 0;
			++BytePos_;
		}
	}

	// Check if more data available
	bool HasMoreData() const
	{
		return BytePos_ < Size_;
	}

	// Get current byte position
	OaUsize GetBytePos() const { return BytePos_; }

	// Get current bit position within byte
	OaU32 GetBitPos() const { return BitPos_; }

private:
	const OaU8* Data_;
	OaUsize Size_;
	OaUsize BytePos_;
	OaU32 BitPos_;  // 0-7
};
