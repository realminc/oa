// OA Vision — H.264/H.265 bitstream Reader
// Exp-Golomb decoding for NAL unit parsing

#pragma once

#include <oa/core/types.h>

namespace oa {

// bitstream reader for H.264/H.265 NAL units
// Handles Exp-Golomb variable-length codes and bit-level reading
class BitstreamReader
{
public:
	// initialize reader with NAL unit data (after start code)
	BitstreamReader(const oa::U8* inData, oa::Usize inSize)
		: data_(inData), size_(inSize), bytePos_(0), bitPos_(0) {}

	// Read single bit
	oa::U32 readBit()
	{
		if (bytePos_ >= size_)
		{
			return 0; // EOF
		}

		oa::U32 bit = (data_[bytePos_] >> (7 - bitPos_)) & 1;
		++bitPos_;
		if (bitPos_ == 8)
		{
			bitPos_ = 0;
			++bytePos_;
		}
		return bit;
	}

	// Read N bits (up to 32)
	oa::U32 readBits(oa::U32 inN)
	{
		oa::U32 value = 0;
		for (oa::U32 i = 0; i < inN; ++i)
		{
			value = (value << 1) | readBit();
		}
		return value;
	}

	// Read unsigned Exp-Golomb code (ue(v))
	oa::U32 readUE()
	{
		// Count leading zeros
		oa::U32 leadingZeros = 0;
		while (readBit() == 0 && leadingZeros < 32)
		{
			++leadingZeros;
		}

		if (leadingZeros == 0)
		{
			return 0;
		}

		// Read remaining bits
		oa::U32 value = (1 << leadingZeros) - 1;
		value += readBits(leadingZeros);
		return value;
	}

	// Read signed Exp-Golomb code (se(v))
	oa::I32 readSE()
	{
		oa::U32 codeNum = readUE();
		if (codeNum == 0)
		{
			return 0;
		}

		// map to signed: 1→+1, 2→-1, 3→+2, 4→-2, ...
		oa::I32 sign = (codeNum & 1) ? 1 : -1;
		return sign * static_cast<oa::I32>((codeNum + 1) >> 1);
	}

	// Skip N bits
	void skipBits(oa::U32 inN)
	{
		for (oa::U32 i = 0; i < inN; ++i)
		{
			readBit();
		}
	}

	// Byte-align (skip to next byte boundary)
	void byteAlign()
	{
		if (bitPos_ != 0)
		{
			bitPos_ = 0;
			++bytePos_;
		}
	}

	// Check if more data available
	bool hasMoreData() const
	{
		return bytePos_ < size_;
	}

	// get current byte position
	oa::Usize getBytePos() const { return bytePos_; }

	// get current bit position within byte
	oa::U32 getBitPos() const { return bitPos_; }

private:
	const oa::U8* data_;
	oa::Usize size_;
	oa::Usize bytePos_;
	oa::U32 bitPos_;  // 0-7
};

} // namespace oa
