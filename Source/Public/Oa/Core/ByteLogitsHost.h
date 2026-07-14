#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

// Host CPU: byte LM logits [seq, 256] or [1, seq, 256] (Float32, row-major) → argmax per row → UInt8 codes.
// Matches OaByteEncoder::Decode layout; no ML headers.

[[nodiscard]] OaVec<OaU8> OaByteLogitsArgmaxFromContiguousF32(OaI64 InSeqLen, const OaF32* InRowMajorLogits);

[[nodiscard]] OaResult<OaVec<OaU8>> OaByteLogitsArgmaxHost(const OaMatrix& InLogits);
