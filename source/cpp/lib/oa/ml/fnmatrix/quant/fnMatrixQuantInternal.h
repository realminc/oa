#pragma once

#include <oa/core/matrix.h>

// Private physical-plane operations. Their stable operation/kernel identities
// remain schema-owned, but consumers use oa::QuantMatrix and the generic
// quantize/Dequantize/MatMulNt surface.
namespace oa {

namespace FnMatrix {

[[nodiscard]] oa::Matrix computeScaleQ4(const oa::Matrix& inInput);
[[nodiscard]] oa::Matrix quantizeQ4(
	const oa::Matrix& inInput,
	const oa::Matrix& inScale);
[[nodiscard]] oa::Matrix dequantizeQ4(
	const oa::Matrix& inInput,
	const oa::Matrix& inScale,
	oa::I64 inCount);
[[nodiscard]] oa::Matrix matMulNtQ4(
	const oa::Matrix& inInput,
	const oa::Matrix& inPayload,
	const oa::Matrix& inScale,
	oa::I64 inOutputFeatures);

[[nodiscard]] oa::Matrix computeScaleQ8(const oa::Matrix& inInput);
[[nodiscard]] oa::Matrix quantizeQ8(
	const oa::Matrix& inInput,
	const oa::Matrix& inScale);
[[nodiscard]] oa::Matrix dequantizeQ8(
	const oa::Matrix& inInput,
	const oa::Matrix& inScale,
	oa::I64 inCount);
[[nodiscard]] oa::Matrix matMulNtQ8(
	const oa::Matrix& inInput,
	const oa::Matrix& inPayload,
	const oa::Matrix& inScale,
	oa::I64 inOutputFeatures);

} // namespace FnMatrix

} // namespace oa
