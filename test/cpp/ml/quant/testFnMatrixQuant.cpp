#include "../../oaTest.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/matrix.h>
#include <oa/core/opRegistry.gen.h>
#include <oa/ml/fnMatrix.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/fnmatrix/quant/fnMatrixQuantInternal.h>
#include <oa/ml/quantMatrixAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/semanticGraph.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr oa::I64 blockSize = 32;
constexpr oa::I64 BytesPerBlock = 16;
constexpr oa::I64 Q8BytesPerBlock = 32;

[[nodiscard]] oa::Matrix makeInput(const std::vector<oa::F32>& inValues) {
	auto matrix = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(inValues.size())},
		oa::ScalarType::Float32);
	for (oa::I64 i = 0; i < matrix.numElements(); ++i) {
		matrix.set(i, inValues[static_cast<std::size_t>(i)]);
	}
	return matrix;
}

template<typename T>
[[nodiscard]] oa::Matrix makeRawMatrix(
	const std::vector<T>& inValues,
	oa::MatrixShape inShape,
	oa::ScalarType inDtype)
{
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(inValues.data()),
			inValues.size() * sizeof(T)),
		inShape,
		inDtype);
}

[[nodiscard]] std::vector<oa::F32> referenceScales(
	const std::vector<oa::F32>& inValues,
	oa::F32 inQuantizedMax)
{
	const oa::I64 blocks = (static_cast<oa::I64>(inValues.size()) + blockSize - 1)
		/ blockSize;
	std::vector<oa::F32> scales(static_cast<std::size_t>(blocks), 1.0F);
	for (oa::I64 block = 0; block < blocks; ++block) {
		oa::F32 blockMax = 0.0F;
		for (oa::I64 lane = 0; lane < blockSize; ++lane) {
			const oa::I64 index = block * blockSize + lane;
			if (index < static_cast<oa::I64>(inValues.size())
				and std::isfinite(inValues[static_cast<std::size_t>(index)]))
			{
				blockMax = std::max(blockMax,
					std::abs(inValues[static_cast<std::size_t>(index)]));
			}
		}
		const oa::F32 scale = blockMax / inQuantizedMax;
		if (scale > 0.0F and std::isfinite(scale)) {
			scales[static_cast<std::size_t>(block)] = scale;
		}
	}
	return scales;
}

template<typename T>
[[nodiscard]] std::vector<T> read(const oa::Matrix& inMatrix) {
	std::vector<T> values(static_cast<std::size_t>(inMatrix.numElements()));
	if (not values.empty()) {
		EXPECT_TRUE(oa::FnMatrix::copyToHost(
			inMatrix, values.data(), values.size() * sizeof(T)).isOk());
	}
	return values;
}

[[nodiscard]] oa::U8 quantizeNibble(oa::F32 inValue, oa::F32 inScale) {
	const auto rounded = static_cast<oa::I32>(std::nearbyint(inValue / inScale));
	return static_cast<oa::U8>(std::clamp(rounded, -7, 7) + 7);
}

[[nodiscard]] std::vector<oa::U8> referencePayload(
	const std::vector<oa::F32>& inValues, const std::vector<oa::F32>& inScales)
{
	const oa::I64 blocks = static_cast<oa::I64>(inScales.size());
	std::vector<oa::U8> payload(
		static_cast<std::size_t>(blocks * BytesPerBlock), 0x77U);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inValues.size()); ++i) {
		const oa::U8 nibble = quantizeNibble(
			inValues[static_cast<std::size_t>(i)],
			inScales[static_cast<std::size_t>(i / blockSize)]);
		auto& packed = payload[static_cast<std::size_t>(i / 2)];
		if (i % 2 == 0) {
			packed = static_cast<oa::U8>((packed & 0xF0U) | nibble);
		} else {
			packed = static_cast<oa::U8>((packed & 0x0FU)
				| (static_cast<oa::U32>(nibble) << 4U));
		}
	}
	return payload;
}

void expectRoundTrip(const std::vector<oa::F32>& inValues) {
	const auto input = makeInput(inValues);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto dequantized = oa::FnMatrix::dequantizeQ4(
		quantized, scale, static_cast<oa::I64>(inValues.size()));
	const auto scales = read<oa::F32>(scale);
	const auto payload = read<oa::U8>(quantized);
	const auto output = read<oa::F32>(dequantized);

	EXPECT_EQ(payload, referencePayload(inValues, scales));
	ASSERT_EQ(output.size(), inValues.size());
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inValues.size()); ++i) {
		const oa::F32 tolerance =
			scales[static_cast<std::size_t>(i / blockSize)] * 0.5001F + 1.0e-6F;
		EXPECT_NEAR(output[static_cast<std::size_t>(i)],
			inValues[static_cast<std::size_t>(i)], tolerance) << "element " << i;
	}
}

[[nodiscard]] oa::I8 quantizeQ8Value(oa::F32 inValue, oa::F32 inScale) {
	const auto rounded = static_cast<oa::I32>(std::nearbyint(inValue / inScale));
	return static_cast<oa::I8>(std::clamp(rounded, -127, 127));
}

[[nodiscard]] std::vector<oa::I8> referenceQ8Payload(
	const std::vector<oa::F32>& inValues, const std::vector<oa::F32>& inScales)
{
	const oa::I64 blocks = static_cast<oa::I64>(inScales.size());
	std::vector<oa::I8> payload(
		static_cast<std::size_t>(blocks * Q8BytesPerBlock), 0);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inValues.size()); ++i) {
		payload[static_cast<std::size_t>(i)] = quantizeQ8Value(
			inValues[static_cast<std::size_t>(i)],
			inScales[static_cast<std::size_t>(i / blockSize)]);
	}
	return payload;
}

void expectQ8RoundTrip(const std::vector<oa::F32>& inValues) {
	const auto input = makeInput(inValues);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	const auto dequantized = oa::FnMatrix::dequantizeQ8(
		quantized, scale, static_cast<oa::I64>(inValues.size()));
	const auto scales = read<oa::F32>(scale);
	const auto payload = read<oa::I8>(quantized);
	const auto output = read<oa::F32>(dequantized);

	EXPECT_EQ(payload, referenceQ8Payload(inValues, scales));
	ASSERT_EQ(output.size(), inValues.size());
	for (oa::I64 i = 0; i < static_cast<oa::I64>(inValues.size()); ++i) {
		const oa::F32 tolerance =
			scales[static_cast<std::size_t>(i / blockSize)] * 0.5001F + 1.0e-6F;
		EXPECT_NEAR(output[static_cast<std::size_t>(i)],
			inValues[static_cast<std::size_t>(i)], tolerance) << "element " << i;
	}
}

[[nodiscard]] std::vector<oa::F32> referenceMatMulNtQ4(
	const std::vector<oa::F32>& inInput,
	const std::vector<oa::U8>& inPayload,
	const std::vector<oa::F32>& inScale,
	oa::I64 inRows,
	oa::I64 inOutputFeatures,
	oa::I64 inInputFeatures)
{
	std::vector<oa::F32> output(
		static_cast<std::size_t>(inRows * inOutputFeatures), 0.0F);
	for (oa::I64 row = 0; row < inRows; ++row) {
		for (oa::I64 column = 0; column < inOutputFeatures; ++column) {
			oa::F32 sum = 0.0F;
			for (oa::I64 k = 0; k < inInputFeatures; ++k) {
				const oa::I64 weightIndex = column * inInputFeatures + k;
				const oa::U8 packed = inPayload[
					static_cast<std::size_t>(weightIndex / 2)];
				const oa::U8 nibble = weightIndex % 2 == 0
					? packed & 0x0FU
					: (packed >> 4U) & 0x0FU;
				const oa::F32 scale = inScale[
					static_cast<std::size_t>(weightIndex / blockSize)];
				const oa::F32 weight = std::isfinite(scale) and scale > 0.0F
					? static_cast<oa::F32>(static_cast<oa::I32>(nibble) - 7) * scale
					: 0.0F;
				sum += inInput[static_cast<std::size_t>(
					row * inInputFeatures + k)] * weight;
			}
			output[static_cast<std::size_t>(row * inOutputFeatures + column)] = sum;
		}
	}
	return output;
}

[[nodiscard]] std::vector<oa::F32> referenceMatMulNtQ8(
	const std::vector<oa::F32>& inInput,
	const std::vector<oa::I8>& inPayload,
	const std::vector<oa::F32>& inScale,
	oa::I64 inRows,
	oa::I64 inOutputFeatures,
	oa::I64 inInputFeatures)
{
	std::vector<oa::F32> output(
		static_cast<std::size_t>(inRows * inOutputFeatures), 0.0F);
	for (oa::I64 row = 0; row < inRows; ++row) {
		for (oa::I64 column = 0; column < inOutputFeatures; ++column) {
			oa::F32 sum = 0.0F;
			for (oa::I64 k = 0; k < inInputFeatures; ++k) {
				const oa::I64 weightIndex = column * inInputFeatures + k;
				const oa::F32 scale = inScale[
					static_cast<std::size_t>(weightIndex / blockSize)];
				const oa::F32 weight = std::isfinite(scale) and scale > 0.0F
					? static_cast<oa::F32>(inPayload[
						static_cast<std::size_t>(weightIndex)]) * scale
					: 0.0F;
				sum += inInput[static_cast<std::size_t>(
					row * inInputFeatures + k)] * weight;
			}
			output[static_cast<std::size_t>(row * inOutputFeatures + column)] = sum;
		}
	}
	return output;
}

void expectNearVectors(
	const std::vector<oa::F32>& inActual,
	const std::vector<oa::F32>& inExpected,
	oa::F32 inRelativeTolerance = 2.0e-4F)
{
	ASSERT_EQ(inActual.size(), inExpected.size());
	for (std::size_t i = 0; i < inExpected.size(); ++i) {
		const oa::F32 tolerance = inRelativeTolerance
			* std::max(1.0F, std::abs(inExpected[i]));
		EXPECT_NEAR(inActual[i], inExpected[i], tolerance) << "element " << i;
	}
}

} // namespace

TEST(FnMatrixQuant, ComputeScaleOwnsEveryBlockAndPartialTail) {
	std::vector<oa::F32> values(73, 0.0F);
	values[0] = -7.0F;
	values[33] = 14.0F;
	values[72] = -21.0F;

	const auto scales = read<oa::F32>(
		oa::FnMatrix::computeScaleQ4(makeInput(values)));
	ASSERT_EQ(scales.size(), 3U);
	EXPECT_FLOAT_EQ(scales[0], 1.0F);
	EXPECT_FLOAT_EQ(scales[1], 2.0F);
	EXPECT_FLOAT_EQ(scales[2], 3.0F);
}

TEST(FnMatrixQuant, ZeroBlocksUseFiniteUnitScale) {
	const auto scales = read<oa::F32>(oa::FnMatrix::computeScaleQ4(
		makeInput(std::vector<oa::F32>(65, 0.0F))));
	ASSERT_EQ(scales.size(), 3U);
	EXPECT_EQ(scales, (std::vector<oa::F32>{1.0F, 1.0F, 1.0F}));
}

TEST(FnMatrixQuant, PackedPayloadMatchesIndependentAlignedOracle) {
	std::vector<oa::F32> values(50);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(values.size()); ++i) {
		values[static_cast<std::size_t>(i)] =
			static_cast<oa::F32>((i * 11) % 29 - 14);
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto scales = read<oa::F32>(scale);
	const auto payload = read<oa::U8>(quantized);

	ASSERT_EQ(payload.size(), 32U);
	EXPECT_EQ(payload, referencePayload(values, scales));
	for (std::size_t i = 25; i < payload.size(); ++i) {
		EXPECT_EQ(payload[i], 0x77U) << "padding byte " << i;
	}
}

TEST(FnMatrixQuant, OddSizesAndPartialBlocksRoundTripWithinHalfStep) {
	for (const oa::I64 count : {1, 7, 31, 32, 33, 50, 257}) {
		std::vector<oa::F32> values(static_cast<std::size_t>(count));
		for (oa::I64 i = 0; i < count; ++i) {
			values[static_cast<std::size_t>(i)] =
				std::sin(static_cast<oa::F32>(i) * 0.17F) * 13.0F;
		}
		expectRoundTrip(values);
	}
}

TEST(FnMatrixQuant, SignAndExtremeValuesArePreserved) {
	std::vector<oa::F32> values(32);
	for (oa::I64 i = 0; i < 32; ++i) {
		values[static_cast<std::size_t>(i)] = i % 2 == 0 ? -7.0F : 7.0F;
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ4(
		quantized, scale, static_cast<oa::I64>(values.size())));
	EXPECT_EQ(output, values);
}

TEST(FnMatrixQuant, NonFiniteValuesEncodeAsZeroAndDoNotPoisonScale) {
	std::vector<oa::F32> values(64, 0.0F);
	values[0] = std::numeric_limits<oa::F32>::quiet_NaN();
	values[1] = std::numeric_limits<oa::F32>::infinity();
	values[2] = -std::numeric_limits<oa::F32>::infinity();
	values[3] = -7.0F;
	values[32] = 7.0e-9F;

	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto scales = read<oa::F32>(scale);
	ASSERT_EQ(scales.size(), 2U);
	EXPECT_FLOAT_EQ(scales[0], 1.0F);
	EXPECT_FLOAT_EQ(scales[1], 1.0e-9F);

	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ4(
		quantized, scale, static_cast<oa::I64>(values.size())));
	ASSERT_EQ(output.size(), values.size());
	EXPECT_FLOAT_EQ(output[0], 0.0F);
	EXPECT_FLOAT_EQ(output[1], 0.0F);
	EXPECT_FLOAT_EQ(output[2], 0.0F);
	EXPECT_FLOAT_EQ(output[3], -7.0F);
	EXPECT_NEAR(output[32], 7.0e-9F, 1.0e-14F);
	for (const oa::F32 value : output) EXPECT_TRUE(std::isfinite(value));
}

TEST(FnMatrixQuant, InvalidScaleValuesProduceDeterministicZeroBlocks) {
	const auto input = oa::FnMatrix::ones(
		oa::MatrixShape{96}, oa::ScalarType::Float32);
	auto scale = oa::FnMatrix::empty(
		oa::MatrixShape{3}, oa::ScalarType::Float32);
	scale.set(0, std::numeric_limits<oa::F32>::quiet_NaN());
	scale.set(1, 0.0F);
	scale.set(2, -1.0F);

	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto payload = read<oa::U8>(quantized);
	EXPECT_EQ(payload, (std::vector<oa::U8>(48, 0x77U)));

	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ4(
		quantized, scale, 96));
	EXPECT_EQ(output, (std::vector<oa::F32>(96, 0.0F)));
}

TEST(FnMatrixQuant, RepeatedQuantizationIsByteDeterministic) {
	std::vector<oa::F32> values(129);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(values.size()); ++i) {
		values[static_cast<std::size_t>(i)] =
			std::cos(static_cast<oa::F32>(i) * 0.031F) * 20.0F;
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto first = oa::FnMatrix::quantizeQ4(input, scale);
	const auto second = oa::FnMatrix::quantizeQ4(input, scale);
	EXPECT_EQ(read<oa::U8>(first), read<oa::U8>(second));
}

TEST(FnMatrixQuant, EmptyInputProducesEmptyPlanesWithoutDispatch) {
	const auto input = oa::FnMatrix::empty(
		oa::MatrixShape{0}, oa::ScalarType::Float32);
	const auto scale = oa::FnMatrix::computeScaleQ4(input);
	const auto quantized = oa::FnMatrix::quantizeQ4(input, scale);
	const auto output = oa::FnMatrix::dequantizeQ4(quantized, scale, 0);
	EXPECT_EQ(scale.numElements(), 0);
	EXPECT_EQ(quantized.numElements(), 0);
	EXPECT_EQ(output.numElements(), 0);
}

TEST(FnMatrixQuant, InvalidDtypesAndShapesFailClosed) {
	const auto fp32 = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::Float32);
	const auto byte = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::UInt8);
	const auto scale = oa::FnMatrix::ones(
		oa::MatrixShape{1}, oa::ScalarType::Float32);
	const auto badScale = oa::FnMatrix::ones(
		oa::MatrixShape{2}, oa::ScalarType::Float32);

	EXPECT_TRUE(oa::FnMatrix::computeScaleQ4(byte).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::quantizeQ4(byte, scale).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::quantizeQ4(fp32, badScale).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ4(byte, scale, 33).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ4(byte, scale, -1).isEmpty());
	constexpr oa::I64 tooManyFloat32Elements =
		static_cast<oa::I64>(std::numeric_limits<oa::U32>::max() / sizeof(oa::F32)) + 1;
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ4(
		byte, scale, tooManyFloat32Elements).isEmpty());
}

TEST(FnMatrixQuant, Q8ComputeScaleOwnsEveryBlockAndPartialTail) {
	std::vector<oa::F32> values(73, 0.0F);
	values[0] = -127.0F;
	values[33] = 254.0F;
	values[72] = -381.0F;

	const auto scales = read<oa::F32>(
		oa::FnMatrix::computeScaleQ8(makeInput(values)));
	ASSERT_EQ(scales.size(), 3U);
	EXPECT_FLOAT_EQ(scales[0], 1.0F);
	EXPECT_FLOAT_EQ(scales[1], 2.0F);
	EXPECT_FLOAT_EQ(scales[2], 3.0F);
}

TEST(FnMatrixQuant, Q8PayloadMatchesIndependentAlignedOracle) {
	std::vector<oa::F32> values(50);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(values.size()); ++i) {
		values[static_cast<std::size_t>(i)] =
			static_cast<oa::F32>((i * 37) % 251 - 125);
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	const auto scales = read<oa::F32>(scale);
	const auto payload = read<oa::I8>(quantized);

	ASSERT_EQ(payload.size(), 64U);
	EXPECT_EQ(payload, referenceQ8Payload(values, scales));
	for (std::size_t i = values.size(); i < payload.size(); ++i) {
		EXPECT_EQ(payload[i], 0) << "padding byte " << i;
	}
}

TEST(FnMatrixQuant, Q8OddSizesAndPartialBlocksRoundTripWithinHalfStep) {
	for (const oa::I64 count : {1, 7, 31, 32, 33, 50, 257}) {
		std::vector<oa::F32> values(static_cast<std::size_t>(count));
		for (oa::I64 i = 0; i < count; ++i) {
			values[static_cast<std::size_t>(i)] =
				std::sin(static_cast<oa::F32>(i) * 0.17F) * 130.0F;
		}
		expectQ8RoundTrip(values);
	}
}

TEST(FnMatrixQuant, Q8SignAndExtremeValuesArePreserved) {
	std::vector<oa::F32> values(32);
	for (oa::I64 i = 0; i < 32; ++i) {
		values[static_cast<std::size_t>(i)] = i % 2 == 0 ? -127.0F : 127.0F;
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ8(
		quantized, scale, static_cast<oa::I64>(values.size())));
	EXPECT_EQ(output, values);
}

TEST(FnMatrixQuant, Q8NonFiniteValuesEncodeAsZeroAndDoNotPoisonScale) {
	std::vector<oa::F32> values(64, 0.0F);
	values[0] = std::numeric_limits<oa::F32>::quiet_NaN();
	values[1] = std::numeric_limits<oa::F32>::infinity();
	values[2] = -std::numeric_limits<oa::F32>::infinity();
	values[3] = -127.0F;
	values[32] = 127.0e-9F;

	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto scales = read<oa::F32>(scale);
	ASSERT_EQ(scales.size(), 2U);
	EXPECT_FLOAT_EQ(scales[0], 1.0F);
	EXPECT_FLOAT_EQ(scales[1], 1.0e-9F);

	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ8(
		quantized, scale, static_cast<oa::I64>(values.size())));
	ASSERT_EQ(output.size(), values.size());
	EXPECT_FLOAT_EQ(output[0], 0.0F);
	EXPECT_FLOAT_EQ(output[1], 0.0F);
	EXPECT_FLOAT_EQ(output[2], 0.0F);
	EXPECT_FLOAT_EQ(output[3], -127.0F);
	EXPECT_NEAR(output[32], 127.0e-9F, 1.0e-12F);
	for (const oa::F32 value : output) EXPECT_TRUE(std::isfinite(value));
}

TEST(FnMatrixQuant, Q8InvalidScaleValuesProduceDeterministicZeroBlocks) {
	const auto input = oa::FnMatrix::ones(
		oa::MatrixShape{96}, oa::ScalarType::Float32);
	auto scale = oa::FnMatrix::empty(
		oa::MatrixShape{3}, oa::ScalarType::Float32);
	scale.set(0, std::numeric_limits<oa::F32>::quiet_NaN());
	scale.set(1, 0.0F);
	scale.set(2, -1.0F);

	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	EXPECT_EQ(read<oa::I8>(quantized), (std::vector<oa::I8>(96, 0)));

	const auto output = read<oa::F32>(oa::FnMatrix::dequantizeQ8(
		quantized, scale, 96));
	EXPECT_EQ(output, (std::vector<oa::F32>(96, 0.0F)));
}

TEST(FnMatrixQuant, Q8RepeatedQuantizationIsByteDeterministic) {
	std::vector<oa::F32> values(129);
	for (oa::I64 i = 0; i < static_cast<oa::I64>(values.size()); ++i) {
		values[static_cast<std::size_t>(i)] =
			std::cos(static_cast<oa::F32>(i) * 0.031F) * 200.0F;
	}
	const auto input = makeInput(values);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto first = oa::FnMatrix::quantizeQ8(input, scale);
	const auto second = oa::FnMatrix::quantizeQ8(input, scale);
	EXPECT_EQ(read<oa::I8>(first), read<oa::I8>(second));
}

TEST(FnMatrixQuant, Q8EmptyInputProducesEmptyPlanesWithoutDispatch) {
	const auto input = oa::FnMatrix::empty(
		oa::MatrixShape{0}, oa::ScalarType::Float32);
	const auto scale = oa::FnMatrix::computeScaleQ8(input);
	const auto quantized = oa::FnMatrix::quantizeQ8(input, scale);
	const auto output = oa::FnMatrix::dequantizeQ8(quantized, scale, 0);
	EXPECT_EQ(scale.numElements(), 0);
	EXPECT_EQ(quantized.numElements(), 0);
	EXPECT_EQ(output.numElements(), 0);
	EXPECT_EQ(quantized.getDtype(), oa::ScalarType::Int8);
}

TEST(FnMatrixQuant, Q8InvalidDtypesAndShapesFailClosed) {
	const auto fp32 = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::Float32);
	const auto signedByte = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::Int8);
	const auto unsignedByte = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::UInt8);
	const auto scale = oa::FnMatrix::ones(
		oa::MatrixShape{1}, oa::ScalarType::Float32);
	const auto badScale = oa::FnMatrix::ones(
		oa::MatrixShape{2}, oa::ScalarType::Float32);

	EXPECT_TRUE(oa::FnMatrix::computeScaleQ8(unsignedByte).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::quantizeQ8(signedByte, scale).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::quantizeQ8(fp32, badScale).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ8(unsignedByte, scale, 32).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ8(signedByte, scale, 33).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ8(signedByte, scale, -1).isEmpty());
	constexpr oa::I64 tooManyFloat32Elements =
		static_cast<oa::I64>(std::numeric_limits<oa::U32>::max() / sizeof(oa::F32)) + 1;
	EXPECT_TRUE(oa::FnMatrix::dequantizeQ8(
		signedByte, scale, tooManyFloat32Elements).isEmpty());
}

TEST(FnMatrixQuant, MatMulNtQ4FusesDequantizationAcrossOddWeightRowsAndKTail) {
	constexpr oa::I64 rows = 6;
	constexpr oa::I64 outputFeatures = 5;
	constexpr oa::I64 inputFeatures = 35;
	std::vector<oa::F32> input(static_cast<std::size_t>(rows * inputFeatures));
	std::vector<oa::F32> weight(
		static_cast<std::size_t>(outputFeatures * inputFeatures));
	for (oa::I64 i = 0; i < static_cast<oa::I64>(input.size()); ++i) {
		input[static_cast<std::size_t>(i)] =
			std::sin(static_cast<oa::F32>(i) * 0.13F) * 2.0F;
	}
	for (oa::I64 i = 0; i < static_cast<oa::I64>(weight.size()); ++i) {
		weight[static_cast<std::size_t>(i)] =
			std::cos(static_cast<oa::F32>(i) * 0.071F) * 5.0F;
	}
	const auto scales = referenceScales(weight, 7.0F);
	const auto payload = referencePayload(weight, scales);
	const auto expected = referenceMatMulNtQ4(
		input, payload, scales, rows, outputFeatures, inputFeatures);

	const auto output = oa::FnMatrix::matMulNtQ4(
		makeRawMatrix(input, oa::MatrixShape{2, 3, inputFeatures},
			oa::ScalarType::Float32),
		makeRawMatrix(payload,
			oa::MatrixShape{static_cast<oa::I64>(payload.size())},
			oa::ScalarType::UInt8),
		makeRawMatrix(scales,
			oa::MatrixShape{static_cast<oa::I64>(scales.size())},
			oa::ScalarType::Float32),
		outputFeatures);
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 3, outputFeatures}));
	expectNearVectors(read<oa::F32>(output), expected);
}

TEST(FnMatrixQuant, MatMulNtQ8FusesDequantizationAcrossOddWeightRowsAndKTail) {
	constexpr oa::I64 rows = 3;
	constexpr oa::I64 outputFeatures = 7;
	constexpr oa::I64 inputFeatures = 33;
	std::vector<oa::F32> input(static_cast<std::size_t>(rows * inputFeatures));
	std::vector<oa::F32> weight(
		static_cast<std::size_t>(outputFeatures * inputFeatures));
	for (oa::I64 i = 0; i < static_cast<oa::I64>(input.size()); ++i) {
		input[static_cast<std::size_t>(i)] =
			std::cos(static_cast<oa::F32>(i) * 0.19F) * 1.5F;
	}
	for (oa::I64 i = 0; i < static_cast<oa::I64>(weight.size()); ++i) {
		weight[static_cast<std::size_t>(i)] =
			std::sin(static_cast<oa::F32>(i) * 0.047F) * 8.0F;
	}
	const auto scales = referenceScales(weight, 127.0F);
	const auto payload = referenceQ8Payload(weight, scales);
	const auto expected = referenceMatMulNtQ8(
		input, payload, scales, rows, outputFeatures, inputFeatures);

	const auto output = oa::FnMatrix::matMulNtQ8(
		makeRawMatrix(input, oa::MatrixShape{rows, inputFeatures},
			oa::ScalarType::Float32),
		makeRawMatrix(payload,
			oa::MatrixShape{static_cast<oa::I64>(payload.size())},
			oa::ScalarType::Int8),
		makeRawMatrix(scales,
			oa::MatrixShape{static_cast<oa::I64>(scales.size())},
			oa::ScalarType::Float32),
		outputFeatures);
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{rows, outputFeatures}));
	expectNearVectors(read<oa::F32>(output), expected);
}

TEST(FnMatrixQuant, QuantizedMatMulPipelineConsumesGpuProducedPlanes) {
	constexpr oa::I64 rows = 2;
	constexpr oa::I64 outputFeatures = 3;
	constexpr oa::I64 inputFeatures = 37;
	std::vector<oa::F32> input(static_cast<std::size_t>(rows * inputFeatures));
	std::vector<oa::F32> weight(
		static_cast<std::size_t>(outputFeatures * inputFeatures));
	for (oa::I64 i = 0; i < static_cast<oa::I64>(input.size()); ++i) {
		input[static_cast<std::size_t>(i)] =
			static_cast<oa::F32>((i * 7) % 17 - 8) * 0.25F;
	}
	for (oa::I64 i = 0; i < static_cast<oa::I64>(weight.size()); ++i) {
		weight[static_cast<std::size_t>(i)] =
			static_cast<oa::F32>((i * 13) % 31 - 15) * 0.2F;
	}
	const auto inputMatrix = makeRawMatrix(
		input, oa::MatrixShape{rows, inputFeatures}, oa::ScalarType::Float32);
	const auto weightMatrix = makeRawMatrix(
		weight, oa::MatrixShape{outputFeatures, inputFeatures},
		oa::ScalarType::Float32);

	const auto q4Scale = oa::FnMatrix::computeScaleQ4(weightMatrix);
	const auto q4Payload = oa::FnMatrix::quantizeQ4(weightMatrix, q4Scale);
	const auto q4Expected = referenceMatMulNtQ4(
		input, read<oa::U8>(q4Payload), read<oa::F32>(q4Scale),
		rows, outputFeatures, inputFeatures);
	expectNearVectors(
		read<oa::F32>(oa::FnMatrix::matMulNtQ4(
			inputMatrix, q4Payload, q4Scale, outputFeatures)),
		q4Expected);

	const auto q8Scale = oa::FnMatrix::computeScaleQ8(weightMatrix);
	const auto q8Payload = oa::FnMatrix::quantizeQ8(weightMatrix, q8Scale);
	const auto q8Expected = referenceMatMulNtQ8(
		input, read<oa::I8>(q8Payload), read<oa::F32>(q8Scale),
		rows, outputFeatures, inputFeatures);
	expectNearVectors(
		read<oa::F32>(oa::FnMatrix::matMulNtQ8(
			inputMatrix, q8Payload, q8Scale, outputFeatures)),
		q8Expected);
}

TEST(FnMatrixQuant, QuantizedMatMulCaptureMatchesEagerOracleAndReplays) {
	constexpr oa::I64 rows = 2;
	constexpr oa::I64 outputFeatures = 3;
	constexpr oa::I64 inputFeatures = 35;
	std::vector<oa::F32> input(static_cast<std::size_t>(rows * inputFeatures));
	std::vector<oa::F32> weight(
		static_cast<std::size_t>(outputFeatures * inputFeatures));
	for (oa::I64 i = 0; i < static_cast<oa::I64>(input.size()); ++i) {
		input[static_cast<std::size_t>(i)] =
			std::sin(static_cast<oa::F32>(i) * 0.11F);
	}
	for (oa::I64 i = 0; i < static_cast<oa::I64>(weight.size()); ++i) {
		weight[static_cast<std::size_t>(i)] =
			std::cos(static_cast<oa::F32>(i) * 0.07F) * 4.0F;
	}
	const auto q4Scales = referenceScales(weight, 7.0F);
	const auto q4PayloadValues = referencePayload(weight, q4Scales);
	const auto q8Scales = referenceScales(weight, 127.0F);
	const auto q8PayloadValues = referenceQ8Payload(weight, q8Scales);
	const auto q4Expected = referenceMatMulNtQ4(
		input, q4PayloadValues, q4Scales,
		rows, outputFeatures, inputFeatures);
	const auto q8Expected = referenceMatMulNtQ8(
		input, q8PayloadValues, q8Scales,
		rows, outputFeatures, inputFeatures);

	const auto inputMatrix = makeRawMatrix(
		input, oa::MatrixShape{rows, inputFeatures}, oa::ScalarType::Float32);
	const auto q4Payload = makeRawMatrix(
		q4PayloadValues,
		oa::MatrixShape{static_cast<oa::I64>(q4PayloadValues.size())},
		oa::ScalarType::UInt8);
	const auto q4Scale = makeRawMatrix(
		q4Scales, oa::MatrixShape{static_cast<oa::I64>(q4Scales.size())},
		oa::ScalarType::Float32);
	const auto q8Payload = makeRawMatrix(
		q8PayloadValues,
		oa::MatrixShape{static_cast<oa::I64>(q8PayloadValues.size())},
		oa::ScalarType::Int8);
	const auto q8Scale = makeRawMatrix(
		q8Scales, oa::MatrixShape{static_cast<oa::I64>(q8Scales.size())},
		oa::ScalarType::Float32);

	// Materialize the upload boundary before capturing an execution-only plan.
	ASSERT_EQ(read<oa::F32>(inputMatrix), input);
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto& defaultContext = oa::ExecutionSession::getActive();
	defaultContext.clear();

	oa::Matrix q4Output;
	oa::Matrix q8Output;
	auto captured = engine->capture([&]() {
		q4Output = oa::FnMatrix::matMulNtQ4(
			inputMatrix, q4Payload, q4Scale, outputFeatures);
		q8Output = oa::FnMatrix::matMulNtQ8(
			inputMatrix, q8Payload, q8Scale, outputFeatures);
	});
	ASSERT_TRUE(captured.isOk()) << captured.getStatus().getMessage();
	auto plan = oa::move(captured).getValue();
	ASSERT_TRUE(plan.isCompiled());
	EXPECT_EQ(plan.nodeCount(), 2U);
	EXPECT_EQ(plan.dnnSourceOpCount(), 2U);
	EXPECT_EQ(plan.dnnCapturedOpCount(), 2U);

	for (oa::I32 replay = 0; replay < 2; ++replay) {
		auto submitted = engine->submit(plan);
		ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().getMessage();
		ASSERT_TRUE(engine->wait(submitted.getValue()).isOk());
		expectNearVectors(read<oa::F32>(q4Output), q4Expected);
		expectNearVectors(read<oa::F32>(q8Output), q8Expected);
	}
	defaultContext.clear();
}

TEST(FnMatrixQuant, QuantizedMatMulInvalidContractsFailClosed) {
	const auto input = oa::FnMatrix::ones(
		oa::MatrixShape{2, 33}, oa::ScalarType::Float32);
	const auto inputByte = oa::FnMatrix::ones(
		oa::MatrixShape{2, 33}, oa::ScalarType::UInt8);
	const auto q4Payload = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::UInt8);
	const auto q8Payload = oa::FnMatrix::zeros(
		oa::MatrixShape{64}, oa::ScalarType::Int8);
	const auto scale = oa::FnMatrix::ones(
		oa::MatrixShape{2}, oa::ScalarType::Float32);
	const auto badScale = oa::FnMatrix::ones(
		oa::MatrixShape{3}, oa::ScalarType::Float32);

	EXPECT_TRUE(oa::FnMatrix::matMulNtQ4(
		inputByte, q4Payload, scale, 1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ4(
		input, q8Payload, scale, 1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ8(
		input, q4Payload, scale, 1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ4(
		input, q4Payload, badScale, 1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ8(
		input, q8Payload, badScale, 1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ4(
		input, q4Payload, scale, -1).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNtQ8(
		oa::FnMatrix::ones(oa::MatrixShape{33}, oa::ScalarType::Float32),
		q8Payload, scale, 1).isEmpty());
}

TEST(FnMatrixQuant, QuantizedMatMulZeroRowsAndColumnsDoNotDispatch) {
	const auto emptyRows = oa::FnMatrix::empty(
		oa::MatrixShape{0, 7}, oa::ScalarType::Float32);
	const auto q4Payload = oa::FnMatrix::zeros(
		oa::MatrixShape{32}, oa::ScalarType::UInt8);
	const auto scale = oa::FnMatrix::ones(
		oa::MatrixShape{2}, oa::ScalarType::Float32);
	const auto rowsOutput = oa::FnMatrix::matMulNtQ4(
		emptyRows, q4Payload, scale, 5);
	EXPECT_EQ(rowsOutput.getShape(), (oa::MatrixShape{0, 5}));
	EXPECT_EQ(rowsOutput.numElements(), 0);

	const auto input = oa::FnMatrix::ones(
		oa::MatrixShape{3, 7}, oa::ScalarType::Float32);
	const auto emptyQ8 = oa::FnMatrix::empty(
		oa::MatrixShape{0}, oa::ScalarType::Int8);
	const auto emptyScale = oa::FnMatrix::empty(
		oa::MatrixShape{0}, oa::ScalarType::Float32);
	const auto columnsOutput = oa::FnMatrix::matMulNtQ8(
		input, emptyQ8, emptyScale, 0);
	EXPECT_EQ(columnsOutput.getShape(), (oa::MatrixShape{3, 0}));
	EXPECT_EQ(columnsOutput.numElements(), 0);
}

TEST(FnMatrixQuant, SemanticQuantizedValueOwnsFormatShapeAndPrivatePlanes) {
	const std::vector<oa::F32> values = {
		-3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F,
	};
	const auto input = makeRawMatrix(
		values, oa::MatrixShape{2, 3}, oa::ScalarType::Float32);

	const auto q4 = oa::FnMatrix::quantize(input, oa::Quantization::Q4);
	ASSERT_FALSE(q4.isEmpty());
	EXPECT_EQ(q4.getQuantization(), oa::Quantization::Q4);
	EXPECT_EQ(q4.getShape(), (oa::MatrixShape{2, 3}));
	EXPECT_EQ(q4.numElements(), 6);
	EXPECT_EQ(oa::QuantMatrixAccess::payload(q4).getDtype(),
		oa::ScalarType::UInt8);
	EXPECT_EQ(oa::QuantMatrixAccess::scale(q4).getDtype(),
		oa::ScalarType::Float32);

	const auto restored = oa::FnMatrix::dequantize(q4);
	ASSERT_EQ(restored.getShape(), input.getShape());
	expectNearVectors(read<oa::F32>(restored), values, 0.45F);
}

TEST(FnMatrixQuant, SemanticQuantizedOperationsOwnLogicalGraphContracts) {
	const auto weight = makeRawMatrix(
		std::vector<oa::F32>{
			0.5F, 1.0F, -0.25F,
			-1.0F, 0.75F, 0.5F,
		},
		oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	const auto input = makeRawMatrix(
		std::vector<oa::F32>{1.0F, -2.0F, 0.5F},
		oa::MatrixShape{1, 3}, oa::ScalarType::Float32);
	auto& context = oa::ExecutionSession::getActive();
	context.clear();

	const auto quantized = oa::FnMatrix::quantize(weight, oa::Quantization::Q4);
	ASSERT_FALSE(quantized.isEmpty());
	const auto expanded = oa::FnMatrix::dequantize(quantized);
	ASSERT_FALSE(expanded.isEmpty());
	const auto output = oa::FnMatrix::matMulNt(input, quantized);
	ASSERT_FALSE(output.isEmpty());

	const auto* semantic = context.semanticGraph();
	const auto* executable = context.graph();
	ASSERT_NE(semantic, nullptr);
	ASSERT_NE(executable, nullptr);
	ASSERT_TRUE(semantic->validate().isOk());
	ASSERT_TRUE(oa::validateSemanticLowering(*semantic, *executable).isOk());
	ASSERT_EQ(semantic->operationCount(), 3U);

	const auto operations = semantic->operations();
	EXPECT_EQ(operations[0].name,
		oa::detail::opRegistry::FnMatrix::quantize.name);
	EXPECT_EQ(operations[1].name,
		oa::detail::opRegistry::FnMatrix::dequantize.name);
	EXPECT_EQ(operations[2].name,
		oa::detail::opRegistry::FnMatrix::matMulNtQuantized.name);
	for (const auto& operation : operations) {
		ASSERT_EQ(operation.attributes.size(), 1U);
		EXPECT_EQ(operation.attributes[0].name, "quantization");
		EXPECT_EQ(operation.attributes[0].kind, oa::OpAttributeKind::Enum);
		EXPECT_EQ(operation.attributes[0].text, "Q4");
	}

	ASSERT_EQ(operations[0].outputs.size(), 1U);
	ASSERT_EQ(operations[1].inputs.size(), 1U);
	ASSERT_EQ(operations[2].inputs.size(), 2U);
	EXPECT_EQ(operations[1].inputs[0], operations[0].outputs[0]);
	EXPECT_EQ(operations[2].inputs[1], operations[0].outputs[0]);
	const auto* quantizedValue = semantic->findValue(operations[0].outputs[0]);
	ASSERT_NE(quantizedValue, nullptr);
	EXPECT_EQ(quantizedValue->kind, oa::OpValueKind::QuantMatrix);
	EXPECT_EQ(quantizedValue->shape, (oa::MatrixShape{2, 3}));
	EXPECT_EQ(quantizedValue->dtype, oa::ScalarType::Float32);

	context.clear();
}

TEST(FnMatrixQuant, SemanticQuantizedMatMulSelectsQ4AndQ8WithoutExpansion) {
	const std::vector<oa::F32> inputValues = {
		1.0F, -2.0F, 0.5F,
		-1.0F, 0.25F, 3.0F,
	};
	const std::vector<oa::F32> weightValues = {
		0.5F, 1.0F, -0.25F,
		-1.0F, 0.75F, 0.5F,
	};
	const auto input = makeRawMatrix(
		inputValues, oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	const auto weight = makeRawMatrix(
		weightValues, oa::MatrixShape{2, 3}, oa::ScalarType::Float32);

	for (const auto quantization : {oa::Quantization::Q4, oa::Quantization::Q8}) {
		const auto quantized = oa::FnMatrix::quantize(weight, quantization);
		ASSERT_FALSE(quantized.isEmpty());
		const auto expanded = oa::FnMatrix::dequantize(quantized);
		const auto expected = oa::FnMatrix::matMulNt(input, expanded);
		const auto actual = oa::FnMatrix::matMulNt(input, quantized);
		ASSERT_EQ(actual.getShape(), (oa::MatrixShape{2, 2}));
		expectNearVectors(read<oa::F32>(actual), read<oa::F32>(expected), 2e-4F);
	}
}

TEST(FnMatrixQuant, SemanticQuantizedSurfaceFailsClosed) {
	const auto fp32 = oa::FnMatrix::ones(
		oa::MatrixShape{2, 3}, oa::ScalarType::Float32);
	const auto byte = oa::FnMatrix::ones(
		oa::MatrixShape{2, 3}, oa::ScalarType::UInt8);

	EXPECT_TRUE(oa::FnMatrix::quantize(byte, oa::Quantization::Q4).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::quantize(
		fp32, static_cast<oa::Quantization>(255)).isEmpty());
	EXPECT_TRUE(oa::FnMatrix::dequantize(oa::QuantMatrix{}).isEmpty());

	const auto weight = oa::FnMatrix::quantize(fp32, oa::Quantization::Q8);
	ASSERT_FALSE(weight.isEmpty());
	EXPECT_TRUE(oa::FnMatrix::matMulNt(
		oa::FnMatrix::ones(oa::MatrixShape{2, 4}, oa::ScalarType::Float32),
		weight).isEmpty());
}

TEST(FnMatrixQuant, ModelFileV3LoadsTheSameSemanticWeightUsedByFusedMatMul) {
	const std::vector<oa::F32> weightValues = {
		0.5F, 1.0F, -0.25F,
		-1.0F, 0.75F, 0.5F,
	};
	const oa::U64 shape[] = {2, 3};
	oa::ModelFile dense;
	dense.addWeight("linear.weight", oa::ScalarType::Float32, {shape, 2},
		weightValues.data(), weightValues.size() * sizeof(oa::F32));
	auto encoded = dense.quantizeWeights(oa::Quantization::Q8);
	ASSERT_TRUE(encoded.isOk()) << encoded.getStatus().toString().cStr();
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	auto loaded = encoded->loadQuantMatrix(*engine, "linear.weight");
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString().cStr();
	EXPECT_EQ(loaded->getQuantization(), oa::Quantization::Q8);
	EXPECT_EQ(loaded->getShape(), (oa::MatrixShape{2, 3}));

	const auto input = makeRawMatrix(
		std::vector<oa::F32>{1.0F, -2.0F, 0.5F},
		oa::MatrixShape{1, 3}, oa::ScalarType::Float32);
	const auto actual = oa::FnMatrix::matMulNt(input, loaded.getValue());
	const auto expanded = oa::FnMatrix::dequantize(loaded.getValue());
	const auto expected = oa::FnMatrix::matMulNt(input, expanded);
	expectNearVectors(read<oa::F32>(actual), read<oa::F32>(expected), 2e-4F);
}
