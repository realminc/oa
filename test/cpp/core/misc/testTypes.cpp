// ═══════════════════════════════════════════════════════════════════════════════
// Core Types + status + timestamp Tests
// ═══════════════════════════════════════════════════════════════════════════════

#include "../../oaTest.h"

static_assert(oa::scalarTypeToString(oa::ScalarType::Float32)[0] == 'F');

// ─── scalar Types ──────────────────────────────────────────────────────────────

TEST(CoreTypes, ScalarSizes) {
	EXPECT_EQ(sizeof(oa::I8), 1u);
	EXPECT_EQ(sizeof(oa::I16), 2u);
	EXPECT_EQ(sizeof(oa::I32), 4u);
	EXPECT_EQ(sizeof(oa::I64), 8u);
	EXPECT_EQ(sizeof(oa::U8), 1u);
	EXPECT_EQ(sizeof(oa::U32), 4u);
	EXPECT_EQ(sizeof(oa::U64), 8u);
	EXPECT_EQ(sizeof(oa::F32), 4u);
	EXPECT_EQ(sizeof(oa::F64), 8u);
}

TEST(CoreTypes, ScalarTypeEnum) {
	EXPECT_EQ(oa::scalarSize(oa::ScalarType::Float32), 4u);
	EXPECT_EQ(oa::scalarSize(oa::ScalarType::Float64), 8u);
	EXPECT_EQ(oa::scalarSize(oa::ScalarType::Int8), 1u);
	EXPECT_EQ(oa::scalarSize(oa::ScalarType::BFloat16), 2u);
}

TEST(CoreTypes, GeneratedEnumConversionsAreHeaderComplete) {
	EXPECT_STREQ(oa::scalarTypeToString(oa::ScalarType::BFloat16), "BFloat16");
	EXPECT_STREQ(
		oa::determinismModeToString(oa::DeterminismMode::Deterministic),
		"Deterministic");
	EXPECT_STREQ(
		oa::matMulPrecisionToString(oa::MatMulPrecision::Bf16), "Bf16");
	EXPECT_STREQ(oa::precisionToString(oa::Precision::FP64), "FP64");
	EXPECT_EQ(oa::precisionFromString("BF16"), oa::Precision::BF16);
	EXPECT_EQ(oa::precisionDtype(oa::Precision::FP32), oa::ScalarType::Float32);
	EXPECT_EQ(oa::precisionDtype(oa::Precision::BF16), oa::ScalarType::BFloat16);
	EXPECT_EQ(oa::precisionDtype(oa::Precision::FP64), oa::ScalarType::Float64);
	EXPECT_STREQ(
		oa::filterToString(static_cast<oa::Filter>(255)), "unknown");
	EXPECT_EQ(oa::scalarTypeFromString("Int8"), oa::ScalarType::Int8);
	EXPECT_EQ(oa::scalarTypeFromString("invalid"), oa::ScalarType::Float32);
	EXPECT_EQ(oa::scalarTypeFromString(nullptr), oa::ScalarType::Float32);
}

TEST(CoreTypes, GeneratedPodDefaultsAreInitialized) {
	const oa::PrecisionConfig config{};
	EXPECT_EQ(config.mode, oa::Precision::FP32);
	EXPECT_EQ(config.determinism, oa::DeterminismMode::Stable);
}

// ─── Device ────────────────────────────────────────────────────────────────────

TEST(CoreTypes, Device) {
	oa::Device cpu = oa::HostDevice;
	EXPECT_TRUE(cpu.isHost());
	EXPECT_FALSE(cpu.isVulkan());
	oa::Device vk{oa::DeviceType::VkDiscrete, 0};
	EXPECT_TRUE(vk.isVulkan());
	EXPECT_TRUE(vk.isGpu());
	EXPECT_EQ(vk.index, 0);
}

// ─── status ────────────────────────────────────────────────────────────────────

TEST(CoreStatus, ok) {
	oa::Status s = oa::Status::ok();
	EXPECT_TRUE(s.isOk());
	EXPECT_FALSE(s.isError());
}

TEST(CoreStatus, Error) {
	oa::Status s = oa::Status::error(oa::StatusCode::InvalidArgument, "test error");
	EXPECT_FALSE(s.isOk());
	EXPECT_TRUE(s.isError());
	EXPECT_EQ(s.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_EQ(s.getMessage(), "test error");
}

TEST(CoreStatus, ResultOk) {
	oa::Result<oa::I32> r = 42;
	EXPECT_TRUE(r.isOk());
	EXPECT_EQ(r.getValue(), 42);
}

TEST(CoreStatus, ResultError) {
	oa::Result<oa::I32> r = oa::Status::notFound("not found");
	EXPECT_FALSE(r.isOk());
	EXPECT_EQ(r.getStatus().getCode(), oa::StatusCode::NotFound);
}

TEST(CoreTimestamp, FromSeconds) {
	oa::Timestamp ts = oa::Timestamp::fromSeconds(1000);
	EXPECT_EQ(ts.secs(), 1000);
	EXPECT_EQ(ts.millis(), 1000000);
}

TEST(CoreTimestamp, Duration) {
	oa::Timestamp d = oa::Timestamp::fromSeconds(300);
	EXPECT_EQ(d.secs(), 300);
	oa::Timestamp h = oa::Timestamp::fromSeconds(7200);
	EXPECT_EQ(h.secs(), 7200);
}

// ─── Fixed Point ───────────────────────────────────────────────────────────────

TEST(CoreFixed, Construction) {
	oa::Fixed<8> zero = oa::Fixed<8>::zero();
	EXPECT_EQ(zero.raw, 0);
	oa::Fixed<8> one = oa::Fixed<8>::one();
	EXPECT_EQ(one.raw, 100000000);
}

TEST(CoreFixed, Arithmetic) {
	oa::Price a = oa::Price::fromDouble(100.5);
	oa::Price b = oa::Price::fromDouble(50.25);
	oa::Price c = a + b;
	EXPECT_EQ(c.raw, oa::Price::fromDouble(150.75).raw);
}

// ─── Safe Arithmetic ───────────────────────────────────────────────────────────

TEST(CoreSafe, SafeAdd) {
	oa::U64 result;
	EXPECT_TRUE(oa::safeAdd(oa::U64(100), oa::U64(200), result));
	EXPECT_EQ(result, 300u);
	EXPECT_FALSE(oa::safeAdd(oa::U64Max, oa::U64(1), result));
}

TEST(CoreSafe, SafeAddClamped) {
	EXPECT_EQ(oa::safeAddClamped(oa::U64(100), oa::U64(200)), 300u);
	EXPECT_EQ(oa::safeAddClamped(oa::U64Max, oa::U64(1)), oa::U64Max);
}

TEST(CoreSafe, ByteSwap) {
	EXPECT_EQ(oa::byteSwap16(oa::U16{0x1234}), oa::U16{0x3412});
	EXPECT_EQ(oa::byteSwap32(oa::U32{0x12345678}), oa::U32{0x78563412});
	EXPECT_EQ(oa::byteSwap64(oa::U64{0x0123456789ABCDEF}), oa::U64{0xEFCDAB8967452301});
}

TEST(CoreShape, AcceptsMaximumRank) {
	const oa::MatrixShape shape({1, 2, 3, 4, 5, 6, 7, 8});
	EXPECT_EQ(shape.rank, OA_MAX_TENSOR_DIMS);
	EXPECT_EQ(shape[0], 1);
	EXPECT_EQ(shape[OA_MAX_TENSOR_DIMS - 1], 8);
}

TEST(CoreShape, RejectsRankBeyondStorage) {
	EXPECT_THROW(
		static_cast<void>(oa::MatrixShape({1, 2, 3, 4, 5, 6, 7, 8, 9})),
		std::length_error);
}
