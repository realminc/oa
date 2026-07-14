// ═══════════════════════════════════════════════════════════════════════════════
// Core Types + Status + Timestamp Tests
// ═══════════════════════════════════════════════════════════════════════════════

#include "../../OaTest.h"

// ─── Scalar Types ──────────────────────────────────────────────────────────────

TEST(CoreTypes, ScalarSizes) {
	EXPECT_EQ(sizeof(OaI8), 1u);
	EXPECT_EQ(sizeof(OaI16), 2u);
	EXPECT_EQ(sizeof(OaI32), 4u);
	EXPECT_EQ(sizeof(OaI64), 8u);
	EXPECT_EQ(sizeof(OaU8), 1u);
	EXPECT_EQ(sizeof(OaU32), 4u);
	EXPECT_EQ(sizeof(OaU64), 8u);
	EXPECT_EQ(sizeof(OaF32), 4u);
	EXPECT_EQ(sizeof(OaF64), 8u);
}

TEST(CoreTypes, ScalarTypeEnum) {
	EXPECT_EQ(OaScalarSize(OaScalarType::Float32), 4u);
	EXPECT_EQ(OaScalarSize(OaScalarType::Float64), 8u);
	EXPECT_EQ(OaScalarSize(OaScalarType::Int8), 1u);
	EXPECT_EQ(OaScalarSize(OaScalarType::BFloat16), 2u);
}

// ─── Device ────────────────────────────────────────────────────────────────────

TEST(CoreTypes, Device) {
	OaDevice cpu = OA_DEVICE_HOST;
	EXPECT_TRUE(cpu.IsHost());
	EXPECT_FALSE(cpu.IsVulkan());
	OaDevice vk{OaDeviceType::VkDiscrete, 0};
	EXPECT_TRUE(vk.IsVulkan());
	EXPECT_TRUE(vk.IsGpu());
	EXPECT_EQ(vk.Index, 0);
}

// ─── Status ────────────────────────────────────────────────────────────────────

TEST(CoreStatus, Ok) {
	OaStatus s = OaStatus::Ok();
	EXPECT_TRUE(s.IsOk());
	EXPECT_FALSE(s.IsError());
}

TEST(CoreStatus, Error) {
	OaStatus s = OaStatus::Error(OaStatusCode::InvalidArgument, "test error");
	EXPECT_FALSE(s.IsOk());
	EXPECT_TRUE(s.IsError());
	EXPECT_EQ(s.GetCode(), OaStatusCode::InvalidArgument);
	EXPECT_EQ(s.GetMessage(), "test error");
}

TEST(CoreStatus, ResultOk) {
	OaResult<OaI32> r = 42;
	EXPECT_TRUE(r.IsOk());
	EXPECT_EQ(r.GetValue(), 42);
}

TEST(CoreStatus, ResultError) {
	OaResult<OaI32> r = OaStatus::NotFound("not found");
	EXPECT_FALSE(r.IsOk());
	EXPECT_EQ(r.GetStatus().GetCode(), OaStatusCode::NotFound);
}

TEST(CoreTimestamp, FromSeconds) {
	OaTimestamp ts = OaTimestamp::FromSeconds(1000);
	EXPECT_EQ(ts.Secs(), 1000);
	EXPECT_EQ(ts.Millis(), 1000000);
}

TEST(CoreTimestamp, Duration) {
	OaTimestamp d = OaTimestamp::FromSeconds(300);
	EXPECT_EQ(d.Secs(), 300);
	OaTimestamp h = OaTimestamp::FromSeconds(7200);
	EXPECT_EQ(h.Secs(), 7200);
}

// ─── Fixed Point ───────────────────────────────────────────────────────────────

TEST(CoreFixed, Construction) {
	OaFixed<8> zero = OaFixed<8>::Zero();
	EXPECT_EQ(zero.Raw, 0);
	OaFixed<8> one = OaFixed<8>::One();
	EXPECT_EQ(one.Raw, 100000000);
}

TEST(CoreFixed, Arithmetic) {
	OaPrice a = OaPrice::FromDouble(100.5);
	OaPrice b = OaPrice::FromDouble(50.25);
	OaPrice c = a + b;
	EXPECT_EQ(c.Raw, OaPrice::FromDouble(150.75).Raw);
}

// ─── Safe Arithmetic ───────────────────────────────────────────────────────────

TEST(CoreSafe, SafeAdd) {
	OaU64 result;
	EXPECT_TRUE(OaSafeAdd(OaU64(100), OaU64(200), result));
	EXPECT_EQ(result, 300u);
	EXPECT_FALSE(OaSafeAdd(OA_U64_MAX, OaU64(1), result));
}

TEST(CoreSafe, SafeAddClamped) {
	EXPECT_EQ(OaSafeAddClamped(OaU64(100), OaU64(200)), 300u);
	EXPECT_EQ(OaSafeAddClamped(OA_U64_MAX, OaU64(1)), OA_U64_MAX);
}
