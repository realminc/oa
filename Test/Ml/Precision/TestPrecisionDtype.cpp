#include <gtest/gtest.h>

#include <Oa/Ml/Precision.h>

TEST(PrecisionDtype, LoadStoreStrideBf16Engine)
{
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Bf16, OaPrecision::BF16), 2u);
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Fp32, OaPrecision::BF16), 4u);
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Mixed, OaPrecision::BF16), 4u);
}

TEST(PrecisionDtype, LoadStoreStrideFp32Engine)
{
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Bf16, OaPrecision::FP32), 4u);
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Fp32, OaPrecision::FP32), 4u);
	EXPECT_EQ(OaLoadStoreStrideBytes(OaPrecisionTag::Mixed, OaPrecision::FP32), 4u);
}

TEST(PrecisionDtype, TableMatchesStrideUnderBf16Engine)
{
	for (OaU32 i = 0; i < kPrecisionTableSize; ++i) {
		const OaPrecisionEntry& row = kPrecisionTable[i];
		const OaU32 stride = OaLoadStoreStrideBytes(row.Tag, OaPrecision::BF16);
		const OaU32 dtype = OaPrecisionDtype(row.Tag, OaPrecision::BF16);
		const OaU32 expect = (dtype == 0u) ? 4u : 2u;
		EXPECT_EQ(stride, expect) << row.Name;
	}
}

TEST(PrecisionDtype, DefaultShaderIsBf16Tag)
{
	EXPECT_EQ(OaLookupPrecision("Matmul"), OaPrecisionTag::Bf16);
	EXPECT_EQ(OaLoadStoreStrideBytes(OaLookupPrecision("Matmul"), OaPrecision::BF16), 2u);
}

TEST(PrecisionDtype, F32PipelineSuffix)
{
	EXPECT_TRUE(OaPipelineNameIsF32Variant("Matmul_f32"));
	EXPECT_FALSE(OaPipelineNameIsF32Variant("Matmul"));
	EXPECT_FALSE(OaPipelineNameIsF32Variant("f32"));
}
