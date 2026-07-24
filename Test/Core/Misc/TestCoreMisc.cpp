// Core: OaFilesystem, OaDevice helpers, OaGetMemoryUsage (Host / no engine), OaSimd (Highway).

#include "../../OaTest.h"

#include <Oa/Core/Device.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Core/Simd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>

struct OaCoreMiscTestPod {
	OaU32 A = 0;
	OaU32 B = 0;
};

static std::atomic<OaU64> gOaCoreMiscDirSeq{0};

static OaPath OaCoreMiscMakeWorkDir() {
	OaPath tmp = OaPaths::Temp();
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	OaString name = OaString("oa_core_misc_") + std::to_string(static_cast<long long>(++gOaCoreMiscDirSeq)) + "_"
		+ std::to_string(static_cast<long long>(tick));
	return tmp / OaPath(name);
}

class CoreMiscFs : public ::testing::Test {
protected:
	OaPath WorkDir_;

	void SetUp() override {
		WorkDir_ = OaCoreMiscMakeWorkDir();
		ASSERT_TRUE(OaFilesystem::CreateDirectories(WorkDir_).IsOk());
	}

	void TearDown() override {
		(void)OaFilesystem::RemoveDirectory(WorkDir_, true);
	}
};

TEST(CoreDevice, TypeNames) {
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::Host), "Host");
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::VkDiscrete), "VkDiscrete");
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::VkIntegrated), "VkIntegrated");
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::VkCpu), "VkCpu");
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::VkVirtualGpu), "VkVirtualGpu");
	EXPECT_EQ(OaDeviceTypeName(OaDeviceType::VkOther), "VkOther");
}

TEST(CoreDevice, VulkanClassification) {
	EXPECT_FALSE(OaIsVulkanDevice(OaDeviceType::Host));
	EXPECT_TRUE(OaIsVulkanDevice(OaDeviceType::VkDiscrete));
	EXPECT_TRUE(OaIsVulkanDevice(OaDeviceType::VkIntegrated));
	EXPECT_TRUE(OaIsVulkanDevice(OaDeviceType::VkCpu));
	EXPECT_TRUE(OaIsVulkanDevice(OaDeviceType::VkVirtualGpu));
	EXPECT_TRUE(OaIsVulkanDevice(OaDeviceType::VkOther));
}

TEST(CoreDevice, Struct) {
	OaDevice cpu;
	EXPECT_TRUE(cpu.IsHost());
	EXPECT_FALSE(cpu.IsVulkan());

	OaDevice gpu(OaDeviceType::VkDiscrete, 1);
	EXPECT_FALSE(gpu.IsHost());
	EXPECT_TRUE(gpu.IsVulkan());
	EXPECT_TRUE(gpu.IsGpu());
	EXPECT_EQ(gpu, OaDevice(OaDeviceType::VkDiscrete, 1));
	EXPECT_NE(gpu, OaDevice(OaDeviceType::VkDiscrete, 2));
}

TEST(CoreDevice, MemoryLocationNames) {
	EXPECT_EQ(OaMemoryLocationName(OaMemoryLocation::Host), "Host");
	EXPECT_EQ(OaMemoryLocationName(OaMemoryLocation::Device), "Device");
	EXPECT_EQ(OaMemoryLocationName(OaMemoryLocation::Shared), "Shared");
}

TEST(CoreDevice, MemoryUsageHost) {
	OaMemoryUsage z = OaGetMemoryUsage(OA_DEVICE_HOST);
	EXPECT_EQ(z.TotalBytes, 0u);
	EXPECT_EQ(z.FreeBytes, 0u);
	EXPECT_EQ(z.UsedBytes, 0u);
	EXPECT_DOUBLE_EQ(z.UsedPercent, 0.0);
}

#if defined(__linux__) || defined(_WIN32)
TEST(CoreDevice, MemoryUsageVkCpuRam) {
	OaMemoryUsage m = OaGetMemoryUsage(OaDevice(OaDeviceType::VkCpu, 0));
	EXPECT_GT(m.TotalBytes, 0u);
	EXPECT_EQ(m.TotalBytes, m.FreeBytes);
}
#endif

TEST(CoreSimd, DotF32) {
	const OaF32 a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	const OaF32 b[] = {0.5f, 1.5f, 2.0f, 2.5f, 3.0f};
	OaF32 ref = 0.0f;
	for (OaI64 i = 0; i < 5; ++i) {
		ref += a[i] * b[i];
	}
	EXPECT_NEAR(OaSimd::DotF32(a, b, 5), ref, 1e-5f);
	EXPECT_FLOAT_EQ(OaSimd::DotF32(a, b, 0), 0.0f);
}

TEST(CoreSimd, ScaleF32) {
	OaF32 buf[] = {1.0f, -2.0f, 3.5f, 0.0f, 8.0f};
	OaSimd::ScaleF32(buf, 2.0f, 5);
	EXPECT_NEAR(buf[0], 2.0f, 1e-6f);
	EXPECT_NEAR(buf[1], -4.0f, 1e-6f);
	EXPECT_NEAR(buf[2], 7.0f, 1e-6f);
	EXPECT_FLOAT_EQ(buf[3], 0.0f);
	EXPECT_NEAR(buf[4], 16.0f, 1e-6f);
}

TEST(CoreSimd, AddF32) {
	OaF32 x[] = {1.0f, 2.0f, 3.0f};
	const OaF32 y[] = {10.0f, 20.0f, 30.0f};
	OaSimd::AddF32(x, y, 3);
	EXPECT_NEAR(x[0], 11.0f, 1e-6f);
	EXPECT_NEAR(x[1], 22.0f, 1e-6f);
	EXPECT_NEAR(x[2], 33.0f, 1e-6f);
}

static VlmVec4 TransformRowVector(const VlmVec4& InV, const VlmMat4& InM) {
	return {
		InV.X * InM.M[0][0] + InV.Y * InM.M[1][0] +
			InV.Z * InM.M[2][0] + InV.W * InM.M[3][0],
		InV.X * InM.M[0][1] + InV.Y * InM.M[1][1] +
			InV.Z * InM.M[2][1] + InV.W * InM.M[3][1],
		InV.X * InM.M[0][2] + InV.Y * InM.M[1][2] +
			InV.Z * InM.M[2][2] + InV.W * InM.M[3][2],
		InV.X * InM.M[0][3] + InV.Y * InM.M[1][3] +
			InV.Z * InM.M[2][3] + InV.W * InM.M[3][3],
	};
}

TEST(CoreVlm, PerspectiveUsesVulkanDepthRange) {
	const OaF32 nearPlane = 0.1F;
	const OaF32 farPlane = 100.0F;
	const VlmMat4 projection =
		Vlm::Perspective(60.0F, 16.0F / 9.0F, nearPlane, farPlane);
	const VlmVec4 nearClip =
		TransformRowVector({0.0F, 0.0F, -nearPlane, 1.0F}, projection);
	const VlmVec4 farClip =
		TransformRowVector({0.0F, 0.0F, -farPlane, 1.0F}, projection);

	ASSERT_NEAR(nearClip.W, nearPlane, 1e-6F);
	ASSERT_NEAR(farClip.W, farPlane, 1e-4F);
	EXPECT_NEAR(nearClip.Z / nearClip.W, 0.0F, 1e-5F);
	EXPECT_NEAR(farClip.Z / farClip.W, 1.0F, 1e-5F);
}

TEST(CoreVlm, OrthographicUsesVulkanDepthRange) {
	const OaF32 nearPlane = -1.0F;
	const OaF32 farPlane = 1.0F;
	const VlmMat4 projection =
		Vlm::Orthographic(1920.0F, 1080.0F, nearPlane, farPlane);
	const VlmVec4 nearClip =
		TransformRowVector({0.0F, 0.0F, -nearPlane, 1.0F}, projection);
	const VlmVec4 farClip =
		TransformRowVector({0.0F, 0.0F, -farPlane, 1.0F}, projection);

	EXPECT_NEAR(nearClip.Z, 0.0F, 1e-6F);
	EXPECT_NEAR(farClip.Z, 1.0F, 1e-6F);
}

TEST(CorePaths, NamedLocationsAndLexicalOwnership) {
	const OaPath asset = OaPaths::Asset(
		"Image/VisionTestPattern320x180.jpg");
	EXPECT_TRUE(OaFilesystem::IsFile(asset));
	EXPECT_EQ(asset.Filename().String(), "VisionTestPattern320x180.jpg");
	EXPECT_EQ(asset.Stem().String(), "VisionTestPattern320x180");
	EXPECT_EQ(asset.Extension().String(), ".jpg");

	const OaPath nested = OaPath("one") / "two" / ".." / "file.txt";
	EXPECT_EQ(nested.LexicallyNormal().GenericString(), "one/file.txt");
	EXPECT_FALSE(OaPaths::Current().Empty());
	EXPECT_TRUE(OaFilesystem::IsDirectory(OaPaths::Temp()));
	auto absolute = OaFilesystem::Absolute(OaPath("."));
	ASSERT_TRUE(absolute.IsOk());
	EXPECT_TRUE(absolute->IsAbsolute());
}

TEST_F(CoreMiscFs, ReadTextMissing) {
	OaPath p = WorkDir_ / "nope.txt";
	auto r = OaFilesystem::ReadText(p);
	EXPECT_FALSE(r.IsOk());
}

TEST_F(CoreMiscFs, TextRoundTrip) {
	OaPath p = WorkDir_ / "nested" / "round.txt";
	ASSERT_TRUE(OaFilesystem::WriteText(p, "hello\nline2\n").IsOk());
	auto r = OaFilesystem::ReadText(p);
	ASSERT_TRUE(r.IsOk());
	EXPECT_EQ(r.GetValue(), OaString("hello\nline2\n"));
	auto lines = OaFilesystem::ReadLines(p);
	ASSERT_TRUE(lines.IsOk());
	ASSERT_EQ(lines.GetValue().Size(), 2u);
	EXPECT_EQ(lines.GetValue()[0], OaString("hello"));
	EXPECT_EQ(lines.GetValue()[1], OaString("line2"));
}

TEST_F(CoreMiscFs, BinaryRoundTrip) {
	OaPath p = WorkDir_ / "b.bin";
	const OaU8 src[] = {0x00, 0xFF, 0x42, 0x13};
	ASSERT_TRUE(OaFilesystem::WriteBinary(p, OaSpan<const OaU8>(src, 4)).IsOk());
	auto r = OaFilesystem::ReadBinary(p);
	ASSERT_TRUE(r.IsOk());
	ASSERT_EQ(r.GetValue().Size(), 4u);
	EXPECT_EQ(std::memcmp(r.GetValue().Data(), src, 4), 0);
}

TEST_F(CoreMiscFs, PodRoundTrip) {
	OaPath p = WorkDir_ / "pod.bin";
	OaVec<OaCoreMiscTestPod> in;
	in.PushBack(OaCoreMiscTestPod{7, 42});
	in.PushBack(OaCoreMiscTestPod{99, 1});
	ASSERT_TRUE(OaFilesystem::WritePod(p, OaSpan<const OaCoreMiscTestPod>(in.Data(), in.Size())).IsOk());
	auto out = OaFilesystem::ReadPod<OaCoreMiscTestPod>(p);
	ASSERT_TRUE(out.IsOk());
	ASSERT_EQ(out.GetValue().Size(), 2u);
	EXPECT_EQ(out.GetValue()[0].A, 7u);
	EXPECT_EQ(out.GetValue()[0].B, 42u);
	EXPECT_EQ(out.GetValue()[1].A, 99u);
	EXPECT_EQ(out.GetValue()[1].B, 1u);
}

TEST_F(CoreMiscFs, GlobTxt) {
	OaPath a = WorkDir_ / "x.txt";
	OaPath b = WorkDir_ / "y.txt";
	ASSERT_TRUE(OaFilesystem::WriteText(a, "a").IsOk());
	ASSERT_TRUE(OaFilesystem::WriteText(b, "b").IsOk());
	auto g = OaFilesystem::Glob(WorkDir_, "*.txt");
	ASSERT_TRUE(g.IsOk());
	ASSERT_EQ(g.GetValue().Size(), 2u);
	EXPECT_EQ(g.GetValue()[0].Filename().String(), "x.txt");
	EXPECT_EQ(g.GetValue()[1].Filename().String(), "y.txt");
}
