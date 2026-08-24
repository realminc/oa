// Core: oa::Filesystem, oa::Device helpers, oa::getMemoryUsage (Host / no engine), oa::Simd (Highway).

#include "../../oaTest.h"

#include <oa/core/device.h>
#include <oa/core/log.h>
#include <oa/core/vlm.h>
#include <oa/core/simd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

struct CoreMiscTestPod {
	oa::U32 a = 0;
	oa::U32 b = 0;
};

static std::atomic<oa::U64> gOaCoreMiscDirSeq{0};
inline constexpr oa::LogComponent TestLogComponent{"TST"};

static oa::Path coreMiscMakeWorkDir() {
	oa::Path tmp = oa::Paths::temp();
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	oa::String name = oa::String("oa_core_misc_") + std::to_string(static_cast<long long>(++gOaCoreMiscDirSeq)) + "_"
		+ std::to_string(static_cast<long long>(tick));
	return tmp / oa::Path(name);
}

class CoreMiscFs : public ::testing::Test {
protected:
	oa::Path workDir_;

	void SetUp() override {
		workDir_ = coreMiscMakeWorkDir();
		ASSERT_TRUE(oa::Filesystem::createDirectories(workDir_).isOk());
	}

	void TearDown() override {
		(void)oa::Filesystem::removeDirectory(workDir_, true);
	}
};

TEST(CoreDevice, TypeNames) {
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::Host), "Host");
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::VkDiscrete), "VkDiscrete");
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::VkIntegrated), "VkIntegrated");
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::VkCpu), "VkCpu");
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::VkVirtualGpu), "VkVirtualGpu");
	EXPECT_EQ(oa::deviceTypeName(oa::DeviceType::VkOther), "VkOther");
}

TEST(CoreDevice, VulkanClassification) {
	EXPECT_FALSE(oa::isVulkanDevice(oa::DeviceType::Host));
	EXPECT_TRUE(oa::isVulkanDevice(oa::DeviceType::VkDiscrete));
	EXPECT_TRUE(oa::isVulkanDevice(oa::DeviceType::VkIntegrated));
	EXPECT_TRUE(oa::isVulkanDevice(oa::DeviceType::VkCpu));
	EXPECT_TRUE(oa::isVulkanDevice(oa::DeviceType::VkVirtualGpu));
	EXPECT_TRUE(oa::isVulkanDevice(oa::DeviceType::VkOther));
}

TEST(CoreDevice, Struct) {
	oa::Device cpu;
	EXPECT_TRUE(cpu.isHost());
	EXPECT_FALSE(cpu.isVulkan());

	oa::Device gpu(oa::DeviceType::VkDiscrete, 1);
	EXPECT_FALSE(gpu.isHost());
	EXPECT_TRUE(gpu.isVulkan());
	EXPECT_TRUE(gpu.isGpu());
	EXPECT_EQ(gpu, oa::Device(oa::DeviceType::VkDiscrete, 1));
	EXPECT_NE(gpu, oa::Device(oa::DeviceType::VkDiscrete, 2));
}

TEST(CoreDevice, MemoryLocationNames) {
	EXPECT_EQ(oa::memoryLocationName(oa::MemoryLocation::Host), "Host");
	EXPECT_EQ(oa::memoryLocationName(oa::MemoryLocation::Device), "Device");
	EXPECT_EQ(oa::memoryLocationName(oa::MemoryLocation::Shared), "Shared");
}

TEST(CoreDevice, MemoryPlacementValuesRemainStable) {
	EXPECT_EQ(static_cast<oa::U8>(oa::MemoryPlacement::Auto), 0u);
	EXPECT_EQ(static_cast<oa::U8>(oa::MemoryPlacement::DeviceLocal), 1u);
	EXPECT_EQ(static_cast<oa::U8>(oa::MemoryPlacement::HostUpload), 2u);
	EXPECT_EQ(static_cast<oa::U8>(oa::MemoryPlacement::HostReadback), 3u);
	EXPECT_EQ(static_cast<oa::U8>(oa::MemoryPlacement::Unified), 4u);
}

TEST(CoreDevice, MemoryUsageHost) {
	oa::MemoryUsage z = oa::getMemoryUsage(oa::HostDevice);
	EXPECT_EQ(z.totalBytes, 0u);
	EXPECT_EQ(z.freeBytes, 0u);
	EXPECT_EQ(z.usedBytes, 0u);
	EXPECT_DOUBLE_EQ(z.usedPercent, 0.0);
}

TEST(CoreDevice, LogicalVulkanMemoryUsageRequiresAnEngine) {
	const oa::MemoryUsage memory = oa::getMemoryUsage(
		oa::Device(oa::DeviceType::VkDiscrete, 0));
	EXPECT_EQ(memory.totalBytes, 0u);
	EXPECT_EQ(memory.freeBytes, 0u);
	EXPECT_EQ(memory.usedBytes, 0u);
	EXPECT_DOUBLE_EQ(memory.usedPercent, 0.0);
}

#if defined(__linux__) || defined(_WIN32)
TEST(CoreDevice, MemoryUsageVkCpuRam) {
	oa::MemoryUsage m = oa::getMemoryUsage(oa::Device(oa::DeviceType::VkCpu, 0));
	EXPECT_GT(m.totalBytes, 0u);
	EXPECT_EQ(m.totalBytes, m.freeBytes);
}
#endif

TEST(CoreSimd, DotF32) {
	const oa::F32 a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	const oa::F32 b[] = {0.5f, 1.5f, 2.0f, 2.5f, 3.0f};
	oa::F32 ref = 0.0f;
	for (oa::I64 i = 0; i < 5; ++i) {
		ref += a[i] * b[i];
	}
	EXPECT_NEAR(oa::Simd::dotF32(a, b, 5), ref, 1e-5f);
	EXPECT_FLOAT_EQ(oa::Simd::dotF32(a, b, 0), 0.0f);
}

TEST(CoreSimd, ScaleF32) {
	oa::F32 buf[] = {1.0f, -2.0f, 3.5f, 0.0f, 8.0f};
	oa::Simd::scaleF32(buf, 2.0f, 5);
	EXPECT_NEAR(buf[0], 2.0f, 1e-6f);
	EXPECT_NEAR(buf[1], -4.0f, 1e-6f);
	EXPECT_NEAR(buf[2], 7.0f, 1e-6f);
	EXPECT_FLOAT_EQ(buf[3], 0.0f);
	EXPECT_NEAR(buf[4], 16.0f, 1e-6f);
}

TEST(CoreSimd, addF32) {
	oa::F32 x[] = {1.0f, 2.0f, 3.0f};
	const oa::F32 y[] = {10.0f, 20.0f, 30.0f};
	oa::Simd::addF32(x, y, 3);
	EXPECT_NEAR(x[0], 11.0f, 1e-6f);
	EXPECT_NEAR(x[1], 22.0f, 1e-6f);
	EXPECT_NEAR(x[2], 33.0f, 1e-6f);
}

static oa::vlm::Vec4 transformRowVectorOracle(
	const oa::vlm::Vec4& inV,
	const oa::vlm::Mat4& inM
) {
	return {
		inV.x * inM.m[0][0] + inV.y * inM.m[1][0] +
			inV.z * inM.m[2][0] + inV.w * inM.m[3][0],
		inV.x * inM.m[0][1] + inV.y * inM.m[1][1] +
			inV.z * inM.m[2][1] + inV.w * inM.m[3][1],
		inV.x * inM.m[0][2] + inV.y * inM.m[1][2] +
			inV.z * inM.m[2][2] + inV.w * inM.m[3][2],
		inV.x * inM.m[0][3] + inV.y * inM.m[1][3] +
			inV.z * inM.m[2][3] + inV.w * inM.m[3][3],
	};
}

static_assert(std::is_standard_layout_v<oa::vlm::Vec2>);
static_assert(std::is_standard_layout_v<oa::vlm::Vec3>);
static_assert(std::is_standard_layout_v<oa::vlm::Vec4>);
static_assert(std::is_standard_layout_v<oa::vlm::Quat>);
static_assert(std::is_standard_layout_v<oa::vlm::Mat4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec2>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec3>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Vec4>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Quat>);
static_assert(std::is_trivially_copyable_v<oa::vlm::Mat4>);
static_assert(sizeof(oa::vlm::Vec2) == sizeof(oa::F32) * 2);
static_assert(sizeof(oa::vlm::Vec3) == sizeof(oa::F32) * 3);
static_assert(sizeof(oa::vlm::Vec4) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Quat) == sizeof(oa::F32) * 4);
static_assert(sizeof(oa::vlm::Mat4) == sizeof(oa::F32) * 16);

TEST(CoreVlm, VectorOperatorsMatchNamedFunctions) {
	const oa::vlm::Vec2 a2{1.0F, 2.0F};
	const oa::vlm::Vec2 b2{3.0F, 4.0F};
	EXPECT_EQ(a2 + b2, oa::vlm::add(a2, b2));
	EXPECT_EQ(a2 - b2, oa::vlm::sub(a2, b2));
	EXPECT_EQ(a2 * 2.0F, oa::vlm::scale(a2, 2.0F));
	EXPECT_EQ(a2 / 2.0F, oa::vlm::divide(a2, 2.0F));
	EXPECT_EQ(2.0F * a2, oa::vlm::scale(a2, 2.0F));

	const oa::vlm::Vec3 a3{1.0F, 2.0F, 3.0F};
	const oa::vlm::Vec3 b3{4.0F, 5.0F, 6.0F};
	EXPECT_EQ(a3 + b3, oa::vlm::add(a3, b3));
	EXPECT_EQ(-a3, oa::vlm::scale(a3, -1.0F));
	EXPECT_EQ(a3 - b3, oa::vlm::sub(a3, b3));
	EXPECT_EQ(a3 * 2.0F, oa::vlm::scale(a3, 2.0F));
	EXPECT_EQ(a3 / 2.0F, oa::vlm::divide(a3, 2.0F));

	const oa::vlm::Vec4 a4{1.0F, 2.0F, 3.0F, 4.0F};
	const oa::vlm::Vec4 b4{5.0F, 6.0F, 7.0F, 8.0F};
	EXPECT_EQ(a4 + b4, oa::vlm::add(a4, b4));
	EXPECT_EQ(a4 - b4, oa::vlm::sub(a4, b4));
	EXPECT_EQ(a4 * 2.0F, oa::vlm::scale(a4, 2.0F));
	EXPECT_EQ(a4 / 2.0F, oa::vlm::divide(a4, 2.0F));

	oa::vlm::Vec3 compound = a3;
	compound += b3;
	compound -= b3;
	compound *= 4.0F;
	compound /= 2.0F;
	EXPECT_EQ(compound, oa::vlm::scale(a3, 2.0F));
}

TEST(CoreVlm, QuaternionOperatorsMatchNamedFunctions) {
	const oa::vlm::Quat a{1.0F, 2.0F, 3.0F, 4.0F};
	const oa::vlm::Quat b{-2.0F, 1.0F, 0.5F, 3.0F};
	EXPECT_EQ(a + b, oa::vlm::add(a, b));
	EXPECT_EQ(-a, oa::vlm::scale(a, -1.0F));
	EXPECT_EQ(a - b, oa::vlm::sub(a, b));
	EXPECT_EQ(a * 2.0F, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(a / 2.0F, oa::vlm::divide(a, 2.0F));
	EXPECT_EQ(2.0F * a, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(a * b, oa::vlm::quaternionMul(a, b));

	oa::vlm::Quat compound = a;
	compound += b;
	compound -= b;
	compound *= 2.0F;
	compound /= 2.0F;
	compound *= b;
	EXPECT_EQ(compound, oa::vlm::quaternionMul(a, b));
}

TEST(CoreVlm, MatrixOperatorsMatchNamedFunctions) {
	const oa::vlm::Mat4 a = oa::vlm::translation(
		oa::vlm::Vec3{1.0F, 2.0F, 3.0F});
	const oa::vlm::Mat4 b = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{2.0F, 3.0F, 4.0F});
	EXPECT_EQ(a + b, oa::vlm::add(a, b));
	EXPECT_EQ(-a, oa::vlm::scale(a, -1.0F));
	EXPECT_EQ(a - b, oa::vlm::sub(a, b));
	EXPECT_EQ(a * 2.0F, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(a / 2.0F, oa::vlm::divide(a, 2.0F));
	EXPECT_EQ(2.0F * a, oa::vlm::scale(a, 2.0F));
	EXPECT_EQ(a * b, oa::vlm::matrixMul(a, b));

	oa::vlm::Mat4 compound = a;
	compound += b;
	compound -= b;
	compound *= 2.0F;
	compound /= 2.0F;
	compound *= b;
	EXPECT_EQ(compound, oa::vlm::matrixMul(a, b));

	const oa::vlm::Vec4 value{3.0F, 4.0F, 5.0F, 1.0F};
	EXPECT_EQ(value * a, oa::vlm::transform(value, a));
	EXPECT_EQ(value * a, transformRowVectorOracle(value, a));
}

TEST(CoreVlm, UsesRightHandedSpatialBasis) {
	const oa::vlm::Vec3 z = oa::vlm::cross(
		oa::vlm::Vec3{1.0F, 0.0F, 0.0F},
		oa::vlm::Vec3{0.0F, 1.0F, 0.0F});
	EXPECT_FLOAT_EQ(z.x, 0.0F);
	EXPECT_FLOAT_EQ(z.y, 0.0F);
	EXPECT_FLOAT_EQ(z.z, 1.0F);
}

TEST(CoreVlm, QuaternionAndRowMajorMatrixRotateIdentically) {
	const oa::vlm::Quat rotation = oa::vlm::Quat::fromAxisAngle(
		{0.0F, 0.0F, 1.0F}, oa::vlm::kPi * 0.5F);
	const oa::vlm::Vec3 value{1.0F, 0.0F, 0.0F};
	const oa::vlm::Vec3 byQuaternion = rotation.rotate(value);
	const oa::vlm::Vec4 byMatrix = transformRowVectorOracle(
		{value.x, value.y, value.z, 0.0F},
		oa::vlm::quaternionToMatrix(rotation));

	EXPECT_NEAR(byQuaternion.x, byMatrix.x, 1e-6F);
	EXPECT_NEAR(byQuaternion.y, byMatrix.y, 1e-6F);
	EXPECT_NEAR(byQuaternion.z, byMatrix.z, 1e-6F);
	EXPECT_NEAR(byMatrix.x, 0.0F, 1e-6F);
	EXPECT_NEAR(byMatrix.y, 1.0F, 1e-6F);
}

TEST(CoreVlm, QuaternionMatrixRoundTripPreservesRotation) {
	const oa::vlm::Quat input = oa::vlm::Quat::fromAxisAngle(
		{1.0F, 2.0F, -3.0F}, 1.234F);
	const oa::vlm::Quat output = oa::vlm::quaternionFromMatrix(
		oa::vlm::quaternionToMatrix(input));
	const oa::F32 alignment = input.x * output.x + input.y * output.y
		+ input.z * output.z + input.w * output.w;
	EXPECT_NEAR(std::abs(alignment), 1.0F, 1e-5F);
}

TEST(CoreVlm, PerspectiveUsesVulkanDepthRange) {
	const oa::F32 nearPlane = 0.1F;
	const oa::F32 farPlane = 100.0F;
	const oa::vlm::Mat4 projection =
		oa::vlm::perspective(60.0F, 16.0F / 9.0F, nearPlane, farPlane);
	const oa::vlm::Vec4 nearClip =
		transformRowVectorOracle({0.0F, 0.0F, -nearPlane, 1.0F}, projection);
	const oa::vlm::Vec4 farClip =
		transformRowVectorOracle({0.0F, 0.0F, -farPlane, 1.0F}, projection);

	ASSERT_NEAR(nearClip.w, nearPlane, 1e-6F);
	ASSERT_NEAR(farClip.w, farPlane, 1e-4F);
	EXPECT_NEAR(nearClip.z / nearClip.w, 0.0F, 1e-5F);
	EXPECT_NEAR(farClip.z / farClip.w, 1.0F, 1e-5F);
}

TEST(CoreVlm, OrthographicUsesVulkanDepthRange) {
	const oa::F32 nearPlane = -1.0F;
	const oa::F32 farPlane = 1.0F;
	const oa::vlm::Mat4 projection =
		oa::vlm::orthographic(1920.0F, 1080.0F, nearPlane, farPlane);
	const oa::vlm::Vec4 nearClip =
		transformRowVectorOracle({0.0F, 0.0F, -nearPlane, 1.0F}, projection);
	const oa::vlm::Vec4 farClip =
		transformRowVectorOracle({0.0F, 0.0F, -farPlane, 1.0F}, projection);

	EXPECT_NEAR(nearClip.z, 0.0F, 1e-6F);
	EXPECT_NEAR(farClip.z, 1.0F, 1e-6F);
}

TEST(CorePaths, NamedLocationsAndLexicalOwnership) {
	const oa::Path asset = oa::Paths::asset(
		"image/visionTestPattern320x180.jpg");
	EXPECT_TRUE(oa::Filesystem::isFile(asset));
	EXPECT_EQ(asset.filename().string(), "visionTestPattern320x180.jpg");
	EXPECT_EQ(asset.stem().string(), "visionTestPattern320x180");
	EXPECT_EQ(asset.extension().string(), ".jpg");

	const oa::Path nested = oa::Path("one") / "two" / ".." / "file.txt";
	EXPECT_EQ(nested.lexicallyNormal().genericString(), "one/file.txt");
	EXPECT_FALSE(oa::Paths::current().empty());
	EXPECT_TRUE(oa::Filesystem::isDirectory(oa::Paths::temp()));
	auto absolute = oa::Filesystem::absolute(oa::Path("."));
	ASSERT_TRUE(absolute.isOk());
	EXPECT_TRUE(absolute->isAbsolute());
}

TEST_F(CoreMiscFs, ReadTextMissing) {
	oa::Path p = workDir_ / "nope.txt";
	auto r = oa::Filesystem::readText(p);
	EXPECT_FALSE(r.isOk());
}

TEST_F(CoreMiscFs, TextRoundTrip) {
	oa::Path p = workDir_ / "nested" / "round.txt";
	ASSERT_TRUE(oa::Filesystem::writeText(p, "hello\nline2\n").isOk());
	auto r = oa::Filesystem::readText(p);
	ASSERT_TRUE(r.isOk());
	EXPECT_EQ(r.getValue(), oa::String("hello\nline2\n"));
	auto lines = oa::Filesystem::readLines(p);
	ASSERT_TRUE(lines.isOk());
	ASSERT_EQ(lines.getValue().size(), 2u);
	EXPECT_EQ(lines.getValue()[0], oa::String("hello"));
	EXPECT_EQ(lines.getValue()[1], oa::String("line2"));
}

TEST(Log, ComponentVocabularyIsStableAndExtensible) {
	struct ComponentCase {
		oa::LogComponent component;
		const char* expected;
	};
	constexpr ComponentCase cases[] = {
		{oa::LogComponent::Core,      "CORE"},
		{oa::LogComponent::Runtime,   "RT  "},
		{oa::LogComponent::Engine,    "ENGN"},
		{oa::LogComponent::Compute,   "COMP"},
		{oa::LogComponent::Ml,        "ML  "},
		{oa::LogComponent::Data,      "DATA"},
		{oa::LogComponent::Vision,    "VISN"},
		{oa::LogComponent::Video,     "VID "},
		{oa::LogComponent::Audio,     "AUD "},
		{oa::LogComponent::Render,    "RNDR"},
		{oa::LogComponent::Ui,        "UI  "},
		{oa::LogComponent::Plot,      "PLOT"},
		{oa::LogComponent::Animation, "ANIM"},
		{oa::LogComponent::Network,   "NET "},
		{oa::LogComponent::Crypto,    "CRYP"},
		{oa::LogComponent::Python,    "PY  "},
		{oa::LogComponent::App,       "APP "},
	};
	for (const auto& item : cases) EXPECT_STREQ(item.component.cStr(), item.expected);
	EXPECT_STREQ(TestLogComponent.cStr(), "TST ");
}

TEST_F(CoreMiscFs, LogSessionOwnsFileAndReportsClosedWrites) {
	auto created = oa::Log::create(oa::LogOptions{
		.directory = workDir_.string(),
		.prefix = "session",
		.minimumLevel = oa::LogLevel::Trace,
		.consoleOutput = false,
		.fileOutput = true,
	});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto log = oa::move(created).getValue();
	ASSERT_TRUE(log->write(
		oa::LogLevel::Info, TestLogComponent, "owned record %d", 7).isOk());
	ASSERT_TRUE(log->flush().isOk());
	const oa::Path path(log->getLogPath());
	ASSERT_TRUE(oa::Filesystem::isFile(path));
	auto contents = oa::Filesystem::readText(path);
	ASSERT_TRUE(contents.isOk());
	EXPECT_NE(contents->stdStr().find("[TST ]"), std::string::npos);
	EXPECT_NE(contents->stdStr().find("owned record 7"), std::string::npos);
	ASSERT_TRUE(log->close().isOk());
	EXPECT_EQ(log->write(
		oa::LogLevel::Info, oa::LogComponent::Core, "after close").getCode(),
		oa::StatusCode::FailedPrecondition);
}

TEST_F(CoreMiscFs, LogSessionSerializesWritersAndLevelChanges) {
	auto created = oa::Log::create(oa::LogOptions{
		.directory = workDir_.string(),
		.prefix = "concurrent",
		.minimumLevel = oa::LogLevel::Trace,
		.consoleOutput = false,
		.fileOutput = true,
	});
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto log = oa::move(created).getValue();
	std::atomic<oa::Bool> failed{false};
	std::vector<std::thread> writers;
	for (oa::I32 thread = 0; thread < 4; ++thread) {
		writers.emplace_back([&, thread] {
			for (oa::I32 record = 0; record < 100; ++record) {
				log->setLevel((record & 1) == 0
					? oa::LogLevel::Trace : oa::LogLevel::Info);
				if (not log->write(oa::LogLevel::Info, oa::LogComponent::Core,
					"writer=%d record=%d", thread, record).isOk()) {
					failed.store(true, std::memory_order_relaxed);
				}
			}
		});
	}
	for (auto& writer : writers) writer.join();
	EXPECT_FALSE(failed.load(std::memory_order_relaxed));
	EXPECT_TRUE(log->flush().isOk());
	EXPECT_TRUE(log->close().isOk());
}

TEST_F(CoreMiscFs, BinaryRoundTrip) {
	oa::Path p = workDir_ / "b.bin";
	const oa::U8 src[] = {0x00, 0xFF, 0x42, 0x13};
	ASSERT_TRUE(oa::Filesystem::writeBinary(p, oa::Span<const oa::U8>(src, 4)).isOk());
	auto r = oa::Filesystem::readBinary(p);
	ASSERT_TRUE(r.isOk());
	ASSERT_EQ(r.getValue().size(), 4u);
	EXPECT_EQ(std::memcmp(r.getValue().data(), src, 4), 0);
}

TEST_F(CoreMiscFs, PodRoundTrip) {
	oa::Path p = workDir_ / "pod.bin";
	oa::Vec<CoreMiscTestPod> in;
	in.pushBack(CoreMiscTestPod{7, 42});
	in.pushBack(CoreMiscTestPod{99, 1});
	ASSERT_TRUE(oa::Filesystem::writePod(p, oa::Span<const CoreMiscTestPod>(in.data(), in.size())).isOk());
	auto out = oa::Filesystem::readPod<CoreMiscTestPod>(p);
	ASSERT_TRUE(out.isOk());
	ASSERT_EQ(out.getValue().size(), 2u);
	EXPECT_EQ(out.getValue()[0].a, 7u);
	EXPECT_EQ(out.getValue()[0].b, 42u);
	EXPECT_EQ(out.getValue()[1].a, 99u);
	EXPECT_EQ(out.getValue()[1].b, 1u);
}

TEST_F(CoreMiscFs, GlobTxt) {
	oa::Path a = workDir_ / "x.txt";
	oa::Path b = workDir_ / "y.txt";
	ASSERT_TRUE(oa::Filesystem::writeText(a, "a").isOk());
	ASSERT_TRUE(oa::Filesystem::writeText(b, "b").isOk());
	auto g = oa::Filesystem::glob(workDir_, "*.txt");
	ASSERT_TRUE(g.isOk());
	ASSERT_EQ(g.getValue().size(), 2u);
	EXPECT_EQ(g.getValue()[0].filename().string(), "x.txt");
	EXPECT_EQ(g.getValue()[1].filename().string(), "y.txt");
}
