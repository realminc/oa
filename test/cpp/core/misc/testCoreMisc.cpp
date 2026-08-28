// Core: oa::Filesystem, oa::Device helpers, oa::getMemoryUsage, and oa::FnSimd.

#include "../../oaTest.h"

#include <oa/core/device.h>
#include <oa/core/log.h>
#include <oa/core/simd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
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
	const std::string sequence = std::to_string(static_cast<long long>(++gOaCoreMiscDirSeq));
	const std::string timestamp = std::to_string(static_cast<long long>(tick));
	oa::String name = oa::String("oa_core_misc_")
		+ oa::String(sequence.data(), sequence.size()) + "_"
		+ oa::String(timestamp.data(), timestamp.size());
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
	EXPECT_NEAR(oa::FnSimd::dotF32(a, b, 5), ref, 1e-5f);
	EXPECT_FLOAT_EQ(oa::FnSimd::dotF32(a, b, 0), 0.0f);
}

TEST(CoreSimd, ScaleF32) {
	oa::F32 buf[] = {1.0f, -2.0f, 3.5f, 0.0f, 8.0f};
	oa::FnSimd::scaleF32(buf, 2.0f, 5);
	EXPECT_NEAR(buf[0], 2.0f, 1e-6f);
	EXPECT_NEAR(buf[1], -4.0f, 1e-6f);
	EXPECT_NEAR(buf[2], 7.0f, 1e-6f);
	EXPECT_FLOAT_EQ(buf[3], 0.0f);
	EXPECT_NEAR(buf[4], 16.0f, 1e-6f);
}

TEST(CoreSimd, addF32) {
	oa::F32 x[] = {1.0f, 2.0f, 3.0f};
	const oa::F32 y[] = {10.0f, 20.0f, 30.0f};
	oa::FnSimd::addF32(x, y, 3);
	EXPECT_NEAR(x[0], 11.0f, 1e-6f);
	EXPECT_NEAR(x[1], 22.0f, 1e-6f);
	EXPECT_NEAR(x[2], 33.0f, 1e-6f);
}

TEST(CoreSimd, ArithmeticHandlesUnalignedPointersAndScalarTails) {
	constexpr oa::I64 count = 19;
	oa::F32 storageA[21] = {};
	oa::F32 storageB[21] = {};
	oa::F32* a = storageA + 1;
	oa::F32* b = storageB + 1;
	for (oa::I64 index = 0; index < count; ++index) {
		a[index] = static_cast<oa::F32>(index + 1);
		b[index] = static_cast<oa::F32>(index + 2);
	}

	oa::FnSimd::subF32(a, b, count);
	for (oa::I64 index = 0; index < count; ++index) EXPECT_FLOAT_EQ(a[index], -1.0F);

	for (oa::I64 index = 0; index < count; ++index) {
		a[index] = static_cast<oa::F32>(index + 1);
	}
	oa::FnSimd::mulF32(a, b, count);
	for (oa::I64 index = 0; index < count; ++index) {
		EXPECT_FLOAT_EQ(a[index], static_cast<oa::F32>((index + 1) * (index + 2)));
	}

	oa::FnSimd::divF32(a, b, count);
	for (oa::I64 index = 0; index < count; ++index) {
		EXPECT_FLOAT_EQ(a[index], static_cast<oa::F32>(index + 1));
	}
	oa::FnSimd::negF32(a, count);
	for (oa::I64 index = 0; index < count; ++index) {
		EXPECT_FLOAT_EQ(a[index], -static_cast<oa::F32>(index + 1));
	}

	const oa::F32 sentinel = a[0];
	oa::FnSimd::scaleF32(a, 2.0F, 0);
	oa::FnSimd::addF32(a, b, -1);
	EXPECT_FLOAT_EQ(a[0], sentinel);
}

TEST(CorePaths, NamedLocationsAndLexicalOwnership) {
	const oa::Path asset = oa::Paths::asset(
		"image/visionTestPattern320x180.jpg");
	EXPECT_TRUE(oa::Filesystem::isFile(asset));
	EXPECT_EQ(asset.filename().string(), "visionTestPattern320x180.jpg");
	EXPECT_EQ(asset.stem().string(), "visionTestPattern320x180");
	EXPECT_EQ(asset.extension().string(), ".jpg");
	const oa::Path sourceRoot = asset.parentPath().parentPath()
		.parentPath().parentPath();
	EXPECT_EQ(oa::Paths::var(), sourceRoot / "var");
	const oa::Path data = oa::Paths::data("fashionMnist");
	EXPECT_EQ(data.filename().string(), "fashionMnist");
	EXPECT_EQ(data.parentPath(), oa::Paths::data());
	EXPECT_EQ(oa::Paths::data(), sourceRoot / "var" / "data");

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
	const std::string hostedContents = testStdString(*contents);
	EXPECT_NE(hostedContents.find("[TST ]"), std::string::npos);
	EXPECT_NE(hostedContents.find("owned record 7"), std::string::npos);
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

TEST_F(CoreMiscFs, LogMetricsSerializesWritersFlushAndClose) {
	oa::LogMetrics metrics(workDir_.string());
	ASSERT_TRUE(metrics.isOpen());
	metrics.setFlushInterval(7);

	std::vector<std::thread> writers;
	for (oa::I32 thread = 0; thread < 4; ++thread) {
		writers.emplace_back([&, thread] {
			for (oa::I32 record = 0; record < 100; ++record) {
				metrics.logScalar("loss", thread * 100 + record,
					static_cast<oa::F64>(record));
			}
		});
	}
	for (auto& writer : writers) writer.join();
	metrics.flush();
	metrics.close();
	EXPECT_FALSE(metrics.isOpen());

	auto contents = oa::Filesystem::readText(workDir_ / "events.jsonl");
	ASSERT_TRUE(contents.isOk()) << contents.getStatus().toString();
	oa::Usize lineCount = 0;
	for (char character : *contents) {
		if (character == '\n') ++lineCount;
	}
	EXPECT_EQ(lineCount, 400U);
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
	oa::Vector<CoreMiscTestPod> in;
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
