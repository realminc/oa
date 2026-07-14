#include "../../OaTest.h"
#include <Oa/Runtime/Init.h>
#include <Oa/Runtime/Engine.h>
#include <cstdlib>
#include <cstdio>

TEST(EngineInit, GlobalEngineFromEnvironment) {
	ASSERT_TRUE(OaVkTestEngineOk());
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	EXPECT_NE(rt->Device.Device, nullptr);
	EXPECT_NE(rt->Device.PhysicalDevice, nullptr);
	EXPECT_NE(rt->Device.Instance, nullptr);
	EXPECT_FALSE(rt->Device.Info.Hardware.DeviceName.empty());
}

TEST(EngineInit, OptionalExtensionsReport) {
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	const OaVkDevice& d = rt->Device;
	fprintf(stderr,
		"  [EngineInit] optional caps: CoopMat=%d PipLib=%d ExtMemFd=%d 16bit=%d SAM=%d\n",
		d.Info.Software.HasCooperativeMatrix ? 1 : 0,
		d.Info.Software.HasPipelineLibrary ? 1 : 0,
		d.Info.Software.HasExternalMemoryFd ? 1 : 0,
		d.Info.Software.Has16BitStorage ? 1 : 0,
		d.Info.Hardware.HasSAM ? 1 : 0);
	fprintf(stderr,
		"  [EngineInit] extensions policy: %s (enable), %s (query-only fallback), %s, %s+%s\n",
		OaVkExtKhrCooperativeMatrix,
		OaVkExtNvCooperativeMatrix,
		OaVkExtKhrPipelineLibrary,
		OaVkExtKhrExternalMemory,
		OaVkExtKhrExternalMemoryFd);
	SUCCEED();
}

TEST(EngineInit, CooperativeMatrixStrictEnv) {
	const char* req = std::getenv("OA_REQUIRE_COOPMAT");
	if (!req || req[0] == '\0' || std::strcmp(req, "0") == 0) {
		GTEST_SKIP() << "set OA_REQUIRE_COOPMAT=1 to require VK_KHR cooperative matrix path";
	}
	auto* rt = OaComputeEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);
	EXPECT_TRUE(rt->Device.Info.Software.HasCooperativeMatrix)
		<< "Driver must expose usable 16x16x16 cooperative matrices + "
		<< OaVkExtKhrCooperativeMatrix;
}

TEST_VK(OaVkEngineTestFixture, UsesSharedEngine) {
	EXPECT_NE(Rt().Device.Queues.ComputeQueue, nullptr);
}
