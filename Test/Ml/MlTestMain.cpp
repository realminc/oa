#include <gtest/gtest.h>
#include "../OaTest.h"
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

static void OaSetEnv(const char* InName, const char* InValue) {
#if defined(_WIN32)
	_putenv_s(InName, InValue);
#else
	::setenv(InName, InValue, 1);
#endif
}

// Pre-parse --device-index from argv before GTest touches it.
// Sets OA_DEVICE env var so OaTestMergeDeviceEnv picks it up.
static int PreParseDeviceIndex(int& InOutArgc, char** InOutArgv) {
	for (int i = 1; i < InOutArgc; ++i) {
		const char* arg = InOutArgv[i];
		if ((std::strcmp(arg, "--device-index") == 0 || std::strcmp(arg, "-d") == 0)
		    && i + 1 < InOutArgc) {
			long idx = std::strtol(InOutArgv[i + 1], nullptr, 10);
			if (idx >= 0) {
				OaSetEnv("OA_DEVICE", InOutArgv[i + 1]);
			}
			// Remove both tokens
			for (int j = i; j + 2 < InOutArgc; ++j) {
				InOutArgv[j] = InOutArgv[j + 2];
			}
			InOutArgc -= 2;
			return static_cast<int>(idx);
		}
		if (std::strncmp(arg, "--device-index=", 15) == 0) {
			const char* val = arg + 15;
			long idx = std::strtol(val, nullptr, 10);
			if (idx >= 0) {
				OaSetEnv("OA_DEVICE", val);
			}
			// Remove token
			for (int j = i; j + 1 < InOutArgc; ++j) {
				InOutArgv[j] = InOutArgv[j + 1];
			}
			InOutArgc -= 1;
			return static_cast<int>(idx);
		}
	}
	return -1;
}

// Pre-parse --bf16 / --fp32 / --precision=bf16|fp32 from argv before GTest.
// Bridges to OA_TEST_BF16 (read by OaVkTestEnvironment::SetUp) so the test
// binary selects engine precision from the CLI, not the environment.
static void PreParsePrecision(int& InOutArgc, char** InOutArgv) {
	auto removeAt = [&](int i, int n) {
		for (int j = i; j + n < InOutArgc; ++j) InOutArgv[j] = InOutArgv[j + n];
		InOutArgc -= n;
	};
	bool bf16 = false;
	for (int i = 1; i < InOutArgc; ) {
		const char* arg = InOutArgv[i];
		if (std::strcmp(arg, "--bf16") == 0) {
			bf16 = true;
			removeAt(i, 1);
			continue;
		}
		if (std::strcmp(arg, "--fp32") == 0) {
			bf16 = false;
			removeAt(i, 1);
			continue;
		}
		if (std::strncmp(arg, "--precision=", 12) == 0) {
			const char* val = arg + 12;
			bf16 = (std::strcmp(val, "bf16") == 0 || std::strcmp(val, "BF16") == 0);
			removeAt(i, 1);
			continue;
		}
		++i;
	}
	OaSetEnv("OA_TEST_BF16", bf16 ? "1" : "0");
}

int main(int argc, char** argv) {
#if defined(_WIN32)
	SetConsoleCP(CP_UTF8);
	SetConsoleOutputCP(CP_UTF8);
#endif
	(void)PreParseDeviceIndex(argc, argv);
	PreParsePrecision(argc, argv);
	testing::InitGoogleTest(&argc, argv);
	testing::AddGlobalTestEnvironment(new OaVkTestEnvironment);
	return RUN_ALL_TESTS();
}
