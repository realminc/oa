// Main entry point for OaMlTraining examples
// Registers the OaVkTestEnvironment to initialize Vulkan engine.
//
// CLI flags (pre-parsed before GTest sees argv):
//   --bf16 / --fp32 / --precision=bf16|fp32   engine precision (bridges to OA_TEST_BF16)
//   --device-index <n> / -d <n>               device selection   (bridges to OA_DEVICE)

#include "../../Test/OaTest.h"
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

static void PreParsePrecision(int& InOutArgc, char** InOutArgv) {
	auto removeAt = [&](int i, int n) {
		for (int j = i; j + n < InOutArgc; ++j) InOutArgv[j] = InOutArgv[j + n];
		InOutArgc -= n;
	};
	bool bf16 = false;
	for (int i = 1; i < InOutArgc; ) {
		const char* arg = InOutArgv[i];
		if (std::strcmp(arg, "--bf16") == 0) { bf16 = true; removeAt(i, 1); continue; }
		if (std::strcmp(arg, "--fp32") == 0) { bf16 = false; removeAt(i, 1); continue; }
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

static void PreParseDeviceIndex(int& InOutArgc, char** InOutArgv) {
	for (int i = 1; i < InOutArgc; ++i) {
		const char* arg = InOutArgv[i];
		const char* val = nullptr;
		if ((std::strcmp(arg, "--device-index") == 0 || std::strcmp(arg, "-d") == 0) && i + 1 < InOutArgc) {
			val = InOutArgv[i + 1];
			for (int j = i; j + 2 < InOutArgc; ++j) InOutArgv[j] = InOutArgv[j + 2];
			InOutArgc -= 2;
		} else if (std::strncmp(arg, "--device-index=", 15) == 0) {
			val = arg + 15;
			for (int j = i; j + 1 < InOutArgc; ++j) InOutArgv[j] = InOutArgv[j + 1];
			InOutArgc -= 1;
		} else {
			continue;
		}
		OaSetEnv("OA_DEVICE", val);
		--i;
	}
}

int main(int argc, char** argv) {
#if defined(_WIN32)
	SetConsoleCP(CP_UTF8);
	SetConsoleOutputCP(CP_UTF8);
#endif
	PreParseDeviceIndex(argc, argv);
	PreParsePrecision(argc, argv);
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new OaVkTestEnvironment());
	return RUN_ALL_TESTS();
}
