// Entry point for the oa::ItTraining examples.
// Registers the VkTestEnvironment to initialize vulkan engine.
//
// CLI flags (pre-parsed before GTest sees argv):
//   --bf16 / --fp32 / --precision=bf16|fp32   engine precision (bridges to OA_TEST_BF16)
//   --device-index <n> / -d <n>               device selection   (bridges to OA_DEVICE)

#include "oaTest.h"
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

static void oaSetEnv(const char* inName, const char* inValue) {
#if defined(_WIN32)
	_putenv_s(inName, inValue);
#else
	::setenv(inName, inValue, 1);
#endif
}

static void preParsePrecision(int& inOutArgc, char** inOutArgv) {
	auto removeAt = [&](int i, int n) {
		for (int j = i; j + n < inOutArgc; ++j) inOutArgv[j] = inOutArgv[j + n];
		inOutArgc -= n;
	};
	bool bf16 = false;
	for (int i = 1; i < inOutArgc; ) {
		const char* arg = inOutArgv[i];
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
	oaSetEnv("OA_TEST_BF16", bf16 ? "1" : "0");
}

static void preParseDeviceIndex(int& inOutArgc, char** inOutArgv) {
	for (int i = 1; i < inOutArgc; ++i) {
		const char* arg = inOutArgv[i];
		const char* val = nullptr;
		if ((std::strcmp(arg, "--device-index") == 0 || std::strcmp(arg, "-d") == 0) && i + 1 < inOutArgc) {
			val = inOutArgv[i + 1];
			for (int j = i; j + 2 < inOutArgc; ++j) inOutArgv[j] = inOutArgv[j + 2];
			inOutArgc -= 2;
		} else if (std::strncmp(arg, "--device-index=", 15) == 0) {
			val = arg + 15;
			for (int j = i; j + 1 < inOutArgc; ++j) inOutArgv[j] = inOutArgv[j + 1];
			inOutArgc -= 1;
		} else {
			continue;
		}
		oaSetEnv("OA_DEVICE", val);
		--i;
	}
}

int main(int argc, char** argv) {
#if defined(_WIN32)
	setConsoleCP(CP_UTF8);
	setConsoleOutputCP(CP_UTF8);
#endif
	preParseDeviceIndex(argc, argv);
	preParsePrecision(argc, argv);
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new VkTestEnvironment());
	return RUN_ALL_TESTS();
}
