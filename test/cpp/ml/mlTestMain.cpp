#include <gtest/gtest.h>
#include "../oaTest.h"
#include "../oaTestMain.h"
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

static void setEnv(const char* inName, const char* inValue) {
#if defined(_WIN32)
	_putenv_s(inName, inValue);
#else
	::setenv(inName, inValue, 1);
#endif
}

// Pre-parse --device-index from argv before GTest touches it.
// Sets OA_DEVICE env var so testMergeDeviceEnv picks it up.
static int preParseDeviceIndex(int& inOutArgc, char** inOutArgv) {
	for (int i = 1; i < inOutArgc; ++i) {
		const char* arg = inOutArgv[i];
		if ((std::strcmp(arg, "--device-index") == 0 || std::strcmp(arg, "-d") == 0)
		    && i + 1 < inOutArgc) {
			long idx = std::strtol(inOutArgv[i + 1], nullptr, 10);
			if (idx >= 0) {
				setEnv("OA_DEVICE", inOutArgv[i + 1]);
			}
			// remove both tokens
			for (int j = i; j + 2 < inOutArgc; ++j) {
				inOutArgv[j] = inOutArgv[j + 2];
			}
			inOutArgc -= 2;
			return static_cast<int>(idx);
		}
		if (std::strncmp(arg, "--device-index=", 15) == 0) {
			const char* val = arg + 15;
			long idx = std::strtol(val, nullptr, 10);
			if (idx >= 0) {
				setEnv("OA_DEVICE", val);
			}
			// remove token
			for (int j = i; j + 1 < inOutArgc; ++j) {
				inOutArgv[j] = inOutArgv[j + 1];
			}
			inOutArgc -= 1;
			return static_cast<int>(idx);
		}
	}
	return -1;
}

// Pre-parse --bf16 / --fp32 / --precision=bf16|fp32 from argv before GTest.
// An explicit CLI choice overrides OA_TEST_BF16; otherwise preserve the
// environment so CI and direct invocations follow the documented contract.
static void preParsePrecision(int& inOutArgc, char** inOutArgv) {
	auto removeAt = [&](int i, int n) {
		for (int j = i; j + n < inOutArgc; ++j) inOutArgv[j] = inOutArgv[j + n];
		inOutArgc -= n;
	};
	bool bf16 = false;
	bool hasCliPrecision = false;
	for (int i = 1; i < inOutArgc; ) {
		const char* arg = inOutArgv[i];
		if (std::strcmp(arg, "--bf16") == 0) {
			bf16 = true;
			hasCliPrecision = true;
			removeAt(i, 1);
			continue;
		}
		if (std::strcmp(arg, "--fp32") == 0) {
			bf16 = false;
			hasCliPrecision = true;
			removeAt(i, 1);
			continue;
		}
		if (std::strncmp(arg, "--precision=", 12) == 0) {
			const char* val = arg + 12;
			bf16 = (std::strcmp(val, "bf16") == 0 || std::strcmp(val, "BF16") == 0);
			hasCliPrecision = true;
			removeAt(i, 1);
			continue;
		}
		++i;
	}
	if (hasCliPrecision) {
		setEnv("OA_TEST_BF16", bf16 ? "1" : "0");
	}
}

int main(int argc, char** argv) {
#if defined(_WIN32)
	setConsoleCP(CP_UTF8);
	setConsoleOutputCP(CP_UTF8);
#endif
	(void)preParseDeviceIndex(argc, argv);
	preParsePrecision(argc, argv);
	testing::InitGoogleTest(&argc, argv);
	testing::AddGlobalTestEnvironment(new VkTestEnvironment);
	return runAllTests();
}
