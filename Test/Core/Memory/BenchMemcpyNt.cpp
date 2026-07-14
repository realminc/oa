// Non-temporal memcpy sweep: OA_MEMCPY_NT_PREFETCH × buffer size → CSV.
// Default mode (no args): Linux prints header, re-execs per (prefetch, size) so Memory.cpp static init sees env.
//   ./bin/release/Test/bench_memcpy_nt
// Single run (e.g. from parent sweep):
//   OA_MEMCPY_NT_PREFETCH=1024 ./bin/release/Test/bench_memcpy_nt --once --mb=16
// Optional baseline row (same process, after NT timing):
//   ... --once --mb=16 --compare

#include <Oa/Core/Memory.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
	#include <sys/wait.h>
	#include <unistd.h>
#endif

using Clock = std::chrono::steady_clock;

static OaUsize ParseSizeMiB(int argc, char** argv) {
	OaUsize mib = 16;
	for (int i = 2; i < argc; ++i) {
		const char* arg = argv[i];
		if (std::strncmp(arg, "--mb=", 5) == 0) {
			char* end = nullptr;
			unsigned long v = std::strtoul(arg + 5, &end, 10);
			if (end != arg + 5 && *end == '\0' && v > 0) mib = static_cast<OaUsize>(v);
		}
	}
	return mib * 1024 * 1024;
}

static bool ArgHasCompare(int argc, char** argv) {
	for (int i = 2; i < argc; ++i) {
		if (std::strcmp(argv[i], "--compare") == 0) return true;
	}
	return false;
}

static unsigned PrefetchBytesForCsv() {
	const char* env = std::getenv("OA_MEMCPY_NT_PREFETCH");
	if (!env || !*env) return 512;
	char* end = nullptr;
	unsigned long v = std::strtoul(env, &end, 10);
	if (end == env) return 512;
	return static_cast<unsigned>(v);
}

static void RunTimed(
	const char* ImplName,
	unsigned PrefetchCsv,
	void* InDst,
	const void* InSrc,
	OaUsize InSize,
	int InWarmup,
	int InIters,
	void (*InFn)(void*, const void*, OaUsize)
) {
	volatile OaU8 sink = 0;
	for (int warmIdx = 0; warmIdx < InWarmup; ++warmIdx) {
		InFn(InDst, InSrc, InSize);
		sink ^= static_cast<OaU8*>(InDst)[InSize - 1];
	}
	(void)sink;

	const auto t0 = Clock::now();
	for (int iterIdx = 0; iterIdx < InIters; ++iterIdx) InFn(InDst, InSrc, InSize);
	const auto t1 = Clock::now();
	const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(InIters);
	const double gbPerSec = static_cast<double>(InSize) / ns;
	std::printf(
		"%s,%u,%zu,%.1f,%.4f\n",
		ImplName,
		PrefetchCsv,
		static_cast<std::size_t>(InSize),
		ns,
		gbPerSec
	);
}

static void CallMemcpyNT(void* InDst, const void* InSrc, OaUsize InSize) {
	OaMemcpyNT(InDst, InSrc, InSize);
}

static void CallStdMemcpy(void* InDst, const void* InSrc, OaUsize InSize) {
	std::memcpy(InDst, InSrc, static_cast<std::size_t>(InSize));
}

static int RunOnce(int argc, char** argv) {
	const OaUsize bytes = ParseSizeMiB(argc, argv);
	const bool compare = ArgHasCompare(argc, argv);
	const int warmup = 5;
	const int iters = 50;

	void* src = OaAlignedAlloc(bytes, 64);
	void* dst = OaAlignedAlloc(bytes, 64);
	if (!src || !dst) {
		std::fprintf(stderr, "bench_memcpy_nt: alloc failed (%zu bytes)\n", static_cast<std::size_t>(bytes));
		OaAlignedFree(src);
		OaAlignedFree(dst);
		return 1;
	}
	std::memset(src, 0x5A, static_cast<std::size_t>(bytes));

	RunTimed("nt", PrefetchBytesForCsv(), dst, src, bytes, warmup, iters, CallMemcpyNT);
	if (compare) RunTimed("memcpy", 0, dst, src, bytes, warmup, iters, CallStdMemcpy);

	OaAlignedFree(src);
	OaAlignedFree(dst);
	return 0;
}

#if defined(__linux__)
static bool ResolveSelfExe(char* OutBuf, size_t OutLen) {
	const ssize_t n = readlink("/proc/self/exe", OutBuf, OutLen - 1);
	if (n <= 0) return false;
	OutBuf[n] = '\0';
	return true;
}

static int RunSweep(char** argv) {
	static const unsigned kPrefs[] = {0, 256, 512, 1024, 2048};
	static const unsigned kMiB[] = {4, 16, 64};

	char exePath[4096];
	if (!ResolveSelfExe(exePath, sizeof(exePath))) {
		std::fprintf(stderr, "bench_memcpy_nt: readlink /proc/self/exe failed; pass path via argv[0]\n");
		return 1;
	}

	std::printf("impl,prefetch_bytes,size_bytes,avg_ns_per_iter,gb_per_s\n");
	std::fflush(stdout);

	for (unsigned mib : kMiB) {
		char mbArg[48];
		std::snprintf(mbArg, sizeof(mbArg), "--mb=%u", mib);
		for (unsigned pref : kPrefs) {
			char prefBuf[32];
			std::snprintf(prefBuf, sizeof(prefBuf), "%u", pref);

			const pid_t pid = fork();
			if (pid < 0) {
				perror("fork");
				return 1;
			}
			if (pid == 0) {
				setenv("OA_MEMCPY_NT_PREFETCH", prefBuf, 1);
				execl(exePath, "bench_memcpy_nt", "--once", mbArg, "--compare", nullptr);
				perror("execl");
				_exit(127);
			}
			int st = 0;
			if (waitpid(pid, &st, 0) < 0) {
				perror("waitpid");
				return 1;
			}
			if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
				std::fprintf(stderr, "bench_memcpy_nt: child exit %d\n", st);
				return 1;
			}
		}
	}
	(void)argv;
	return 0;
}
#endif

int main(int argc, char** argv) {
	if (argc >= 2 && std::strcmp(argv[1], "--once") == 0) return RunOnce(argc, argv);

#if defined(__linux__)
	return RunSweep(argv);
#else
	std::fprintf(
		stderr,
		"bench_memcpy_nt: sweep needs Linux (fork+exec). Use: OA_MEMCPY_NT_PREFETCH=N %s --once --mb=16 [--compare]\n",
		argv[0]
	);
	return 1;
#endif
}
