// Correctness-gated allocator-policy benchmark.
//
// This measures OA's engine-owned host-upload buffer cache, not VMA's internal
// TLSF implementation. The cache is the policy seam OA owns and can improve.

#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/resourceAccess.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
	#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
	oa::U32 depth = 1024;
	oa::U32 iterations = 10000;
	oa::U32 samples = 21;
};

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (int index = 1; index < inArgc; ++index) {
		if (std::strcmp(inArgv[index], "--depth") == 0 and index + 1 < inArgc) {
			options.depth = static_cast<oa::U32>(std::strtoul(inArgv[++index], nullptr, 10));
		} else if (std::strcmp(inArgv[index], "--iterations") == 0 and index + 1 < inArgc) {
			options.iterations = static_cast<oa::U32>(
				std::strtoul(inArgv[++index], nullptr, 10));
		} else if (std::strcmp(inArgv[index], "--samples") == 0 and index + 1 < inArgc) {
			options.samples = static_cast<oa::U32>(std::strtoul(inArgv[++index], nullptr, 10));
		}
	}
	options.depth = std::clamp<oa::U32>(options.depth, 1U, 4096U);
	options.iterations = std::max(options.iterations, 1U);
	options.samples = std::max(options.samples, 3U);
	return options;
}

void pinToFirstAllowedCpu() {
#if defined(__linux__)
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
	for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if (not CPU_ISSET(cpu, &allowed)) continue;
		cpu_set_t selected;
		CPU_ZERO(&selected);
		CPU_SET(cpu, &selected);
		(void)sched_setaffinity(0, sizeof(selected), &selected);
		return;
	}
#endif
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	pinToFirstAllowedCpu();

	auto engineResult = oa::Engine::create({
		.preloadEmbeddedPipelines = false,
		.appName = "benchAllocator",
		.selectForThread = false,
	});
	if (not engineResult) {
		std::fprintf(stderr, "BenchAllocator: engine creation failed: %s\n",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	auto engine = std::move(*engineResult);

	std::vector<oavk::Buffer> held;
	held.reserve(options.depth);
	for (oa::U32 index = 0; index < options.depth; ++index) {
		auto result = oa::EngineResourceAccess::allocBuffer(
			*engine, static_cast<oa::U64>(index + 1U) * 4U,
			oa::MemoryPlacement::HostUpload);
		if (not result) {
			std::fprintf(stderr, "BenchAllocator: cache population failed at %u: %s\n",
				index, result.getStatus().toString().cStr());
			return 2;
		}
		held.push_back(std::move(*result));
	}
	for (auto& buffer : held) oa::EngineResourceAccess::freeBuffer(*engine, buffer);

	// A cache hit must not create a new VMA allocation.
	const oa::U64 allocationsBefore =
		oa::EngineAllocatorAccess::get(*engine).getStats().allocationCount;
	auto probe = oa::EngineResourceAccess::allocBuffer(
		*engine, 4U, oa::MemoryPlacement::HostUpload);
	if (not probe) return 3;
	oa::EngineResourceAccess::freeBuffer(*engine, *probe);
	const oa::U64 allocationsAfter =
		oa::EngineAllocatorAccess::get(*engine).getStats().allocationCount;
	if (allocationsAfter != allocationsBefore) {
		std::fprintf(stderr,
			"BenchAllocator: expected cache hit changed allocation count (%llu -> %llu)\n",
			static_cast<unsigned long long>(allocationsBefore),
			static_cast<unsigned long long>(allocationsAfter));
		return 4;
	}

	std::vector<double> samples;
	samples.reserve(options.samples);
	for (oa::U32 warmup = 0; warmup < 5U; ++warmup) {
		for (oa::U32 iteration = 0; iteration < options.iterations; ++iteration) {
			auto result = oa::EngineResourceAccess::allocBuffer(
				*engine, 4U, oa::MemoryPlacement::HostUpload);
			if (not result) return 5;
			oa::EngineResourceAccess::freeBuffer(*engine, *result);
		}
	}
	for (oa::U32 sample = 0; sample < options.samples; ++sample) {
		const auto begin = Clock::now();
		for (oa::U32 iteration = 0; iteration < options.iterations; ++iteration) {
			auto result = oa::EngineResourceAccess::allocBuffer(
				*engine, 4U, oa::MemoryPlacement::HostUpload);
			if (not result) return 6;
			oa::EngineResourceAccess::freeBuffer(*engine, *result);
		}
		const auto end = Clock::now();
		const double elapsed = std::chrono::duration<double, std::nano>(end - begin).count();
		samples.push_back(elapsed / static_cast<double>(options.iterations));
	}
	std::sort(samples.begin(), samples.end());
	std::printf(
		"host_upload_cache_hit depth=%u iterations=%u samples=%u median=%.3f ns/op min=%.3f max=%.3f\n",
		options.depth, options.iterations, options.samples,
		samples[samples.size() / 2U], samples.front(), samples.back());

	const oa::Status closeStatus = engine->close();
	if (not closeStatus.isOk()) {
		std::fprintf(stderr, "BenchAllocator: close failed: %s\n",
			closeStatus.toString().cStr());
		return 7;
	}
	return 0;
}
