// Direct, correctness-gated VMA allocator-algorithm benchmark.
//
// The stock mode makes the same source compile against pristine AMD VMA:
//   clang++ -O3 -DNDEBUG -std=c++20 -fno-strict-aliasing \
//     -fno-strict-overflow -DOA_BENCH_VMA_STOCK \
//     -I/path/to/upstream/include benchVma.cpp -pthread -o benchVmaStock

#if defined(OA_BENCH_VMA_STOCK)
	#define VMA_STATIC_VULKAN_FUNCTIONS 0
	#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
	#define VMA_IMPLEMENTATION
	#include <vk_mem_alloc.h>

namespace benchVma {
using VirtualBlock = VmaVirtualBlock;
using VirtualAllocation = VmaVirtualAllocation;
using VirtualBlockCreateInfo = VmaVirtualBlockCreateInfo;
using VirtualAllocationCreateInfo = VmaVirtualAllocationCreateInfo;
using Statistics = VmaStatistics;
using DetailedStatistics = VmaDetailedStatistics;

inline VkResult createVirtualBlock(
	const VirtualBlockCreateInfo* inInfo,
	VirtualBlock* outBlock
) noexcept {
	return vmaCreateVirtualBlock(inInfo, outBlock);
}
inline void destroyVirtualBlock(VirtualBlock inBlock) noexcept {
	vmaDestroyVirtualBlock(inBlock);
}
inline bool isVirtualBlockEmpty(VirtualBlock inBlock) noexcept {
	return vmaIsVirtualBlockEmpty(inBlock) != VK_FALSE;
}
inline VkResult virtualAllocate(
	VirtualBlock inBlock,
	const VirtualAllocationCreateInfo* inInfo,
	VirtualAllocation* outAllocation,
	VkDeviceSize* outOffset
) noexcept {
	return vmaVirtualAllocate(inBlock, inInfo, outAllocation, outOffset);
}
inline void virtualFree(
	VirtualBlock inBlock,
	VirtualAllocation inAllocation
) noexcept {
	vmaVirtualFree(inBlock, inAllocation);
}
inline void getVirtualBlockStatistics(
	VirtualBlock inBlock,
	Statistics* outStatistics
) noexcept {
	vmaGetVirtualBlockStatistics(inBlock, outStatistics);
}
inline void calculateVirtualBlockStatistics(
	VirtualBlock inBlock,
	DetailedStatistics* outStatistics
) noexcept {
	vmaCalculateVirtualBlockStatistics(inBlock, outStatistics);
}
} // namespace benchVma
#else
	// Compile the derived implementation directly into this benchmark so the
	// same source can exercise private container/copy algorithms. The ordinary
	// product target continues to compile vma.cpp exactly once in liboa.
	#include <vma/vma.cpp>
	#include <vma/vma.hpp>
namespace benchVma = vma;
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#if defined(__linux__)
	#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

volatile std::uint64_t gSink = 0;

struct Options {
	std::size_t items = 1U << 14U;
	int warmups = 5;
	int samples = 21;
};

struct Stats {
	double median = 0.0;
	double p10 = 0.0;
	double p90 = 0.0;
};

[[noreturn]] void optionError(
	const char* inMessage,
	const char* inArgument = nullptr
) {
	std::fprintf(stderr, "benchVma: %s%s%s\n", inMessage,
		inArgument != nullptr ? ": " : "",
		inArgument != nullptr ? inArgument : "");
	std::fprintf(stderr,
		"usage: benchVma [--items N] [--warmups N] [--samples N]\n"
		"       items >= 64, warmups >= 1, samples >= 7\n");
	std::exit(2);
}

std::uint64_t parseUnsigned(const char* inText, const char* inOption) {
	if (inText == nullptr or inText[0] == '\0' or inText[0] == '-') {
		optionError("invalid non-negative integer", inOption);
	}
	errno = 0;
	char* end = nullptr;
	const unsigned long long value = std::strtoull(inText, &end, 10);
	if (errno == ERANGE or end == inText or *end != '\0') {
		optionError("invalid non-negative integer", inOption);
	}
	return static_cast<std::uint64_t>(value);
}

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (int index = 1; index < inArgc; ++index) {
		const char* option = inArgv[index];
		if (std::strcmp(option, "--help") == 0) {
			std::printf(
				"usage: benchVma [--items N] [--warmups N] [--samples N]\n");
			std::exit(0);
		}
		const bool recognized = std::strcmp(option, "--items") == 0
			or std::strcmp(option, "--samples") == 0
			or std::strcmp(option, "--warmups") == 0;
		if (not recognized) optionError("unknown option", option);
		if (index + 1 >= inArgc) optionError("missing option value", option);
		const std::uint64_t value = parseUnsigned(inArgv[++index], option);
		if (std::strcmp(option, "--items") == 0) {
			if (value < 64U or value > (1U << 20U)) {
				optionError("--items is outside the supported range", option);
			}
			options.items = static_cast<std::size_t>(value);
		} else if (std::strcmp(option, "--samples") == 0) {
			if (value < 7U or value > static_cast<std::uint64_t>(INT_MAX)) {
				optionError("--samples is outside the supported range", option);
			}
			options.samples = static_cast<int>(value);
		} else {
			if (value < 1U or value > static_cast<std::uint64_t>(INT_MAX)) {
				optionError("--warmups is outside the supported range", option);
			}
			options.warmups = static_cast<int>(value);
		}
	}
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

Stats summarize(std::vector<double> inSamples) {
	std::sort(inSamples.begin(), inSamples.end());
	const std::size_t count = inSamples.size();
	return {
		inSamples[count / 2U],
		inSamples[(count - 1U) / 10U],
		inSamples[((count - 1U) * 9U) / 10U],
	};
}

template<typename Fn>
bool runCase(
	const char* inName,
	std::size_t inOperations,
	const Options& inOptions,
	Fn&& inFn
) {
	const std::uint64_t oracle = inFn();
	if (oracle == 0U) {
		std::fprintf(stderr, "%s correctness gate failed\n", inName);
		return false;
	}
	for (int warmup = 0; warmup < inOptions.warmups; ++warmup) {
		gSink = gSink ^ inFn();
	}
	std::vector<double> samples;
	samples.reserve(static_cast<std::size_t>(inOptions.samples));
	for (int sample = 0; sample < inOptions.samples; ++sample) {
		const auto begin = Clock::now();
		const std::uint64_t result = inFn();
		const auto end = Clock::now();
		if (result != oracle) {
			std::fprintf(stderr, "%s oracle changed\n", inName);
			return false;
		}
		gSink = gSink ^ result;
		const double elapsed =
			std::chrono::duration<double, std::nano>(end - begin).count();
		samples.push_back(elapsed / static_cast<double>(inOperations));
	}
	const Stats stats = summarize(std::move(samples));
	std::printf("%-28s median=%9.3f ns/op p10=%9.3f p90=%9.3f\n",
		inName, stats.median, stats.p10, stats.p90);
	return true;
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	pinToFirstAllowedCpu();
	std::printf("VMA direct benchmark: implementation=%s items=%zu warmups=%d samples=%d\n",
#if defined(OA_BENCH_VMA_STOCK)
		"amd-stock",
#else
		"oa-vma",
#endif
		options.items, options.warmups, options.samples);

	std::vector<VkDeviceSize> sizes(options.items);
	std::vector<VkDeviceSize> alignments(options.items);
	std::vector<std::size_t> freeOrder(options.items);
	VkDeviceSize blockSize = 0;
	std::uint64_t state = 0xD1B54A32D192ED03ULL;
	for (std::size_t index = 0; index < options.items; ++index) {
		state ^= state >> 12U;
		state ^= state << 25U;
		state ^= state >> 27U;
		const std::uint64_t random = state * 0x2545F4914F6CDD1DULL;
		sizes[index] = 32U + random % 4065U;
		alignments[index] = VkDeviceSize{1} << (4U + ((random >> 16U) & 3U));
		blockSize += sizes[index] + 256U;
		freeOrder[index] = index;
	}
	std::mt19937_64 shuffle(0xA110C8EDULL);
	std::shuffle(freeOrder.begin(), freeOrder.end(), shuffle);

	benchVma::VirtualBlock block = VK_NULL_HANDLE;
	const benchVma::VirtualBlockCreateInfo blockInfo{
		.size = blockSize,
	};
	if (benchVma::createVirtualBlock(&blockInfo, &block) != VK_SUCCESS) {
		std::fprintf(stderr, "benchVma: virtual block creation failed\n");
		return 1;
	}

	std::vector<benchVma::VirtualAllocation> allocations(
		options.items, VK_NULL_HANDLE);
	std::vector<VkDeviceSize> offsets(options.items, 0U);
	auto allocateFree = [&]() -> std::uint64_t {
		std::uint64_t checksum = 0xCBF29CE484222325ULL;
		VkDeviceSize requestedBytes = 0;
		for (std::size_t index = 0; index < options.items; ++index) {
			const benchVma::VirtualAllocationCreateInfo info{
				.size = sizes[index],
				.alignment = alignments[index],
			};
			if (benchVma::virtualAllocate(
					block, &info, &allocations[index], &offsets[index]) != VK_SUCCESS) {
				return 0U;
			}
			if ((offsets[index] & (alignments[index] - 1U)) != 0U) return 0U;
			requestedBytes += sizes[index];
			checksum ^= offsets[index] + sizes[index] * (index + 1U);
			checksum *= 0x100000001B3ULL;
		}
		benchVma::Statistics statistics{};
		benchVma::getVirtualBlockStatistics(block, &statistics);
		if (statistics.allocationCount != options.items
			or statistics.allocationBytes != requestedBytes) return 0U;
		for (const std::size_t index : freeOrder) {
			benchVma::virtualFree(block, allocations[index]);
			allocations[index] = VK_NULL_HANDLE;
		}
		if (not benchVma::isVirtualBlockEmpty(block)) return 0U;
		return checksum != 0U ? checksum : 1U;
	};

	bool passed = runCase(
		"virtual allocate/free",
		options.items * 2U,
		options,
		allocateFree);

	// Keep a populated tree while measuring the detailed full-walk statistics
	// path. Setup and teardown are outside the timed section.
	VkDeviceSize requestedBytes = 0;
	for (std::size_t index = 0; index < options.items; ++index) {
		const benchVma::VirtualAllocationCreateInfo info{
			.size = sizes[index],
			.alignment = alignments[index],
		};
		if (benchVma::virtualAllocate(
				block, &info, &allocations[index], &offsets[index]) != VK_SUCCESS) {
			std::fprintf(stderr, "benchVma: statistics setup allocation failed\n");
			return 1;
		}
		requestedBytes += sizes[index];
	}
	const std::size_t statisticsCalls = std::max<std::size_t>(
		16U, (1U << 20U) / options.items);
	auto detailedStatistics = [&]() -> std::uint64_t {
		std::uint64_t checksum = 0;
		for (std::size_t iteration = 0; iteration < statisticsCalls; ++iteration) {
			benchVma::DetailedStatistics statistics{};
			benchVma::calculateVirtualBlockStatistics(block, &statistics);
			if (statistics.statistics.allocationCount != options.items
				or statistics.statistics.allocationBytes != requestedBytes) return 0U;
			checksum += statistics.unusedRangeCount
				+ statistics.statistics.allocationBytes + iteration;
		}
		return checksum != 0U ? checksum : 1U;
	};
	passed = runCase(
		"virtual detailed stats",
		statisticsCalls * options.items,
		options,
		detailedStatistics) and passed;

	for (const std::size_t index : freeOrder) {
		benchVma::virtualFree(block, allocations[index]);
	}
	if (not benchVma::isVirtualBlockEmpty(block)) passed = false;
	benchVma::destroyVirtualBlock(block);

	// The public virtual-allocation workload above does not materially exercise
	// VMA's internal POD-container copies. These cases keep the actual upstream
	// container and allocation-callback machinery in the measurement while
	// validating every copied value.
	using InternalAllocator = VmaStlAllocator<std::uint64_t>;
	const InternalAllocator internalAllocator(nullptr);
	const std::size_t growthPasses = std::max<std::size_t>(
		1U, (1U << 20U) / options.items);
	auto vectorGrowth = [&]() -> std::uint64_t {
		std::uint64_t checksum = 0xCBF29CE484222325ULL;
		for (std::size_t pass = 0; pass < growthPasses; ++pass) {
			VmaVector<std::uint64_t, InternalAllocator> values(
				internalAllocator);
			for (std::size_t index = 0; index < options.items; ++index) {
				values.push_back(
					static_cast<std::uint64_t>(index)
						* 0x9E3779B185EBCA87ULL
					+ 0xD1B54A32D192ED03ULL);
			}
			if (values.size() != options.items) return 0U;
			for (std::size_t index = 0; index < options.items; ++index) {
				const std::uint64_t expected =
					static_cast<std::uint64_t>(index)
						* 0x9E3779B185EBCA87ULL
					+ 0xD1B54A32D192ED03ULL;
				if (values[index] != expected) return 0U;
				checksum = (checksum ^ values[index]) * 0x100000001B3ULL;
			}
		}
		return checksum != 0U ? checksum : 1U;
	};
	passed = runCase(
		"internal vector growth",
		options.items * growthPasses,
		options,
		vectorGrowth) and passed;

	VmaVector<std::uint64_t, InternalAllocator> cloneSource(
		4U, internalAllocator);
	for (std::size_t index = 0; index < cloneSource.size(); ++index) {
		cloneSource[index] = 0xA0761D6478BD642FULL
			^ (static_cast<std::uint64_t>(index) * 0xE7037ED1A0B428DBULL);
	}
	const std::size_t cloneIterations = std::max<std::size_t>(
		options.items, 1U << 17U);
	auto vectorClone32 = [&]() -> std::uint64_t {
		std::uint64_t checksum = 0x243F6A8885A308D3ULL;
		for (std::size_t iteration = 0;
			iteration < cloneIterations;
			++iteration)
		{
			const VmaVector<std::uint64_t, InternalAllocator> copy(cloneSource);
			if (copy.size() != cloneSource.size()) return 0U;
			for (std::size_t index = 0; index < copy.size(); ++index) {
				if (copy[index] != cloneSource[index]) return 0U;
				checksum = (checksum ^ copy[index]) * 0x100000001B3ULL;
			}
		}
		return checksum != 0U ? checksum : 1U;
	};
	passed = runCase(
		"internal vector clone 32B",
		cloneIterations,
		options,
		vectorClone32) and passed;

	const std::size_t spillIterations = std::max<std::size_t>(
		1U << 14U, options.items / 4U);
	auto smallVectorSpill = [&]() -> std::uint64_t {
		std::uint64_t checksum = 0x13198A2E03707344ULL;
		for (std::size_t iteration = 0;
			iteration < spillIterations;
			++iteration)
		{
			VmaSmallVector<std::uint64_t, InternalAllocator, 16U> values(
				internalAllocator);
			for (std::size_t index = 0; index < 33U; ++index) {
				values.push_back(
					(static_cast<std::uint64_t>(iteration) << 32U) | index);
			}
			if (values.size() != 33U) return 0U;
			for (std::size_t index = 0; index < values.size(); ++index) {
				const std::uint64_t expected =
					(static_cast<std::uint64_t>(iteration) << 32U) | index;
				if (values[index] != expected) return 0U;
				checksum = (checksum ^ values[index]) * 0x100000001B3ULL;
			}
			values.resize(16U, true);
			if (values.size() != 16U) return 0U;
			for (std::size_t index = 0; index < values.size(); ++index) {
				const std::uint64_t expected =
					(static_cast<std::uint64_t>(iteration) << 32U) | index;
				if (values[index] != expected) return 0U;
			}
		}
		return checksum != 0U ? checksum : 1U;
	};
	passed = runCase(
		"internal small-vector spill",
		spillIterations,
		options,
		smallVectorSpill) and passed;
	return passed ? 0 : 1;
}
