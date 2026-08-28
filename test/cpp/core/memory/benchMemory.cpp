// Statistically robust host-memory benchmark.
//
// Measures OA's host-memory primitives against their platform equivalents.
// Runtime-sized calls are intentional: this is the contract used by OA
// containers, upload staging and codec paths. Each result is a median of
// independent samples after CPU warm-up and correctness checks.

#include <oa/core/std/memory.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
	#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr oa::Usize KiB = 1024;
constexpr oa::Usize MiB = 1024 * KiB;
constexpr int WarmupSamples = 5;
constexpr int MeasuredSamples = 21;

enum class Implementation : oa::U8 {
	Runtime,
	LibcRuntime,
	Fixed,
	CompilerFixed,
	NonTemporal,
	Count,
};

enum class Primitive : oa::U8 {
	Fill,
	Zero,
	Equal,
	EqualEarlyMismatch,
	EqualLateMismatch,
	Compare,
	CompareEarlyMismatch,
	CompareLateMismatch,
	ConstantEqual,
	ConstantEqualEarlyMismatch,
	ConstantEqualLateMismatch,
	MoveRight,
	MoveLeft,
};

struct Case {
	oa::Usize size;
	oa::Usize srcOffset;
	oa::Usize dstOffset;
};

struct Stats {
	double medianNs = 0.0;
	double p10Ns = 0.0;
	double p90Ns = 0.0;
};

struct Options {
	bool quick = false;
	bool primitivesOnly = false;
	bool copyOnly = false;
	bool streamingOnly = false;
};

[[gnu::always_inline]] inline void compilerBarrier() {
	__asm__ __volatile__("" ::: "memory");
}

template <oa::Usize size, bool UseOa>
[[gnu::noinline]] void runFixedCopies(
	oa::Byte* inDst, const oa::Byte* inSrc, oa::Usize inIterations) {
	for (oa::Usize iteration = 0; iteration < inIterations; ++iteration) {
		if constexpr (UseOa) {
			oa::memcpy(inDst, inSrc, size);
		} else {
			std::memcpy(inDst, inSrc, size);
		}
		compilerBarrier();
	}
}

template <bool UseOa>
void runFixedDispatch(
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize,
	oa::Usize inIterations) {
	switch (inSize) {
		case 8: runFixedCopies<8, UseOa>(inDst, inSrc, inIterations); return;
		case 16: runFixedCopies<16, UseOa>(inDst, inSrc, inIterations); return;
		case 32: runFixedCopies<32, UseOa>(inDst, inSrc, inIterations); return;
		case 64: runFixedCopies<64, UseOa>(inDst, inSrc, inIterations); return;
		case 128: runFixedCopies<128, UseOa>(inDst, inSrc, inIterations); return;
		case 256: runFixedCopies<256, UseOa>(inDst, inSrc, inIterations); return;
		default: std::abort();
	}
}

[[gnu::noinline]] void runCopies(
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize,
	oa::Usize inIterations) {
	if (inImplementation == Implementation::Fixed) {
		runFixedDispatch<true>(inDst, inSrc, inSize, inIterations);
		return;
	}
	if (inImplementation == Implementation::CompilerFixed) {
		runFixedDispatch<false>(inDst, inSrc, inSize, inIterations);
		return;
	}
	for (oa::Usize iteration = 0; iteration < inIterations; ++iteration) {
		switch (inImplementation) {
			case Implementation::Runtime:
				oa::memcpy(inDst, inSrc, inSize);
				break;
			case Implementation::LibcRuntime:
				std::memcpy(inDst, inSrc, inSize);
				break;
			case Implementation::NonTemporal:
				oa::memcpyStream(inDst, inSrc, inSize);
				break;
			case Implementation::Fixed:
			case Implementation::CompilerFixed:
			case Implementation::Count:
				std::abort();
		}
		// Prevent identical-copy folding without adding a data dependency to the
		// copy itself. This is outside the implementation under test.
		compilerBarrier();
	}
}

const char* name(Implementation inImplementation) {
	switch (inImplementation) {
		case Implementation::Runtime: return "oa_runtime";
		case Implementation::LibcRuntime: return "libc_runtime";
		case Implementation::Fixed: return "oa_fixed";
		case Implementation::CompilerFixed: return "compiler_fixed";
		case Implementation::NonTemporal: return "oa_stream";
		case Implementation::Count: break;
	}
	return "unknown";
}

const char* name(Primitive inPrimitive) {
	switch (inPrimitive) {
		case Primitive::Fill: return "fill";
		case Primitive::Zero: return "zero";
		case Primitive::Equal: return "equal";
		case Primitive::EqualEarlyMismatch: return "equal_early_mismatch";
		case Primitive::EqualLateMismatch: return "equal_late_mismatch";
		case Primitive::Compare: return "compare";
		case Primitive::CompareEarlyMismatch: return "compare_early_mismatch";
		case Primitive::CompareLateMismatch: return "compare_late_mismatch";
		case Primitive::ConstantEqual: return "constant_equal";
		case Primitive::ConstantEqualEarlyMismatch: return "constant_equal_early_mismatch";
		case Primitive::ConstantEqualLateMismatch: return "constant_equal_late_mismatch";
		case Primitive::MoveRight: return "move_right";
		case Primitive::MoveLeft: return "move_left";
	}
	return "unknown";
}

[[gnu::noinline]] oa::Bool runPrimitive(
	Primitive inPrimitive,
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize,
	oa::Usize inIterations) {
	oa::Bool result = true;
	for (oa::Usize iteration = 0; iteration < inIterations; ++iteration) {
		switch (inPrimitive) {
			case Primitive::Fill:
				if (inImplementation == Implementation::Runtime) {
					oa::memset(inDst, 0xA5, inSize);
				} else {
					std::memset(inDst, 0xA5, inSize);
				}
				break;
			case Primitive::Zero:
				if (inImplementation == Implementation::Runtime) {
					oa::memzero(inDst, inSize);
				} else {
					std::memset(inDst, 0, inSize);
				}
				break;
			case Primitive::Equal:
			case Primitive::EqualEarlyMismatch:
			case Primitive::EqualLateMismatch:
				if (inImplementation == Implementation::Runtime) {
					result = oa::memEqual(inDst, inSrc, inSize);
				} else {
					result = std::memcmp(inDst, inSrc, inSize) == 0;
				}
				break;
			case Primitive::Compare:
			case Primitive::CompareEarlyMismatch:
			case Primitive::CompareLateMismatch:
				if (inImplementation == Implementation::Runtime) {
					result = oa::memcmp(inDst, inSrc, inSize) == 0;
				} else {
					result = std::memcmp(inDst, inSrc, inSize) == 0;
				}
				break;
			case Primitive::ConstantEqual:
			case Primitive::ConstantEqualEarlyMismatch:
			case Primitive::ConstantEqualLateMismatch:
				if (inImplementation == Implementation::Runtime) {
					result = oa::memEqualConstantTime(inDst, inSrc, inSize);
				} else {
					result = std::memcmp(inDst, inSrc, inSize) == 0;
				}
				break;
			case Primitive::MoveRight:
				if (inImplementation == Implementation::Runtime) {
					oa::memmove(inDst + 1U, inDst, inSize);
				} else {
					std::memmove(inDst + 1U, inDst, inSize);
				}
				break;
			case Primitive::MoveLeft:
				if (inImplementation == Implementation::Runtime) {
					oa::memmove(inDst, inDst + 1U, inSize);
				} else {
					std::memmove(inDst, inDst + 1U, inSize);
				}
				break;
		}
		compilerBarrier();
	}
	return result;
}

oa::Usize iterationsFor(oa::Usize inSize, bool inQuick) {
	if (inSize <= 256) return inQuick ? 200000 : 1000000;
	const oa::Usize targetBytes = inQuick ? 64 * MiB : 256 * MiB;
	const oa::Usize iterations = targetBytes / inSize;
	return std::clamp<oa::Usize>(iterations, 4, inQuick ? 100000 : 500000);
}

oa::Usize primitiveIterationsFor(oa::Usize inSize, bool inQuick) {
	if (inSize <= 256) return inQuick ? 25000 : 250000;
	const oa::Usize targetBytes = inQuick ? 2 * MiB : 32 * MiB;
	const oa::Usize iterations = targetBytes / inSize;
	return std::clamp<oa::Usize>(iterations, 2, inQuick ? 25000 : 125000);
}

double percentile(const std::vector<double>& inSorted, double inQuantile) {
	if (inSorted.empty()) return 0.0;
	const double position = inQuantile * static_cast<double>(inSorted.size() - 1);
	const auto low = static_cast<size_t>(std::floor(position));
	const auto high = static_cast<size_t>(std::ceil(position));
	const double fraction = position - static_cast<double>(low);
	return inSorted[low] * (1.0 - fraction) + inSorted[high] * fraction;
}

Stats summarize(std::vector<double> inSamples) {
	std::sort(inSamples.begin(), inSamples.end());
	return Stats{
		.medianNs = percentile(inSamples, 0.5),
		.p10Ns = percentile(inSamples, 0.1),
		.p90Ns = percentile(inSamples, 0.9),
	};
}

void pinToFirstAllowedCpu() {
#if defined(__linux__)
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
	for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if (!CPU_ISSET(cpu, &allowed)) continue;
		cpu_set_t selected;
		CPU_ZERO(&selected);
		CPU_SET(cpu, &selected);
		(void)sched_setaffinity(0, sizeof(selected), &selected);
		return;
	}
#endif
}

void warmCpu() {
	const auto until = Clock::now() + std::chrono::milliseconds(300);
	volatile oa::U64 value = 1;
	while (Clock::now() < until) {
		value = value * 6364136223846793005ULL + 1442695040888963407ULL;
	}
	(void)value;
}

bool verify(
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize) {
	std::memset(inDst, 0xA5, inSize);
	runCopies(inImplementation, inDst, inSrc, inSize, 1);
	return std::memcmp(inDst, inSrc, inSize) == 0;
}

std::array<Stats, static_cast<size_t>(Implementation::Count)> measure(
	const std::vector<Implementation>& inImplementations,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize,
	oa::Usize inIterations) {
	for (Implementation implementation : inImplementations) {
		for (int sample = 0; sample < WarmupSamples; ++sample) {
			runCopies(implementation, inDst, inSrc, inSize, inIterations);
		}
	}

	std::array<std::vector<double>, static_cast<size_t>(Implementation::Count)> samples;
	for (auto& implementationSamples : samples) {
		implementationSamples.reserve(MeasuredSamples);
	}
	for (int sample = 0; sample < MeasuredSamples; ++sample) {
		// rotate the implementation order each sample to distribute clock,
		// temperature and cache-history drift rather than baking it into one row.
		for (size_t order = 0; order < inImplementations.size(); ++order) {
			const Implementation implementation = inImplementations[
				(order + static_cast<size_t>(sample)) % inImplementations.size()];
			const auto begin = Clock::now();
			runCopies(implementation, inDst, inSrc, inSize, inIterations);
			const auto end = Clock::now();
			const double totalNs =
				std::chrono::duration<double, std::nano>(end - begin).count();
			samples[static_cast<size_t>(implementation)].push_back(
				totalNs / static_cast<double>(inIterations));
		}
	}
	std::array<Stats, static_cast<size_t>(Implementation::Count)> result;
	for (Implementation implementation : inImplementations) {
		result[static_cast<size_t>(implementation)] = summarize(
			std::move(samples[static_cast<size_t>(implementation)]));
	}
	return result;
}

std::array<Stats, static_cast<size_t>(Implementation::Count)> measurePrimitive(
	Primitive inPrimitive,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize,
	oa::Usize inIterations) {
	constexpr std::array implementations{
		Implementation::Runtime,
		Implementation::LibcRuntime,
	};
	for (Implementation implementation : implementations) {
		for (int sample = 0; sample < WarmupSamples; ++sample) {
			(void)runPrimitive(inPrimitive, implementation,
				inDst, inSrc, inSize, inIterations);
		}
	}

	std::array<std::vector<double>, static_cast<size_t>(Implementation::Count)> samples;
	for (auto& implementationSamples : samples) {
		implementationSamples.reserve(MeasuredSamples);
	}
	for (int sample = 0; sample < MeasuredSamples; ++sample) {
		for (size_t order = 0; order < implementations.size(); ++order) {
			const Implementation implementation = implementations[
				(order + static_cast<size_t>(sample)) % implementations.size()];
			const auto begin = Clock::now();
			(void)runPrimitive(inPrimitive, implementation,
				inDst, inSrc, inSize, inIterations);
			const auto end = Clock::now();
			const double totalNs =
				std::chrono::duration<double, std::nano>(end - begin).count();
			samples[static_cast<size_t>(implementation)].push_back(
				totalNs / static_cast<double>(inIterations));
		}
	}
	std::array<Stats, static_cast<size_t>(Implementation::Count)> result;
	for (Implementation implementation : implementations) {
		result[static_cast<size_t>(implementation)] = summarize(
			std::move(samples[static_cast<size_t>(implementation)]));
	}
	return result;
}

[[gnu::noinline]] void runStreamingPasses(
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inWorkingSetBytes,
	oa::Usize inChunkBytes,
	oa::Usize inPasses
) {
	for (oa::Usize pass = 0; pass < inPasses; ++pass) {
		for (oa::Usize offset = 0; offset < inWorkingSetBytes;
			offset += inChunkBytes)
		{
			switch (inImplementation) {
				case Implementation::Runtime:
					oa::memcpy(inDst + offset, inSrc + offset, inChunkBytes);
					break;
				case Implementation::LibcRuntime:
					std::memcpy(inDst + offset, inSrc + offset, inChunkBytes);
					break;
				case Implementation::NonTemporal:
					oa::memcpyStream(inDst + offset, inSrc + offset, inChunkBytes);
					break;
				case Implementation::Fixed:
				case Implementation::CompilerFixed:
				case Implementation::Count:
					std::abort();
			}
			compilerBarrier();
		}
	}
}

bool verifyStreaming(
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inWorkingSetBytes,
	oa::Usize inChunkBytes
) {
	std::memset(inDst, 0xA5, inWorkingSetBytes);
	runStreamingPasses(inImplementation, inDst, inSrc,
		inWorkingSetBytes, inChunkBytes, 1U);
	return std::memcmp(inDst, inSrc, inWorkingSetBytes) == 0;
}

std::array<Stats, static_cast<size_t>(Implementation::Count)> measureStreaming(
	const std::array<Implementation, 3>& inImplementations,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inWorkingSetBytes,
	oa::Usize inChunkBytes
) {
	for (Implementation implementation : inImplementations) {
		for (int sample = 0; sample < WarmupSamples; ++sample) {
			runStreamingPasses(implementation, inDst, inSrc,
				inWorkingSetBytes, inChunkBytes, 1U);
		}
	}

	std::array<std::vector<double>, static_cast<size_t>(Implementation::Count)> samples;
	for (auto& implementationSamples : samples) {
		implementationSamples.reserve(MeasuredSamples);
	}
	for (int sample = 0; sample < MeasuredSamples; ++sample) {
		for (size_t order = 0; order < inImplementations.size(); ++order) {
			const Implementation implementation = inImplementations[
				(order + static_cast<size_t>(sample)) % inImplementations.size()];
			const auto begin = Clock::now();
			runStreamingPasses(implementation, inDst, inSrc,
				inWorkingSetBytes, inChunkBytes, 1U);
			const auto end = Clock::now();
			samples[static_cast<size_t>(implementation)].push_back(
				std::chrono::duration<double, std::nano>(end - begin).count());
		}
	}

	std::array<Stats, static_cast<size_t>(Implementation::Count)> result;
	for (Implementation implementation : inImplementations) {
		result[static_cast<size_t>(implementation)] = summarize(
			std::move(samples[static_cast<size_t>(implementation)]));
	}
	return result;
}

int runStreamingBenchmark(bool inQuick) {
	constexpr std::array implementations{
		Implementation::Runtime,
		Implementation::LibcRuntime,
		Implementation::NonTemporal,
	};
	constexpr std::array fullChunks{
		oa::Usize{256}, oa::Usize{512}, 1U * KiB, 2U * KiB, 4U * KiB,
		8U * KiB, 16U * KiB, 64U * KiB, 256U * KiB,
		1U * MiB, 2U * MiB, 4U * MiB, 8U * MiB, 16U * MiB, 64U * MiB,
	};
	constexpr std::array quickChunks{
		oa::Usize{256}, oa::Usize{512}, 1U * KiB, 2U * KiB, 4U * KiB,
		8U * KiB, 16U * KiB, 64U * KiB, 256U * KiB,
		1U * MiB, 4U * MiB, 8U * MiB, 16U * MiB,
	};
	const oa::Usize workingSetBytes = inQuick ? 64U * MiB : 256U * MiB;
	auto* src = static_cast<oa::Byte*>(oa::alignedAlloc(workingSetBytes, 64U));
	auto* dst = static_cast<oa::Byte*>(oa::alignedAlloc(workingSetBytes, 64U));
	if (src == nullptr || dst == nullptr) {
		std::fprintf(stderr,
			"BenchMemory: streaming allocation failed for %zu-byte arenas\n",
			static_cast<size_t>(workingSetBytes));
		oa::alignedFree(src);
		oa::alignedFree(dst);
		return 1;
	}
	std::memset(src, 0x5A, workingSetBytes);
	std::memset(dst, 0, workingSetBytes);

	std::printf(
		"operation,chunk_bytes,working_set_bytes,implementation,passes,median_ns,p10_ns,p90_ns,GB_per_s\n");
	const oa::Usize* chunks = inQuick ? quickChunks.data() : fullChunks.data();
	const size_t chunkCount = inQuick ? quickChunks.size() : fullChunks.size();
	for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
		const oa::Usize chunkBytes = chunks[chunkIndex];
		for (Implementation implementation : implementations) {
			if (!verifyStreaming(implementation, dst, src,
				workingSetBytes, chunkBytes))
			{
				std::fprintf(stderr,
					"BenchMemory: correctness failure operation=stream_copy chunk=%zu working_set=%zu impl=%s\n",
					static_cast<size_t>(chunkBytes),
					static_cast<size_t>(workingSetBytes), name(implementation));
				oa::alignedFree(src);
				oa::alignedFree(dst);
				return 2;
			}
		}
		const auto measurements = measureStreaming(
			implementations, dst, src, workingSetBytes, chunkBytes);
		for (Implementation implementation : implementations) {
			const Stats stats = measurements[static_cast<size_t>(implementation)];
			const double gbPerSecond =
				static_cast<double>(workingSetBytes) / stats.medianNs;
			std::printf(
				"stream_copy,%zu,%zu,%s,1,%.3f,%.3f,%.3f,%.3f\n",
				static_cast<size_t>(chunkBytes),
				static_cast<size_t>(workingSetBytes), name(implementation),
				stats.medianNs, stats.p10Ns, stats.p90Ns, gbPerSecond);
		}
	}
	oa::alignedFree(src);
	oa::alignedFree(dst);
	return 0;
}

bool supportsFixed(oa::Usize inSize) {
	return inSize == 8 || inSize == 16 || inSize == 32 || inSize == 64
		|| inSize == 128 || inSize == 256;
}

std::vector<Case> cases(bool inQuick) {
	const oa::Usize sizesFull[] = {
		1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64,
		65, 96, 127, 128, 129, 192, 255, 256, 257, 512, 1024, 4096,
		64 * KiB, 1 * MiB, 2 * MiB, 4 * MiB, 16 * MiB, 64 * MiB,
	};
	const oa::Usize sizesQuick[] = {
		8, 16, 32, 64, 128, 256, 512, 4096,
		64 * KiB, 1 * MiB, 2 * MiB, 4 * MiB, 16 * MiB, 64 * MiB,
	};
	const oa::Usize* sizes = inQuick ? sizesQuick : sizesFull;
	const size_t count = inQuick ? std::size(sizesQuick) : std::size(sizesFull);
	std::vector<Case> cases;
	for (size_t index = 0; index < count; ++index) {
		cases.push_back(Case{sizes[index], 0, 0});
		cases.push_back(Case{sizes[index], 1, 3});
	}
	return cases;
}

[[noreturn]] void optionError(const char* inArgument) {
	std::fprintf(stderr, "BenchMemory: unknown or incompatible option: %s\n",
		inArgument);
	std::fprintf(stderr,
		"usage: benchMemory [--quick] [--copy | --primitives | --streaming]\n");
	std::exit(2);
}

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (int index = 1; index < inArgc; ++index) {
		if (std::strcmp(inArgv[index], "--quick") == 0) {
			options.quick = true;
		} else if (std::strcmp(inArgv[index], "--copy") == 0) {
			options.copyOnly = true;
		} else if (std::strcmp(inArgv[index], "--primitives") == 0) {
			options.primitivesOnly = true;
		} else if (std::strcmp(inArgv[index], "--streaming") == 0) {
			options.streamingOnly = true;
		} else if (std::strcmp(inArgv[index], "--help") == 0) {
			std::printf(
				"usage: benchMemory [--quick] [--copy | --primitives | --streaming]\n");
			std::exit(0);
		} else {
			optionError(inArgv[index]);
		}
	}
	const int selectedModes = static_cast<int>(options.copyOnly)
		+ static_cast<int>(options.primitivesOnly)
		+ static_cast<int>(options.streamingOnly);
	if (selectedModes > 1) {
		optionError("--copy/--primitives/--streaming");
	}
	return options;
}

bool isMismatchPrimitive(Primitive inPrimitive) {
	return inPrimitive == Primitive::EqualEarlyMismatch
		or inPrimitive == Primitive::EqualLateMismatch
		or inPrimitive == Primitive::CompareEarlyMismatch
		or inPrimitive == Primitive::CompareLateMismatch
		or inPrimitive == Primitive::ConstantEqualEarlyMismatch
		or inPrimitive == Primitive::ConstantEqualLateMismatch;
}

bool isEarlyMismatchPrimitive(Primitive inPrimitive) {
	return inPrimitive == Primitive::EqualEarlyMismatch
		or inPrimitive == Primitive::CompareEarlyMismatch
		or inPrimitive == Primitive::ConstantEqualEarlyMismatch;
}

bool isComparisonPrimitive(Primitive inPrimitive) {
	return inPrimitive == Primitive::Equal
		or inPrimitive == Primitive::EqualEarlyMismatch
		or inPrimitive == Primitive::EqualLateMismatch
		or inPrimitive == Primitive::Compare
		or inPrimitive == Primitive::CompareEarlyMismatch
		or inPrimitive == Primitive::CompareLateMismatch
		or inPrimitive == Primitive::ConstantEqual
		or inPrimitive == Primitive::ConstantEqualEarlyMismatch
		or inPrimitive == Primitive::ConstantEqualLateMismatch;
}

bool verifyPrimitive(
	Primitive inPrimitive,
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize) {
	if (inPrimitive == Primitive::MoveRight
		or inPrimitive == Primitive::MoveLeft)
	{
		std::memcpy(inDst, inSrc, inSize + 1U);
	}
	const oa::Bool result = runPrimitive(
		inPrimitive, inImplementation, inDst, inSrc, inSize, 1);
	switch (inPrimitive) {
		case Primitive::Fill:
			for (oa::Usize index = 0; index < inSize; ++index) {
				if (inDst[index] != 0xA5) return false;
			}
			return true;
		case Primitive::Zero:
			for (oa::Usize index = 0; index < inSize; ++index) {
				if (inDst[index] != 0) return false;
			}
			return true;
		case Primitive::Equal: return result;
		case Primitive::EqualEarlyMismatch:
		case Primitive::EqualLateMismatch:
		case Primitive::CompareEarlyMismatch:
		case Primitive::CompareLateMismatch:
		case Primitive::ConstantEqualEarlyMismatch:
		case Primitive::ConstantEqualLateMismatch: return !result;
		case Primitive::Compare:
		case Primitive::ConstantEqual: return result;
		case Primitive::MoveRight:
			for (oa::Usize index = 0; index < inSize; ++index) {
				if (inDst[index + 1U] != inSrc[index]) return false;
			}
			return true;
		case Primitive::MoveLeft:
			for (oa::Usize index = 0; index < inSize; ++index) {
				if (inDst[index] != inSrc[index + 1U]) return false;
			}
			return true;
	}
	return false;
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	pinToFirstAllowedCpu();
	warmCpu();
	if (options.streamingOnly) return runStreamingBenchmark(options.quick);

	std::printf(
		"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s\n");
	for (const Case& test : cases(options.quick)) {
		const oa::Usize allocation = test.size + 128;
		auto* srcBase = static_cast<oa::Byte*>(oa::alignedAlloc(allocation, 64));
		auto* dstBase = static_cast<oa::Byte*>(oa::alignedAlloc(allocation, 64));
		if (!srcBase || !dstBase) {
			std::fprintf(stderr, "BenchMemory: allocation failed for %zu bytes\n",
				static_cast<size_t>(allocation));
			oa::alignedFree(srcBase);
			oa::alignedFree(dstBase);
			return 1;
		}
		for (oa::Usize index = 0; index < allocation; ++index) {
			srcBase[index] = static_cast<oa::Byte>((index * 131U + 17U) & 0xFFU);
			dstBase[index] = 0;
		}
		auto* src = srcBase + test.srcOffset;
		auto* dst = dstBase + test.dstOffset;
		const oa::Usize iterations = iterationsFor(test.size, options.quick);
		const oa::Usize primitiveIterations =
			primitiveIterationsFor(test.size, options.quick);
		if (!options.primitivesOnly) {
			std::vector<Implementation> implementations{
				Implementation::Runtime,
				Implementation::LibcRuntime,
			};
			if (supportsFixed(test.size)) {
				implementations.push_back(Implementation::Fixed);
				implementations.push_back(Implementation::CompilerFixed);
			}
			implementations.push_back(Implementation::NonTemporal);
			for (Implementation implementation : implementations) {
				if (!verify(implementation, dst, src, test.size)) {
					std::fprintf(stderr,
						"BenchMemory: correctness failure operation=copy size=%zu src=%zu dst=%zu impl=%s\n",
						static_cast<size_t>(test.size),
						static_cast<size_t>(test.srcOffset),
						static_cast<size_t>(test.dstOffset), name(implementation));
					oa::alignedFree(srcBase);
					oa::alignedFree(dstBase);
					return 2;
				}
			}
			const auto measurements = measure(
				implementations, dst, src, test.size, iterations);
			for (Implementation implementation : implementations) {
				const Stats stats = measurements[static_cast<size_t>(implementation)];
				const double gbPerSecond = static_cast<double>(test.size) / stats.medianNs;
				std::printf("copy,%zu,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f\n",
					static_cast<size_t>(test.size),
					static_cast<size_t>(test.srcOffset),
					static_cast<size_t>(test.dstOffset), name(implementation),
					static_cast<size_t>(iterations), stats.medianNs,
					stats.p10Ns, stats.p90Ns, gbPerSecond);
			}
		}

		if (!options.copyOnly) {
			constexpr std::array primitives{
				Primitive::Fill,
				Primitive::Zero,
				Primitive::Equal,
				Primitive::EqualEarlyMismatch,
				Primitive::EqualLateMismatch,
				Primitive::Compare,
				Primitive::CompareEarlyMismatch,
				Primitive::CompareLateMismatch,
				Primitive::ConstantEqual,
				Primitive::ConstantEqualEarlyMismatch,
				Primitive::ConstantEqualLateMismatch,
				Primitive::MoveRight,
				Primitive::MoveLeft,
			};
			constexpr std::array implementations{
				Implementation::Runtime,
				Implementation::LibcRuntime,
			};
			for (Primitive primitive : primitives) {
				std::memcpy(dst, src, test.size + 1U);
				if (isEarlyMismatchPrimitive(primitive) && test.size > 0) {
					dst[0] ^= 0xFF;
				} else if (isMismatchPrimitive(primitive) && test.size > 0) {
					dst[test.size - 1] ^= 0xFF;
				}
				for (Implementation implementation : implementations) {
					if (!verifyPrimitive(primitive, implementation,
						dst, src, test.size)) {
						std::fprintf(stderr,
							"BenchMemory: correctness failure operation=%s size=%zu src=%zu dst=%zu impl=%s\n",
							name(primitive), static_cast<size_t>(test.size),
							static_cast<size_t>(test.srcOffset),
							static_cast<size_t>(test.dstOffset), name(implementation));
						oa::alignedFree(srcBase);
						oa::alignedFree(dstBase);
						return 2;
					}
				}
				// Fill/zero verification changed the destination; restore the
				// comparison fixtures before measuring equality.
				if (isComparisonPrimitive(primitive)) {
					std::memcpy(dst, src, test.size);
					if (isEarlyMismatchPrimitive(primitive) && test.size > 0) {
						dst[0] ^= 0xFF;
					} else if (isMismatchPrimitive(primitive) && test.size > 0) {
						dst[test.size - 1] ^= 0xFF;
					}
				} else if (primitive == Primitive::MoveRight
					|| primitive == Primitive::MoveLeft) {
					std::memcpy(dst, src, test.size + 1U);
				}
				const auto measurements = measurePrimitive(
					primitive, dst, src, test.size, primitiveIterations);
				for (Implementation implementation : implementations) {
					const Stats stats = measurements[static_cast<size_t>(implementation)];
					const double gbPerSecond =
						static_cast<double>(test.size) / stats.medianNs;
					std::printf("%s,%zu,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f\n",
						name(primitive), static_cast<size_t>(test.size),
						static_cast<size_t>(test.srcOffset),
						static_cast<size_t>(test.dstOffset), name(implementation),
						static_cast<size_t>(primitiveIterations), stats.medianNs,
						stats.p10Ns, stats.p90Ns, gbPerSecond);
				}
			}
		}
		oa::alignedFree(srcBase);
		oa::alignedFree(dstBase);
	}
	return 0;
}
