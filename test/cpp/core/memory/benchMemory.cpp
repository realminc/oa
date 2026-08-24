// Statistically robust host-memory benchmark.
//
// Measures OA's host-memory primitives against their platform equivalents.
// Runtime-sized calls are intentional: this is the contract used by OA
// containers, upload staging and codec paths. Each result is a median of
// independent samples after CPU warm-up and correctness checks.

#include <oa/core/memory.h>

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
				oa::memcpyNt(inDst, inSrc, inSize);
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
		case Implementation::NonTemporal: return "nt";
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

bool hasArg(int inArgc, char** inArgv, const char* inValue) {
	for (int index = 1; index < inArgc; ++index) {
		if (std::strcmp(inArgv[index], inValue) == 0) return true;
	}
	return false;
}

bool verifyPrimitive(
	Primitive inPrimitive,
	Implementation inImplementation,
	oa::Byte* inDst,
	const oa::Byte* inSrc,
	oa::Usize inSize) {
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
		case Primitive::EqualLateMismatch: return !result;
	}
	return false;
}

} // namespace

int main(int argc, char** argv) {
	const bool quick = hasArg(argc, argv, "--quick");
	const bool primitivesOnly = hasArg(argc, argv, "--primitives");
	const bool copyOnly = hasArg(argc, argv, "--copy");
	pinToFirstAllowedCpu();
	warmCpu();

	std::printf(
		"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s\n");
	for (const Case& test : cases(quick)) {
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
		const oa::Usize iterations = iterationsFor(test.size, quick);
		if (!primitivesOnly) {
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

		if (!copyOnly) {
			constexpr std::array primitives{
				Primitive::Fill,
				Primitive::Zero,
				Primitive::Equal,
				Primitive::EqualEarlyMismatch,
				Primitive::EqualLateMismatch,
			};
			constexpr std::array implementations{
				Implementation::Runtime,
				Implementation::LibcRuntime,
			};
			for (Primitive primitive : primitives) {
				std::memcpy(dst, src, test.size);
				if (primitive == Primitive::EqualEarlyMismatch && test.size > 0) {
					dst[0] ^= 0xFF;
				} else if (primitive == Primitive::EqualLateMismatch && test.size > 0) {
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
				if (primitive == Primitive::Equal
					|| primitive == Primitive::EqualEarlyMismatch
					|| primitive == Primitive::EqualLateMismatch) {
					std::memcpy(dst, src, test.size);
					if (primitive == Primitive::EqualEarlyMismatch && test.size > 0) {
						dst[0] ^= 0xFF;
					} else if (primitive == Primitive::EqualLateMismatch && test.size > 0) {
						dst[test.size - 1] ^= 0xFF;
					}
				}
				const auto measurements = measurePrimitive(
					primitive, dst, src, test.size, iterations);
				for (Implementation implementation : implementations) {
					const Stats stats = measurements[static_cast<size_t>(implementation)];
					const double gbPerSecond =
						static_cast<double>(test.size) / stats.medianNs;
					std::printf("%s,%zu,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f\n",
						name(primitive), static_cast<size_t>(test.size),
						static_cast<size_t>(test.srcOffset),
						static_cast<size_t>(test.dstOffset), name(implementation),
						static_cast<size_t>(iterations), stats.medianNs,
						stats.p10Ns, stats.p90Ns, gbPerSecond);
				}
			}
		}
		oa::alignedFree(srcBase);
		oa::alignedFree(dstBase);
	}
	return 0;
}
