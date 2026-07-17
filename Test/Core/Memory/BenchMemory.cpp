// Statistically robust host-memory benchmark.
//
// Measures OA's host-memory primitives against their platform equivalents.
// Runtime-sized calls are intentional: this is the contract used by OA
// containers, upload staging and codec paths. Each result is a median of
// independent samples after CPU warm-up and correctness checks.

#include <Oa/Core/Memory.h>

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

constexpr OaUsize KiB = 1024;
constexpr OaUsize MiB = 1024 * KiB;
constexpr int WarmupSamples = 5;
constexpr int MeasuredSamples = 21;

enum class Implementation : OaU8 {
	OaRuntime,
	LibcRuntime,
	OaFixed,
	CompilerFixed,
	NonTemporal,
	Count,
};

enum class Primitive : OaU8 {
	Fill,
	Zero,
	Equal,
	EqualEarlyMismatch,
	EqualLateMismatch,
};

struct Case {
	OaUsize Size;
	OaUsize SrcOffset;
	OaUsize DstOffset;
};

struct Stats {
	double MedianNs = 0.0;
	double P10Ns = 0.0;
	double P90Ns = 0.0;
};

[[gnu::always_inline]] inline void CompilerBarrier() {
	__asm__ __volatile__("" ::: "memory");
}

template <OaUsize Size, bool UseOa>
[[gnu::noinline]] void RunFixedCopies(
	OaByte* InDst, const OaByte* InSrc, OaUsize InIterations) {
	for (OaUsize iteration = 0; iteration < InIterations; ++iteration) {
		if constexpr (UseOa) {
			OaMemcpy(InDst, InSrc, Size);
		} else {
			std::memcpy(InDst, InSrc, Size);
		}
		CompilerBarrier();
	}
}

template <bool UseOa>
void RunFixedDispatch(
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize,
	OaUsize InIterations) {
	switch (InSize) {
		case 8: RunFixedCopies<8, UseOa>(InDst, InSrc, InIterations); return;
		case 16: RunFixedCopies<16, UseOa>(InDst, InSrc, InIterations); return;
		case 32: RunFixedCopies<32, UseOa>(InDst, InSrc, InIterations); return;
		case 64: RunFixedCopies<64, UseOa>(InDst, InSrc, InIterations); return;
		case 128: RunFixedCopies<128, UseOa>(InDst, InSrc, InIterations); return;
		case 256: RunFixedCopies<256, UseOa>(InDst, InSrc, InIterations); return;
		default: std::abort();
	}
}

[[gnu::noinline]] void RunCopies(
	Implementation InImplementation,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize,
	OaUsize InIterations) {
	if (InImplementation == Implementation::OaFixed) {
		RunFixedDispatch<true>(InDst, InSrc, InSize, InIterations);
		return;
	}
	if (InImplementation == Implementation::CompilerFixed) {
		RunFixedDispatch<false>(InDst, InSrc, InSize, InIterations);
		return;
	}
	for (OaUsize iteration = 0; iteration < InIterations; ++iteration) {
		switch (InImplementation) {
			case Implementation::OaRuntime:
				OaMemcpy(InDst, InSrc, InSize);
				break;
			case Implementation::LibcRuntime:
				std::memcpy(InDst, InSrc, InSize);
				break;
			case Implementation::NonTemporal:
				OaMemcpyNT(InDst, InSrc, InSize);
				break;
			case Implementation::OaFixed:
			case Implementation::CompilerFixed:
			case Implementation::Count:
				std::abort();
		}
		// Prevent identical-copy folding without adding a data dependency to the
		// copy itself. This is outside the implementation under test.
		CompilerBarrier();
	}
}

const char* Name(Implementation InImplementation) {
	switch (InImplementation) {
		case Implementation::OaRuntime: return "oa_runtime";
		case Implementation::LibcRuntime: return "libc_runtime";
		case Implementation::OaFixed: return "oa_fixed";
		case Implementation::CompilerFixed: return "compiler_fixed";
		case Implementation::NonTemporal: return "nt";
		case Implementation::Count: break;
	}
	return "unknown";
}

const char* Name(Primitive InPrimitive) {
	switch (InPrimitive) {
		case Primitive::Fill: return "fill";
		case Primitive::Zero: return "zero";
		case Primitive::Equal: return "equal";
		case Primitive::EqualEarlyMismatch: return "equal_early_mismatch";
		case Primitive::EqualLateMismatch: return "equal_late_mismatch";
	}
	return "unknown";
}

[[gnu::noinline]] OaBool RunPrimitive(
	Primitive InPrimitive,
	Implementation InImplementation,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize,
	OaUsize InIterations) {
	OaBool result = true;
	for (OaUsize iteration = 0; iteration < InIterations; ++iteration) {
		switch (InPrimitive) {
			case Primitive::Fill:
				if (InImplementation == Implementation::OaRuntime) {
					OaMemset(InDst, 0xA5, InSize);
				} else {
					std::memset(InDst, 0xA5, InSize);
				}
				break;
			case Primitive::Zero:
				if (InImplementation == Implementation::OaRuntime) {
					OaMemzero(InDst, InSize);
				} else {
					std::memset(InDst, 0, InSize);
				}
				break;
			case Primitive::Equal:
			case Primitive::EqualEarlyMismatch:
			case Primitive::EqualLateMismatch:
				if (InImplementation == Implementation::OaRuntime) {
					result = OaMemEqual(InDst, InSrc, InSize);
				} else {
					result = std::memcmp(InDst, InSrc, InSize) == 0;
				}
				break;
		}
		CompilerBarrier();
	}
	return result;
}

OaUsize IterationsFor(OaUsize InSize, bool InQuick) {
	if (InSize <= 256) return InQuick ? 200000 : 1000000;
	const OaUsize targetBytes = InQuick ? 64 * MiB : 256 * MiB;
	const OaUsize iterations = targetBytes / InSize;
	return std::clamp<OaUsize>(iterations, 4, InQuick ? 100000 : 500000);
}

double Percentile(const std::vector<double>& InSorted, double InQuantile) {
	if (InSorted.empty()) return 0.0;
	const double position = InQuantile * static_cast<double>(InSorted.size() - 1);
	const auto low = static_cast<size_t>(std::floor(position));
	const auto high = static_cast<size_t>(std::ceil(position));
	const double fraction = position - static_cast<double>(low);
	return InSorted[low] * (1.0 - fraction) + InSorted[high] * fraction;
}

Stats Summarize(std::vector<double> InSamples) {
	std::sort(InSamples.begin(), InSamples.end());
	return Stats{
		.MedianNs = Percentile(InSamples, 0.5),
		.P10Ns = Percentile(InSamples, 0.1),
		.P90Ns = Percentile(InSamples, 0.9),
	};
}

void PinToFirstAllowedCpu() {
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

void WarmCpu() {
	const auto until = Clock::now() + std::chrono::milliseconds(300);
	volatile OaU64 value = 1;
	while (Clock::now() < until) {
		value = value * 6364136223846793005ULL + 1442695040888963407ULL;
	}
	(void)value;
}

bool Verify(
	Implementation InImplementation,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize) {
	std::memset(InDst, 0xA5, InSize);
	RunCopies(InImplementation, InDst, InSrc, InSize, 1);
	return std::memcmp(InDst, InSrc, InSize) == 0;
}

std::array<Stats, static_cast<size_t>(Implementation::Count)> Measure(
	const std::vector<Implementation>& InImplementations,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize,
	OaUsize InIterations) {
	for (Implementation implementation : InImplementations) {
		for (int sample = 0; sample < WarmupSamples; ++sample) {
			RunCopies(implementation, InDst, InSrc, InSize, InIterations);
		}
	}

	std::array<std::vector<double>, static_cast<size_t>(Implementation::Count)> samples;
	for (auto& implementationSamples : samples) {
		implementationSamples.reserve(MeasuredSamples);
	}
	for (int sample = 0; sample < MeasuredSamples; ++sample) {
		// Rotate the implementation order each sample to distribute clock,
		// temperature and cache-history drift rather than baking it into one row.
		for (size_t order = 0; order < InImplementations.size(); ++order) {
			const Implementation implementation = InImplementations[
				(order + static_cast<size_t>(sample)) % InImplementations.size()];
			const auto begin = Clock::now();
			RunCopies(implementation, InDst, InSrc, InSize, InIterations);
			const auto end = Clock::now();
			const double totalNs =
				std::chrono::duration<double, std::nano>(end - begin).count();
			samples[static_cast<size_t>(implementation)].push_back(
				totalNs / static_cast<double>(InIterations));
		}
	}
	std::array<Stats, static_cast<size_t>(Implementation::Count)> result;
	for (Implementation implementation : InImplementations) {
		result[static_cast<size_t>(implementation)] = Summarize(
			std::move(samples[static_cast<size_t>(implementation)]));
	}
	return result;
}

std::array<Stats, static_cast<size_t>(Implementation::Count)> MeasurePrimitive(
	Primitive InPrimitive,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize,
	OaUsize InIterations) {
	constexpr std::array implementations{
		Implementation::OaRuntime,
		Implementation::LibcRuntime,
	};
	for (Implementation implementation : implementations) {
		for (int sample = 0; sample < WarmupSamples; ++sample) {
			(void)RunPrimitive(InPrimitive, implementation,
				InDst, InSrc, InSize, InIterations);
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
			(void)RunPrimitive(InPrimitive, implementation,
				InDst, InSrc, InSize, InIterations);
			const auto end = Clock::now();
			const double totalNs =
				std::chrono::duration<double, std::nano>(end - begin).count();
			samples[static_cast<size_t>(implementation)].push_back(
				totalNs / static_cast<double>(InIterations));
		}
	}
	std::array<Stats, static_cast<size_t>(Implementation::Count)> result;
	for (Implementation implementation : implementations) {
		result[static_cast<size_t>(implementation)] = Summarize(
			std::move(samples[static_cast<size_t>(implementation)]));
	}
	return result;
}

bool SupportsFixed(OaUsize InSize) {
	return InSize == 8 || InSize == 16 || InSize == 32 || InSize == 64
		|| InSize == 128 || InSize == 256;
}

std::vector<Case> Cases(bool InQuick) {
	const OaUsize sizesFull[] = {
		1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64,
		65, 96, 127, 128, 129, 192, 255, 256, 257, 512, 1024, 4096,
		64 * KiB, 1 * MiB, 2 * MiB, 4 * MiB, 16 * MiB, 64 * MiB,
	};
	const OaUsize sizesQuick[] = {
		8, 16, 32, 64, 128, 256, 512, 4096,
		64 * KiB, 1 * MiB, 2 * MiB, 4 * MiB, 16 * MiB, 64 * MiB,
	};
	const OaUsize* sizes = InQuick ? sizesQuick : sizesFull;
	const size_t count = InQuick ? std::size(sizesQuick) : std::size(sizesFull);
	std::vector<Case> cases;
	for (size_t index = 0; index < count; ++index) {
		cases.push_back(Case{sizes[index], 0, 0});
		cases.push_back(Case{sizes[index], 1, 3});
	}
	return cases;
}

bool HasArg(int InArgc, char** InArgv, const char* InValue) {
	for (int index = 1; index < InArgc; ++index) {
		if (std::strcmp(InArgv[index], InValue) == 0) return true;
	}
	return false;
}

bool VerifyPrimitive(
	Primitive InPrimitive,
	Implementation InImplementation,
	OaByte* InDst,
	const OaByte* InSrc,
	OaUsize InSize) {
	const OaBool result = RunPrimitive(
		InPrimitive, InImplementation, InDst, InSrc, InSize, 1);
	switch (InPrimitive) {
		case Primitive::Fill:
			for (OaUsize index = 0; index < InSize; ++index) {
				if (InDst[index] != 0xA5) return false;
			}
			return true;
		case Primitive::Zero:
			for (OaUsize index = 0; index < InSize; ++index) {
				if (InDst[index] != 0) return false;
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
	const bool quick = HasArg(argc, argv, "--quick");
	const bool primitivesOnly = HasArg(argc, argv, "--primitives");
	const bool copyOnly = HasArg(argc, argv, "--copy");
	PinToFirstAllowedCpu();
	WarmCpu();

	std::printf(
		"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s\n");
	for (const Case& test : Cases(quick)) {
		const OaUsize allocation = test.Size + 128;
		auto* srcBase = static_cast<OaByte*>(OaAlignedAlloc(allocation, 64));
		auto* dstBase = static_cast<OaByte*>(OaAlignedAlloc(allocation, 64));
		if (!srcBase || !dstBase) {
			std::fprintf(stderr, "BenchMemory: allocation failed for %zu bytes\n",
				static_cast<size_t>(allocation));
			OaAlignedFree(srcBase);
			OaAlignedFree(dstBase);
			return 1;
		}
		for (OaUsize index = 0; index < allocation; ++index) {
			srcBase[index] = static_cast<OaByte>((index * 131U + 17U) & 0xFFU);
			dstBase[index] = 0;
		}
		auto* src = srcBase + test.SrcOffset;
		auto* dst = dstBase + test.DstOffset;
		const OaUsize iterations = IterationsFor(test.Size, quick);
		if (!primitivesOnly) {
			std::vector<Implementation> implementations{
				Implementation::OaRuntime,
				Implementation::LibcRuntime,
			};
			if (SupportsFixed(test.Size)) {
				implementations.push_back(Implementation::OaFixed);
				implementations.push_back(Implementation::CompilerFixed);
			}
			implementations.push_back(Implementation::NonTemporal);
			for (Implementation implementation : implementations) {
				if (!Verify(implementation, dst, src, test.Size)) {
					std::fprintf(stderr,
						"BenchMemory: correctness failure operation=copy size=%zu src=%zu dst=%zu impl=%s\n",
						static_cast<size_t>(test.Size),
						static_cast<size_t>(test.SrcOffset),
						static_cast<size_t>(test.DstOffset), Name(implementation));
					OaAlignedFree(srcBase);
					OaAlignedFree(dstBase);
					return 2;
				}
			}
			const auto measurements = Measure(
				implementations, dst, src, test.Size, iterations);
			for (Implementation implementation : implementations) {
				const Stats stats = measurements[static_cast<size_t>(implementation)];
				const double gbPerSecond = static_cast<double>(test.Size) / stats.MedianNs;
				std::printf("copy,%zu,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f\n",
					static_cast<size_t>(test.Size),
					static_cast<size_t>(test.SrcOffset),
					static_cast<size_t>(test.DstOffset), Name(implementation),
					static_cast<size_t>(iterations), stats.MedianNs,
					stats.P10Ns, stats.P90Ns, gbPerSecond);
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
				Implementation::OaRuntime,
				Implementation::LibcRuntime,
			};
			for (Primitive primitive : primitives) {
				std::memcpy(dst, src, test.Size);
				if (primitive == Primitive::EqualEarlyMismatch && test.Size > 0) {
					dst[0] ^= 0xFF;
				} else if (primitive == Primitive::EqualLateMismatch && test.Size > 0) {
					dst[test.Size - 1] ^= 0xFF;
				}
				for (Implementation implementation : implementations) {
					if (!VerifyPrimitive(primitive, implementation,
						dst, src, test.Size)) {
						std::fprintf(stderr,
							"BenchMemory: correctness failure operation=%s size=%zu src=%zu dst=%zu impl=%s\n",
							Name(primitive), static_cast<size_t>(test.Size),
							static_cast<size_t>(test.SrcOffset),
							static_cast<size_t>(test.DstOffset), Name(implementation));
						OaAlignedFree(srcBase);
						OaAlignedFree(dstBase);
						return 2;
					}
				}
				// Fill/zero verification changed the destination; restore the
				// comparison fixtures before measuring equality.
				if (primitive == Primitive::Equal
					|| primitive == Primitive::EqualEarlyMismatch
					|| primitive == Primitive::EqualLateMismatch) {
					std::memcpy(dst, src, test.Size);
					if (primitive == Primitive::EqualEarlyMismatch && test.Size > 0) {
						dst[0] ^= 0xFF;
					} else if (primitive == Primitive::EqualLateMismatch && test.Size > 0) {
						dst[test.Size - 1] ^= 0xFF;
					}
				}
				const auto measurements = MeasurePrimitive(
					primitive, dst, src, test.Size, iterations);
				for (Implementation implementation : implementations) {
					const Stats stats = measurements[static_cast<size_t>(implementation)];
					const double gbPerSecond =
						static_cast<double>(test.Size) / stats.MedianNs;
					std::printf("%s,%zu,%zu,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f\n",
						Name(primitive), static_cast<size_t>(test.Size),
						static_cast<size_t>(test.SrcOffset),
						static_cast<size_t>(test.DstOffset), Name(implementation),
						static_cast<size_t>(iterations), stats.MedianNs,
						stats.P10Ns, stats.P90Ns, gbPerSecond);
				}
			}
		}
		OaAlignedFree(srcBase);
		OaAlignedFree(dstBase);
	}
	return 0;
}
