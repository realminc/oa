#include <oa/core/simd.h>

#include <xsimd/xsimd.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile oa::F32 gSink = 0.0F;

struct Options {
	oa::Usize items = 1U << 20U;
	oa::I32 warmups = 3;
	oa::I32 samples = 15;
};

[[nodiscard]] bool parsePositive(const char* inValue, oa::Usize& outValue) {
	errno = 0;
	char* end = nullptr;
	const unsigned long long value = std::strtoull(inValue, &end, 10);
	if (errno == ERANGE or end == inValue or *end != '\0' or value == 0
		or value > std::numeric_limits<oa::Usize>::max()
		or value > static_cast<unsigned long long>(
			std::numeric_limits<oa::I64>::max())) {
		return false;
	}
	outValue = static_cast<oa::Usize>(value);
	return true;
}

[[nodiscard]] bool parsePositive(const char* inValue, oa::I32& outValue) {
	errno = 0;
	char* end = nullptr;
	const long value = std::strtol(inValue, &end, 10);
	if (errno == ERANGE or end == inValue or *end != '\0' or value <= 0
		or value > std::numeric_limits<oa::I32>::max()) {
		return false;
	}
	outValue = static_cast<oa::I32>(value);
	return true;
}

[[nodiscard]] bool parseOptions(
	int inArgc,
	char** inArgv,
	Options& outOptions) {
	for (int index = 1; index < inArgc; ++index) {
		const char* argument = inArgv[index];
		if (std::strcmp(argument, "--items") == 0 and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.items)) return false;
		} else if (std::strcmp(argument, "--warmups") == 0
			and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.warmups)) return false;
		} else if (std::strcmp(argument, "--samples") == 0
			and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.samples)) return false;
		} else {
			return false;
		}
	}
	return outOptions.samples >= 3;
}

[[nodiscard]] oa::F64 dotOracle(
	const std::vector<oa::F32>& inA,
	const std::vector<oa::F32>& inB) {
	oa::F64 result = 0.0;
	for (oa::Usize index = 0; index < inA.size(); ++index) {
		result += static_cast<oa::F64>(inA[index])
			* static_cast<oa::F64>(inB[index]);
	}
	return result;
}

[[nodiscard]] oa::F32 stockXsimdDot(
	const std::vector<oa::F32>& inA,
	const std::vector<oa::F32>& inB) {
	using Batch = xsimd::batch<oa::F32>;
	constexpr oa::Usize lanes = Batch::size;
	Batch sum(0.0F);
	oa::Usize index = 0;
	for (; index + lanes <= inA.size(); index += lanes) {
		const Batch a = Batch::load_unaligned(inA.data() + index);
		const Batch b = Batch::load_unaligned(inB.data() + index);
		sum = xsimd::fma(a, b, sum);
	}
	oa::F32 result = xsimd::reduce_add(sum);
	for (; index < inA.size(); ++index) result += inA[index] * inB[index];
	return result;
}

[[nodiscard]] oa::F64 median(std::vector<oa::F64> inValues) {
	std::sort(inValues.begin(), inValues.end());
	return inValues[inValues.size() / 2U];
}

template <typename Fn>
[[nodiscard]] oa::F64 measure(Fn&& inFunction, oa::Usize inItems) {
	const auto begin = Clock::now();
	const oa::F32 checksum = inFunction();
	const auto end = Clock::now();
	gSink = checksum;
	return std::chrono::duration<oa::F64, std::nano>(end - begin).count()
		/ static_cast<oa::F64>(inItems);
}

} // namespace

int main(int argc, char** argv) {
	Options options{};
	if (not parseOptions(argc, argv, options)) {
		std::fprintf(stderr,
			"usage: %s [--items N] [--warmups N] [--samples N]\n",
			argv[0]);
		return 2;
	}
	std::vector<oa::F32> a(options.items);
	std::vector<oa::F32> b(options.items);
	for (oa::Usize index = 0; index < options.items; ++index) {
		const oa::F32 value =
			static_cast<oa::F32>(index % 1021U) * (1.0F / 1021.0F);
		a[index] = value + 0.125F;
		b[index] = 1.25F - value * 0.5F;
	}

	const oa::F64 expected = dotOracle(a, b);
	const oa::F32 stock = stockXsimdDot(a, b);
	const oa::F32 product = oa::FnSimd::dotF32(
		a.data(), b.data(), static_cast<oa::I64>(options.items));
	const oa::F64 tolerance = oa::max(1.0, oa::abs(expected)) * 5.0e-4;
	if (oa::abs(static_cast<oa::F64>(stock) - expected) > tolerance
		or oa::abs(static_cast<oa::F64>(product) - expected) > tolerance) {
		std::fprintf(stderr,
			"dot oracle mismatch: expected=%g stock_xsimd=%g fnsimd=%g\n",
			expected, static_cast<oa::F64>(stock),
			static_cast<oa::F64>(product));
		return 3;
	}

	const auto runProduct = [&]() {
		return oa::FnSimd::dotF32(
			a.data(), b.data(), static_cast<oa::I64>(options.items));
	};
	const auto runStock = [&]() { return stockXsimdDot(a, b); };
	for (oa::I32 warmup = 0; warmup < options.warmups; ++warmup) {
		if ((warmup & 1) == 0) {
			gSink = runProduct();
			gSink = runStock();
		} else {
			gSink = runStock();
			gSink = runProduct();
		}
	}

	std::vector<oa::F64> productTimes;
	std::vector<oa::F64> stockTimes;
	productTimes.reserve(static_cast<oa::Usize>(options.samples));
	stockTimes.reserve(static_cast<oa::Usize>(options.samples));
	for (oa::I32 sample = 0; sample < options.samples; ++sample) {
		if ((sample & 1) == 0) {
			productTimes.push_back(measure(runProduct, options.items));
			stockTimes.push_back(measure(runStock, options.items));
		} else {
			stockTimes.push_back(measure(runStock, options.items));
			productTimes.push_back(measure(runProduct, options.items));
		}
	}
	const oa::F64 productMedian = median(productTimes);
	const oa::F64 stockMedian = median(stockTimes);
	std::printf(
		"PAIR case=dot_f32 contract=equivalent oa_ns=%.6f xsimd_ns=%.6f ratio=%.6f checksum=%.9g\n",
		productMedian, stockMedian, productMedian / stockMedian,
		static_cast<oa::F64>(product));
	std::printf("BENCHMARK oracle=PASS\n");
	return 0;
}
