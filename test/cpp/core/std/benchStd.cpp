#include <oa/core/std.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

volatile oa::U64 gSink = 0;

struct Options {
	oa::Usize items = 1U << 15U;
	oa::I32 samples = 21;
	oa::I32 warmups = 5;
};

struct Stats {
	double median = 0.0;
	double p10 = 0.0;
	double p90 = 0.0;
};

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (oa::I32 index = 1; index < inArgc; ++index) {
		if (std::strcmp(inArgv[index], "--items") == 0 and index + 1 < inArgc) {
			options.items = static_cast<oa::Usize>(std::strtoull(inArgv[++index], nullptr, 10));
		} else if (std::strcmp(inArgv[index], "--samples") == 0 and index + 1 < inArgc) {
			options.samples = static_cast<oa::I32>(std::strtol(inArgv[++index], nullptr, 10));
		} else if (std::strcmp(inArgv[index], "--warmups") == 0 and index + 1 < inArgc) {
			options.warmups = static_cast<oa::I32>(std::strtol(inArgv[++index], nullptr, 10));
		}
	}
	options.items = std::max<oa::Usize>(options.items, 64U);
	options.samples = std::max<oa::I32>(options.samples, 7);
	options.warmups = std::max<oa::I32>(options.warmups, 1);
	return options;
}

Stats summarize(std::vector<double> inSamples) {
	std::sort(inSamples.begin(), inSamples.end());
	const oa::Usize count = inSamples.size();
	return {
		inSamples[count / 2U],
		inSamples[(count - 1U) / 10U],
		inSamples[((count - 1U) * 9U) / 10U],
	};
}

template<typename Fn>
double measure(Fn&& inFn, oa::Usize inOperations) {
	const auto begin = std::chrono::steady_clock::now();
	const oa::U64 result = inFn();
	const auto end = std::chrono::steady_clock::now();
	gSink = gSink ^ result;
	const double ns = std::chrono::duration<double, std::nano>(end - begin).count();
	return ns / static_cast<double>(inOperations);
}

template<typename OaFn, typename StdFn>
bool runPair(
	const char* inName,
	oa::Usize inOperations,
	const Options& inOptions,
	OaFn&& inOaFn,
	StdFn&& inStdFn
) {
	const oa::U64 oaOracle = inOaFn();
	const oa::U64 stdOracle = inStdFn();
	if (oaOracle != stdOracle) {
		std::fprintf(stderr, "%s oracle mismatch: oa=%llu std=%llu\n", inName,
			static_cast<unsigned long long>(oaOracle),
			static_cast<unsigned long long>(stdOracle));
		return false;
	}

	for (oa::I32 warmup = 0; warmup < inOptions.warmups; ++warmup) {
		gSink = gSink ^ inOaFn();
		gSink = gSink ^ inStdFn();
	}

	std::vector<double> oaSamples;
	std::vector<double> stdSamples;
	oaSamples.reserve(static_cast<oa::Usize>(inOptions.samples));
	stdSamples.reserve(static_cast<oa::Usize>(inOptions.samples));
	for (oa::I32 sample = 0; sample < inOptions.samples; ++sample) {
		if ((sample & 1) == 0) {
			oaSamples.push_back(measure(inOaFn, inOperations));
			stdSamples.push_back(measure(inStdFn, inOperations));
		} else {
			stdSamples.push_back(measure(inStdFn, inOperations));
			oaSamples.push_back(measure(inOaFn, inOperations));
		}
	}

	const Stats oaStats = summarize(oaSamples);
	const Stats stdStats = summarize(stdSamples);
	const double ratio = stdStats.median > 0.0 ? oaStats.median / stdStats.median : 0.0;
	std::printf(
		"%-24s oa=%9.3f ns/op [%9.3f,%9.3f]  "
		"std=%9.3f ns/op [%9.3f,%9.3f]  oa/std=%7.3fx\n",
		inName,
		oaStats.median, oaStats.p10, oaStats.p90,
		stdStats.median, stdStats.p10, stdStats.p90,
		ratio);
	return true;
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	const oa::Usize hashItems = std::max<oa::Usize>(options.items / 4U, 64U);
	std::printf(
		"OA foundation benchmark: items=%zu hashItems=%zu warmups=%d samples=%d\n",
		options.items, hashItems, options.warmups, options.samples);

	std::vector<oa::U32> baseValues(options.items);
	std::vector<float> scalarValues(options.items);
	std::string text(options.items, 'a');
	for (oa::Usize index = 0; index < options.items; ++index) {
		baseValues[index] = static_cast<oa::U32>(
			(index * 2'654'435'761ULL) ^ (index >> 3U));
		scalarValues[index] = static_cast<float>(index % 4096U) * (1.0F / 1024.0F) - 2.0F;
		text[index] = static_cast<char>('a' + index % 23U);
	}

	bool passed = true;
	passed = runPair("Steady clock read", options.items, options,
		[&]() {
			oa::U64 ordered = 0;
			oa::SteadyTimePoint previous = oa::steadyNow();
			for (oa::Usize index = 0; index < options.items; ++index) {
				const oa::SteadyTimePoint current = oa::steadyNow();
				ordered += current >= previous ? 1U : 0U;
				previous = current;
			}
			return ordered;
		},
		[&]() {
			oa::U64 ordered = 0;
			auto previous = std::chrono::steady_clock::now();
			for (oa::Usize index = 0; index < options.items; ++index) {
				const auto current = std::chrono::steady_clock::now();
				ordered += current >= previous ? 1U : 0U;
				previous = current;
			}
			return ordered;
		}) and passed;

	passed = runPair("Mutex lock/unlock", options.items, options,
		[&]() {
			oa::Mutex mutex;
			oa::U64 counter = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				mutex.lock();
				++counter;
				mutex.unlock();
			}
			return counter;
		},
		[&]() {
			std::mutex mutex;
			oa::U64 counter = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				mutex.lock();
				++counter;
				mutex.unlock();
			}
			return counter;
		}) and passed;

	passed = runPair("Algorithm sort", options.items, options,
		[&]() {
			std::vector<oa::U32> values = baseValues;
			oa::sort(values.data(), values.data() + values.size());
			return static_cast<oa::U64>(values.front())
				+ values[values.size() / 2U] + values.back();
		},
		[&]() {
			std::vector<oa::U32> values = baseValues;
			std::sort(values.begin(), values.end());
			return static_cast<oa::U64>(values.front())
				+ values[values.size() / 2U] + values.back();
		}) and passed;

	passed = runPair("Algorithm find/count", options.items * 2U, options,
		[&]() {
			const oa::U32 needle = baseValues[options.items / 3U];
			const auto found = oa::find(baseValues.data(),
				baseValues.data() + baseValues.size(), needle);
			return static_cast<oa::U64>(found - baseValues.data())
				+ static_cast<oa::U64>(oa::count(baseValues.data(),
					baseValues.data() + baseValues.size(), needle));
		},
		[&]() {
			const oa::U32 needle = baseValues[options.items / 3U];
			const auto found = std::find(baseValues.begin(), baseValues.end(), needle);
			return static_cast<oa::U64>(found - baseValues.begin())
				+ static_cast<oa::U64>(std::count(baseValues.begin(), baseValues.end(), needle));
		}) and passed;

	passed = runPair("C string scan", options.items * 2U, options,
		[&]() {
			const char* found = oa::strchr(text.c_str(), 'w');
			return oa::strlen(text.c_str())
				+ static_cast<oa::U64>(found - text.c_str());
		},
		[&]() {
			const char* found = std::strchr(text.c_str(), 'w');
			return std::strlen(text.c_str())
				+ static_cast<oa::U64>(found - text.c_str());
		}) and passed;

	passed = runPair("Scalar sin/tanh", options.items * 2U, options,
		[&]() {
			float sum = 0.0F;
			for (const float value : scalarValues) {
				sum += oa::sin(value) + oa::tanh(value);
			}
			oa::U32 bits = 0;
			std::memcpy(&bits, &sum, sizeof(bits));
			return static_cast<oa::U64>(bits);
		},
		[&]() {
			float sum = 0.0F;
			for (const float value : scalarValues) {
				sum += std::sin(value) + std::tanh(value);
			}
			oa::U32 bits = 0;
			std::memcpy(&bits, &sum, sizeof(bits));
			return static_cast<oa::U64>(bits);
		}) and passed;

	passed = runPair("Vec reserved push", options.items, options,
		[&]() {
			oa::Vec<oa::U64> values;
			values.reserve(options.items);
			for (oa::Usize index = 0; index < options.items; ++index) {
				values.pushBack(static_cast<oa::U64>(index * 33U + 17U));
			}
			return values.size() + values[options.items / 2U] + values.back();
		},
		[&]() {
			std::vector<oa::U64> values;
			values.reserve(options.items);
			for (oa::Usize index = 0; index < options.items; ++index) {
				values.push_back(static_cast<oa::U64>(index * 33U + 17U));
			}
			return values.size() + values[options.items / 2U] + values.back();
		}) and passed;

	passed = runPair("Vec geometric growth", options.items, options,
		[&]() {
			oa::Vec<oa::U32> values;
			for (oa::Usize index = 0; index < options.items; ++index) {
				values.pushBack(static_cast<oa::U32>(index ^ 0x5A5AU));
			}
			return static_cast<oa::U64>(values.size()) + values.back();
		},
		[&]() {
			std::vector<oa::U32> values;
			for (oa::Usize index = 0; index < options.items; ++index) {
				values.push_back(static_cast<oa::U32>(index ^ 0x5A5AU));
			}
			return static_cast<oa::U64>(values.size()) + values.back();
		}) and passed;

	passed = runPair("String reserved append", options.items, options,
		[&]() {
			oa::String value;
			value.reserve(options.items);
			for (oa::Usize index = 0; index < options.items; ++index) {
				value.pushBack(static_cast<char>('a' + index % 23U));
			}
			return value.size()
				+ static_cast<unsigned char>(value.data()[options.items / 2U])
				+ static_cast<unsigned char>(value.data()[options.items - 1U]);
		},
		[&]() {
			std::string value;
			value.reserve(options.items);
			for (oa::Usize index = 0; index < options.items; ++index) {
				value.push_back(static_cast<char>('a' + index % 23U));
			}
			return value.size()
				+ static_cast<unsigned char>(value[options.items / 2U])
				+ static_cast<unsigned char>(value.back());
		}) and passed;

	passed = runPair("HashMap insert", hashItems, options,
		[&]() {
			oa::HashMap<oa::U64, oa::U64> values;
			values.reserve(hashItems);
			for (oa::Usize index = 0; index < hashItems; ++index) {
				values.emplace(index * 17U, index * 31U + 7U);
			}
			return values.size() + values.at((hashItems / 2U) * 17U);
		},
		[&]() {
			std::unordered_map<oa::U64, oa::U64> values;
			values.reserve(hashItems);
			for (oa::Usize index = 0; index < hashItems; ++index) {
				values.emplace(index * 17U, index * 31U + 7U);
			}
			return values.size() + values.at((hashItems / 2U) * 17U);
		}) and passed;

	oa::HashMap<oa::U64, oa::U64> oaLookup;
	std::unordered_map<oa::U64, oa::U64> stdLookup;
	oaLookup.reserve(hashItems);
	stdLookup.reserve(hashItems);
	for (oa::Usize index = 0; index < hashItems; ++index) {
		oaLookup.emplace(index * 17U, index * 31U + 7U);
		stdLookup.emplace(index * 17U, index * 31U + 7U);
	}
	passed = runPair("HashMap successful find", hashItems, options,
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < hashItems; ++index) {
				sum += oaLookup.find(index * 17U)->second;
			}
			return sum;
		},
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < hashItems; ++index) {
				sum += stdLookup.find(index * 17U)->second;
			}
			return sum;
		}) and passed;

	passed = runPair("Shared make and release", options.items, options,
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				auto value = oa::makeShared<oa::U64>(index + 1U);
				sum += *value;
			}
			return sum;
		},
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				auto value = std::make_shared<oa::U64>(index + 1U);
				sum += *value;
			}
			return sum;
		}) and passed;

	auto oaOwner = oa::makeShared<oa::U64>(79U);
	auto stdOwner = std::make_shared<oa::U64>(79U);
	passed = runPair("Shared copy and release", options.items, options,
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				oa::SharedPtr<oa::U64> copy = oaOwner;
				sum += *copy;
			}
			return sum + static_cast<oa::U64>(oaOwner.useCount());
		},
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				std::shared_ptr<oa::U64> copy = stdOwner;
				sum += *copy;
			}
			return sum + static_cast<oa::U64>(stdOwner.use_count());
		}) and passed;

	std::printf("checksum=%llu status=%s\n",
		static_cast<unsigned long long>(gSink), passed ? "PASS" : "FAIL");
	return passed ? 0 : 2;
}
