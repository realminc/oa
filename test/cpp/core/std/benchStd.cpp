#include <oa/core/std.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
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

struct ConstantHash {
	[[nodiscard]] oa::Usize operator()(oa::U64 /*inValue*/) const noexcept {
		return 0;
	}
};

[[noreturn]] void optionError(const char* inMessage, const char* inArgument = nullptr) {
	if (inArgument != nullptr) {
		std::fprintf(stderr, "benchStd: %s: %s\n", inMessage, inArgument);
	} else {
		std::fprintf(stderr, "benchStd: %s\n", inMessage);
	}
	std::fprintf(stderr,
		"usage: benchStd [--items N] [--warmups N] [--samples N]\n"
		"       items >= 64, warmups >= 1, samples >= 7\n");
	std::exit(2);
}

oa::U64 parseUnsigned(const char* inText, const char* inOption) {
	if (inText == nullptr or inText[0] == '\0' or inText[0] == '-') {
		optionError("invalid non-negative integer", inOption);
	}
	errno = 0;
	char* end = nullptr;
	const unsigned long long value = std::strtoull(inText, &end, 10);
	if (errno == ERANGE or end == inText or *end != '\0') {
		optionError("invalid non-negative integer", inOption);
	}
	return static_cast<oa::U64>(value);
}

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (oa::I32 index = 1; index < inArgc; ++index) {
		const char* option = inArgv[index];
		if (std::strcmp(option, "--help") == 0) {
			std::printf("usage: benchStd [--items N] [--warmups N] [--samples N]\n");
			std::exit(0);
		}
		const bool recognized = std::strcmp(option, "--items") == 0
			or std::strcmp(option, "--samples") == 0
			or std::strcmp(option, "--warmups") == 0;
		if (not recognized) {
			optionError("unknown option", option);
		}
		if (index + 1 >= inArgc) {
			optionError("missing option value", option);
		}
		const oa::U64 value = parseUnsigned(inArgv[++index], option);
		if (std::strcmp(option, "--items") == 0) {
			if (value < 64U or value > static_cast<oa::U64>(static_cast<oa::Usize>(-1) / 2U)) {
				optionError("--items is outside the supported range", option);
			}
			options.items = static_cast<oa::Usize>(value);
		} else if (std::strcmp(option, "--samples") == 0) {
			if (value < 7U or value > static_cast<oa::U64>(INT_MAX)) {
				optionError("--samples is outside the supported range", option);
			}
			options.samples = static_cast<oa::I32>(value);
		} else if (std::strcmp(option, "--warmups") == 0) {
			if (value < 1U or value > static_cast<oa::U64>(INT_MAX)) {
				optionError("--warmups is outside the supported range", option);
			}
			options.warmups = static_cast<oa::I32>(value);
		}
	}
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

template<typename Fn>
bool runPreflight(const char* inName, Fn&& inFn) {
	if (inFn()) {
		return true;
	}
	std::fprintf(stderr, "%s full-content preflight failed\n", inName);
	return false;
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	const oa::Usize hashItems = std::max<oa::Usize>(options.items / 4U, 64U);
	const oa::Usize collisionItems = std::min<oa::Usize>(hashItems, 512U);
	std::printf(
		"OA foundation benchmark: items=%zu hashItems=%zu collisionItems=%zu "
		"warmups=%d samples=%d\n",
		options.items, hashItems, collisionItems, options.warmups, options.samples);

	std::vector<oa::U32> baseValues(options.items);
	std::vector<oa::U64> pushValues(options.items);
	std::vector<float> scalarValues(options.items);
	std::string text(options.items, 'a');
	for (oa::Usize index = 0; index < options.items; ++index) {
		baseValues[index] = static_cast<oa::U32>(
			(index * 2'654'435'761ULL) ^ (index >> 3U));
		pushValues[index] = static_cast<oa::U64>(index * 33U + 17U);
		scalarValues[index] = static_cast<float>(index % 4096U) * (1.0F / 1024.0F) - 2.0F;
		text[index] = static_cast<char>('a' + index % 23U);
	}
	const std::string textCopy = text;

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

	oa::Array<oa::U64, 256> oaArray;
	std::array<oa::U64, 256> stdArray{};
	for (oa::Usize index = 0; index < oaArray.size(); ++index) {
		const oa::U64 value = static_cast<oa::U64>(index * 97U + 11U);
		oaArray[index] = value;
		stdArray[index] = value;
	}
	passed = runPair("Array indexed read", options.items, options,
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				sum += oaArray[index & 255U];
			}
			return sum;
		},
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				sum += stdArray[index & 255U];
			}
			return sum;
		}) and passed;

	passed = runPair("Span traversal", options.items, options,
		[&]() {
			const oa::Span<const oa::U32> values(baseValues.data(), baseValues.size());
			oa::U64 sum = 0;
			for (const oa::U32 value : values) {
				sum += value;
			}
			return sum;
		},
		[&]() {
			const std::span<const oa::U32> values(baseValues.data(), baseValues.size());
			oa::U64 sum = 0;
			for (const oa::U32 value : values) {
				sum += value;
			}
			return sum;
		}) and passed;

	passed = runPair("Atomic relaxed add", options.items, options,
		[&]() {
			oa::Atomic<oa::U64> value{0};
			for (oa::Usize index = 0; index < options.items; ++index) {
				(void)value.fetchAdd(1U, oa::MemoryOrder::Relaxed);
			}
			return value.load(oa::MemoryOrder::Relaxed);
		},
		[&]() {
			std::atomic<oa::U64> value{0};
			for (oa::Usize index = 0; index < options.items; ++index) {
				(void)value.fetch_add(1U, std::memory_order_relaxed);
			}
			return value.load(std::memory_order_relaxed);
		}) and passed;

	passed = runPair("Optional lifecycle", options.items, options,
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				oa::Optional<oa::U64> value(static_cast<oa::U64>(index + 1U));
				sum += *value;
				if ((index & 1U) == 0U) {
					value.emplace(static_cast<oa::U64>(index + 2U));
				} else {
					value.reset();
				}
				sum += value.valueOr(0U);
			}
			return sum;
		},
		[&]() {
			oa::U64 sum = 0;
			for (oa::Usize index = 0; index < options.items; ++index) {
				std::optional<oa::U64> value(static_cast<oa::U64>(index + 1U));
				sum += *value;
				if ((index & 1U) == 0U) {
					value.emplace(static_cast<oa::U64>(index + 2U));
				} else {
					value.reset();
				}
				sum += value.value_or(0U);
			}
			return sum;
		}) and passed;

	passed = runPreflight("Algorithm sort", [&]() {
		std::vector<oa::U32> oaValues = baseValues;
		std::vector<oa::U32> stdValues = baseValues;
		oa::sort(oaValues.data(), oaValues.data() + oaValues.size());
		std::sort(stdValues.begin(), stdValues.end());
		return oaValues == stdValues;
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

	passed = runPair("C strlen / byte", options.items, options,
		[&]() {
			return oa::strlen(text.c_str());
		},
		[&]() {
			return std::strlen(text.c_str());
		}) and passed;

	passed = runPair("C strchr miss / byte", options.items, options,
		[&]() {
			const char* found = oa::strchr(text.c_str(), '\x7f');
			return found == nullptr
				? static_cast<oa::U64>(text.size())
				: static_cast<oa::U64>(found - text.c_str());
		},
		[&]() {
			const char* found = std::strchr(text.c_str(), '\x7f');
			return found == nullptr
				? static_cast<oa::U64>(text.size())
				: static_cast<oa::U64>(found - text.c_str());
		}) and passed;

	passed = runPair("C strcmp equal / byte", options.items, options,
		[&]() {
			return static_cast<oa::U64>(oa::strcmp(text.c_str(), textCopy.c_str()) == 0);
		},
		[&]() {
			return static_cast<oa::U64>(std::strcmp(text.c_str(), textCopy.c_str()) == 0);
		}) and passed;

	passed = runPair("C strncmp equal / byte", options.items, options,
		[&]() {
			return static_cast<oa::U64>(
				oa::strncmp(text.c_str(), textCopy.c_str(), text.size()) == 0);
		},
		[&]() {
			return static_cast<oa::U64>(
				std::strncmp(text.c_str(), textCopy.c_str(), text.size()) == 0);
		}) and passed;

	passed = runPair("StringView find / byte", options.items, options,
		[&]() {
			const oa::StringView view(text.data(), text.size());
			const oa::Usize found = view.find('\x7f');
			return found == oa::StringView::Npos
				? static_cast<oa::U64>(view.size())
				: static_cast<oa::U64>(found);
		},
		[&]() {
			const std::string_view view(text.data(), text.size());
			const std::size_t found = view.find('\x7f');
			return found == std::string_view::npos
				? static_cast<oa::U64>(view.size())
				: static_cast<oa::U64>(found);
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

	passed = runPreflight("Vector reserved push", [&]() {
		oa::Vector<oa::U64> values;
		values.reserve(pushValues.size());
		for (const oa::U64 value : pushValues) {
			values.pushBack(value);
		}
		if (values.size() != pushValues.size()) {
			return false;
		}
		for (oa::Usize index = 0; index < values.size(); ++index) {
			if (values[index] != pushValues[index]) {
				return false;
			}
		}
		return true;
	}) and passed;
	passed = runPair("Vector reserved push", options.items, options,
		[&]() {
			oa::Vector<oa::U64> values;
			values.reserve(options.items);
			for (const oa::U64 value : pushValues) {
				values.pushBack(value);
			}
			return values.size() + values[options.items / 2U] + values.back();
		},
		[&]() {
			std::vector<oa::U64> values;
			values.reserve(options.items);
			for (const oa::U64 value : pushValues) {
				values.push_back(value);
			}
			return values.size() + values[options.items / 2U] + values.back();
		}) and passed;

	passed = runPreflight("Vector geometric growth", [&]() {
		oa::Vector<oa::U32> values;
		for (const oa::U32 value : baseValues) {
			values.pushBack(value);
		}
		if (values.size() != baseValues.size()) {
			return false;
		}
		for (oa::Usize index = 0; index < values.size(); ++index) {
			if (values[index] != baseValues[index]) {
				return false;
			}
		}
		return true;
	}) and passed;
	passed = runPair("Vector geometric growth", options.items, options,
		[&]() {
			oa::Vector<oa::U32> values;
			for (const oa::U32 value : baseValues) {
				values.pushBack(value);
			}
			return static_cast<oa::U64>(values.size()) + values.back();
		},
		[&]() {
			std::vector<oa::U32> values;
			for (const oa::U32 value : baseValues) {
				values.push_back(value);
			}
			return static_cast<oa::U64>(values.size()) + values.back();
		}) and passed;

	passed = runPreflight("Vector bulk append", [&]() {
		oa::Vector<oa::U32> values;
		values.append(baseValues.data(), baseValues.size());
		if (values.size() != baseValues.size()) {
			return false;
		}
		for (oa::Usize index = 0; index < values.size(); ++index) {
			if (values[index] != baseValues[index]) {
				return false;
			}
		}
		return true;
	}) and passed;
	passed = runPair("Vector bulk append / elem", options.items, options,
		[&]() {
			oa::Vector<oa::U32> values;
			values.append(baseValues.data(), baseValues.size());
			return static_cast<oa::U64>(values.size()) + values.front() + values.back();
		},
		[&]() {
			std::vector<oa::U32> values;
			values.insert(values.end(), baseValues.begin(), baseValues.end());
			return static_cast<oa::U64>(values.size()) + values.front() + values.back();
		}) and passed;

	passed = runPreflight("String reserved push", [&]() {
		oa::String value;
		value.reserve(text.size());
		for (const char character : text) {
			value.pushBack(character);
		}
		return value.size() == text.size()
			and value.data()[value.size()] == '\0'
			and std::memcmp(value.data(), text.data(), text.size()) == 0;
	}) and passed;
	passed = runPair("String reserved push", options.items, options,
		[&]() {
			oa::String value;
			value.reserve(options.items);
			for (const char character : text) {
				value.pushBack(character);
			}
			return value.size()
				+ static_cast<unsigned char>(value.data()[options.items / 2U])
				+ static_cast<unsigned char>(value.data()[options.items - 1U]);
		},
		[&]() {
			std::string value;
			value.reserve(options.items);
			for (const char character : text) {
				value.push_back(character);
			}
			return value.size()
				+ static_cast<unsigned char>(value[options.items / 2U])
				+ static_cast<unsigned char>(value.back());
		}) and passed;

	passed = runPreflight("String geometric growth", [&]() {
		oa::String value;
		for (const char character : text) {
			value.pushBack(character);
		}
		return value.size() == text.size()
			and value.data()[value.size()] == '\0'
			and std::memcmp(value.data(), text.data(), text.size()) == 0;
	}) and passed;
	passed = runPair("String geometric growth", options.items, options,
		[&]() {
			oa::String value;
			for (const char character : text) {
				value.pushBack(character);
			}
			return value.size()
				+ static_cast<unsigned char>(value.front())
				+ static_cast<unsigned char>(value.back());
		},
		[&]() {
			std::string value;
			for (const char character : text) {
				value.push_back(character);
			}
			return value.size()
				+ static_cast<unsigned char>(value.front())
				+ static_cast<unsigned char>(value.back());
		}) and passed;

	passed = runPreflight("String bulk append", [&]() {
		oa::String value;
		value.append(oa::StringView(text.data(), text.size()));
		return value.size() == text.size()
			and value.data()[value.size()] == '\0'
			and std::memcmp(value.data(), text.data(), text.size()) == 0;
	}) and passed;
	passed = runPair("String bulk append / byte", options.items, options,
		[&]() {
			oa::String value;
			value.append(oa::StringView(text.data(), text.size()));
			return value.size()
				+ static_cast<unsigned char>(value.front())
				+ static_cast<unsigned char>(value.back());
		},
		[&]() {
			std::string value;
			value.append(text.data(), text.size());
			return value.size()
				+ static_cast<unsigned char>(value.front())
				+ static_cast<unsigned char>(value.back());
		}) and passed;

	passed = runPair("Variant emplace/visit", options.items, options,
		[&]() {
			oa::Variant<oa::U64, oa::U32> value{oa::U64{0}};
			oa::U64 sum = 0;
			for (const oa::U32 input : baseValues) {
				if ((input & 1U) == 0U) {
					value.emplace<oa::U64>(static_cast<oa::U64>(input));
				} else {
					value.emplace<oa::U32>(input);
				}
				value.visit([&sum](const auto stored) {
					sum += static_cast<oa::U64>(stored);
				});
			}
			return sum;
		},
		[&]() {
			std::variant<oa::U64, oa::U32> value{oa::U64{0}};
			oa::U64 sum = 0;
			for (const oa::U32 input : baseValues) {
				if ((input & 1U) == 0U) {
					value.emplace<oa::U64>(static_cast<oa::U64>(input));
				} else {
					value.emplace<oa::U32>(input);
				}
				sum += std::visit([](const auto stored) {
					return static_cast<oa::U64>(stored);
				}, value);
			}
			return sum;
		}) and passed;

	const auto transform = [](oa::U64 inValue) noexcept {
		return (inValue * 33U) ^ (inValue >> 7U);
	};
	oa::Fn<oa::U64(oa::U64)> oaFunction(transform);
	std::function<oa::U64(oa::U64)> stdFunction(transform);
	passed = runPair("Function invoke", options.items, options,
		[&]() {
			oa::U64 sum = 0;
			for (const oa::U32 value : baseValues) {
				sum += oaFunction(value);
			}
			return sum;
		},
		[&]() {
			oa::U64 sum = 0;
			for (const oa::U32 value : baseValues) {
				sum += stdFunction(value);
			}
			return sum;
		}) and passed;

	passed = runPreflight("HashMap insert", [&]() {
		oa::HashMap<oa::U64, oa::U64> values;
		values.reserve(hashItems);
		for (oa::Usize index = 0; index < hashItems; ++index) {
			values.emplace(index * 17U, index * 31U + 7U);
		}
		if (values.size() != hashItems) {
			return false;
		}
		for (oa::Usize index = 0; index < hashItems; ++index) {
			const auto found = values.find(index * 17U);
			if (found == values.end() or found->second != index * 31U + 7U) {
				return false;
			}
		}
		return true;
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

	passed = runPreflight("HashSet insert", [&]() {
		oa::HashSet<oa::U64> values;
		values.reserve(hashItems);
		for (oa::Usize index = 0; index < hashItems; ++index) {
			values.insert(index * 17U);
		}
		if (values.size() != hashItems) {
			return false;
		}
		for (oa::Usize index = 0; index < hashItems; ++index) {
			if (not values.contains(index * 17U)) {
				return false;
			}
		}
		return true;
	}) and passed;
	passed = runPair("HashSet insert", hashItems, options,
		[&]() {
			oa::HashSet<oa::U64> values;
			values.reserve(hashItems);
			for (oa::Usize index = 0; index < hashItems; ++index) {
				values.insert(index * 17U);
			}
			return values.size()
				+ static_cast<oa::U64>(values.contains((hashItems / 2U) * 17U));
		},
		[&]() {
			std::unordered_set<oa::U64> values;
			values.reserve(hashItems);
			for (oa::Usize index = 0; index < hashItems; ++index) {
				values.insert(index * 17U);
			}
			return values.size()
				+ static_cast<oa::U64>(values.contains((hashItems / 2U) * 17U));
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

	oa::HashSet<oa::U64> oaSetLookup;
	std::unordered_set<oa::U64> stdSetLookup;
	oaSetLookup.reserve(hashItems);
	stdSetLookup.reserve(hashItems);
	for (oa::Usize index = 0; index < hashItems; ++index) {
		oaSetLookup.insert(index * 17U);
		stdSetLookup.insert(index * 17U);
	}
	passed = runPair("HashSet successful find", hashItems, options,
		[&]() {
			oa::U64 found = 0;
			for (oa::Usize index = 0; index < hashItems; ++index) {
				found += static_cast<oa::U64>(oaSetLookup.contains(index * 17U));
			}
			return found;
		},
		[&]() {
			oa::U64 found = 0;
			for (oa::Usize index = 0; index < hashItems; ++index) {
				found += static_cast<oa::U64>(stdSetLookup.contains(index * 17U));
			}
			return found;
		}) and passed;

	passed = runPreflight("HashMap forced collisions", [&]() {
		oa::HashMap<oa::U64, oa::U64, ConstantHash> values;
		values.reserve(collisionItems);
		for (oa::Usize index = 0; index < collisionItems; ++index) {
			values.emplace(index, index * 5U + 3U);
		}
		if (values.size() != collisionItems) {
			return false;
		}
		for (oa::Usize index = 0; index < collisionItems; ++index) {
			const auto found = values.find(index);
			if (found == values.end() or found->second != index * 5U + 3U) {
				return false;
			}
		}
		return true;
	}) and passed;
	passed = runPair("HashMap collision insert", collisionItems, options,
		[&]() {
			oa::HashMap<oa::U64, oa::U64, ConstantHash> values;
			values.reserve(collisionItems);
			for (oa::Usize index = 0; index < collisionItems; ++index) {
				values.emplace(index, index * 5U + 3U);
			}
			return values.size() + values.at(collisionItems - 1U);
		},
		[&]() {
			std::unordered_map<oa::U64, oa::U64, ConstantHash> values;
			values.reserve(collisionItems);
			for (oa::Usize index = 0; index < collisionItems; ++index) {
				values.emplace(index, index * 5U + 3U);
			}
			return values.size() + values.at(collisionItems - 1U);
		}) and passed;

	oa::HashMap<oa::U64, oa::U64, ConstantHash> oaCollisionLookup;
	std::unordered_map<oa::U64, oa::U64, ConstantHash> stdCollisionLookup;
	oaCollisionLookup.reserve(collisionItems);
	stdCollisionLookup.reserve(collisionItems);
	for (oa::Usize index = 0; index < collisionItems; ++index) {
		oaCollisionLookup.emplace(index, index);
		stdCollisionLookup.emplace(index, index);
	}
	passed = runPair("HashMap collision miss", collisionItems, options,
		[&]() {
			oa::U64 missing = 0;
			for (oa::Usize index = 0; index < collisionItems; ++index) {
				missing += static_cast<oa::U64>(
				oaCollisionLookup.find(collisionItems + index) == oaCollisionLookup.end());
			}
			return missing;
		},
		[&]() {
			oa::U64 missing = 0;
			for (oa::Usize index = 0; index < collisionItems; ++index) {
				missing += static_cast<oa::U64>(
					stdCollisionLookup.find(collisionItems + index) == stdCollisionLookup.end());
			}
			return missing;
		}) and passed;

	const auto runSharedPairs = [&](const char* inMakeName, const char* inCopyName) {
		passed = runPair(inMakeName, options.items, options,
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
		passed = runPair(inCopyName, options.items, options,
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
	};

	// glibc-backed libstdc++ uses non-atomic reference counts until the process
	// has created a thread. Measure both states: OA keeps the same thread-safe
	// contract in each, while real OA applications normally enter the latter.
	runSharedPairs("Shared make (single)", "Shared copy (single)");
	std::thread([] {}).join();
	runSharedPairs("Shared make (threaded)", "Shared copy (threaded)");

	std::printf("checksum=%llu status=%s\n",
		static_cast<unsigned long long>(gSink), passed ? "PASS" : "FAIL");
	return passed ? 0 : 2;
}
