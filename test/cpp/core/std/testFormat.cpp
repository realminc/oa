#include "oaStdTest.h"

#include <oa/core/std/format.h>
#include <oa/core/std/print.h>

#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <thread>

#if defined(_WIN32)
	#include <io.h>
#else
	#include <unistd.h>
#endif

namespace {

template<typename T>
concept AcceptsRuntimeFormat = requires(T inFormat) {
	oa::format(inFormat, 7);
};

static_assert(not AcceptsRuntimeFormat<oa::StringView>);

void expectSameText(const oa::String& inActual, const std::string& inExpected) {
	ASSERT_EQ(inActual.size(), inExpected.size());
	EXPECT_EQ(std::string(inActual.data(), inActual.size()), inExpected);
}

[[nodiscard]] int duplicateDescriptor(int inDescriptor) {
#if defined(_WIN32)
	return ::_dup(inDescriptor);
#else
	return ::dup(inDescriptor);
#endif
}

[[nodiscard]] int replaceDescriptor(int inSource, int inDestination) {
#if defined(_WIN32)
	return ::_dup2(inSource, inDestination);
#else
	return ::dup2(inSource, inDestination);
#endif
}

void closeDescriptor(int inDescriptor) {
#if defined(_WIN32)
	(void)::_close(inDescriptor);
#else
	(void)::close(inDescriptor);
#endif
}

[[nodiscard]] int fileDescriptor(::FILE* inFile) {
#if defined(_WIN32)
	return ::_fileno(inFile);
#else
	return ::fileno(inFile);
#endif
}

[[nodiscard]] std::string captureStdout(void (*inOperation)()) {
	(void)::fflush(stdout);
	::FILE* temporary = ::tmpfile();
	if (temporary == nullptr) return {};
	const int standardDescriptor = fileDescriptor(stdout);
	const int saved = duplicateDescriptor(standardDescriptor);
	if (saved < 0 or replaceDescriptor(fileDescriptor(temporary), standardDescriptor) < 0) {
		if (saved >= 0) closeDescriptor(saved);
		(void)::fclose(temporary);
		return {};
	}
	inOperation();
	(void)::fflush(stdout);
	(void)replaceDescriptor(saved, standardDescriptor);
	closeDescriptor(saved);
	(void)::fseek(temporary, 0L, SEEK_END);
	const long length = ::ftell(temporary);
	(void)::fseek(temporary, 0L, SEEK_SET);
	std::string result(length > 0 ? static_cast<std::size_t>(length) : 0U, '\0');
	if (not result.empty()) {
		(void)::fread(result.data(), 1U, result.size(), temporary);
	}
	(void)::fclose(temporary);
	return result;
}

} // namespace

TEST(format, IntegerToString) {
	EXPECT_STREQ(oa::toString(oa::U32{0}).cStr(), "0");
	EXPECT_STREQ(oa::toString(oa::U32{4294967295u}).cStr(), "4294967295");
	EXPECT_STREQ(oa::toString(oa::U64{18446744073709551615ULL}).cStr(),
		"18446744073709551615");
	EXPECT_STREQ(oa::toString(oa::I64{-9223372036854775807LL}).cStr(), "-9223372036854775807");
}

TEST(format, FloatToString) {
	EXPECT_STREQ(oa::toString(0.0).cStr(), "0");
	EXPECT_STREQ(oa::toString(1.5).cStr(), "1.5");
	EXPECT_STREQ(oa::toString(1.5F).cStr(), "1.5");   // float overload
	EXPECT_STREQ(oa::toString(100.0).cStr(), "100");  // %g stays compact
	EXPECT_STREQ(oa::toString(-2.25).cStr(), "-2.25");
}

TEST(format, Basic) {
	oa::String s = oa::format("{}={} ({:.2f})", "x", 42, 3.14159);
	EXPECT_STREQ(s.cStr(), "x=42 (3.14)");
	EXPECT_EQ(s.size(), std::strlen("x=42 (3.14)"));
}

TEST(format, Empty) {
	oa::String s = oa::format("{}", "");
	EXPECT_EQ(s.size(), 0u);
	EXPECT_STREQ(s.cStr(), "");
}

TEST(format, LongExceedsStackBuffer) {
	std::string big(1000, 'a');
	oa::String s = oa::format("{}", big.c_str());
	EXPECT_EQ(s.size(), 1000u);
	EXPECT_STREQ(s.cStr(), big.c_str());
}

TEST(format, BraceEscapingAndTypes) {
	enum class Mode : oa::U8 { Fast = 7 };
	const oa::String text = oa::format(
		"{{{} {:#x} {:08d} {} {} {}}}",
		"oa", 255U, -42, true, 'x', Mode::Fast);
	EXPECT_STREQ(text.cStr(), "{oa 0xff -0000042 true x 7}");
}

TEST(format, AlignmentAndPrecision) {
	EXPECT_STREQ(oa::format("{:>5}", "oa").cStr(), "   oa");
	EXPECT_STREQ(oa::format("{:*^6}", "oa").cStr(), "**oa**");
	EXPECT_STREQ(oa::format("{:.3s}", "abcdef").cStr(), "abc");
	EXPECT_STREQ(oa::format("{:+.2f}", 1.25).cStr(), "+1.25");
}

TEST(format, IntegerSpecsMatchStdFormatAcrossExtrema) {
	constexpr oa::I64 signedValues[]{
		oa::Limits<oa::I64>::min(), -65536, -1, 0, 1, 65536,
		oa::Limits<oa::I64>::max(),
	};
	for (const oa::I64 value : signedValues) {
		expectSameText(oa::format("{}", value), std::format("{}", value));
		expectSameText(oa::format("{:+024d}", value), std::format("{:+024d}", value));
	}
	constexpr oa::U64 unsignedValues[]{
		0, 1, 7, 8, 15, 16, 255, 256, oa::Limits<oa::U64>::max(),
	};
	for (const oa::U64 value : unsignedValues) {
		expectSameText(oa::format("{:#x}", value), std::format("{:#x}", value));
		expectSameText(oa::format("{:#X}", value), std::format("{:#X}", value));
		expectSameText(oa::format("{:#b}", value), std::format("{:#b}", value));
		expectSameText(oa::format("{:#o}", value), std::format("{:#o}", value));
	}
}

TEST(format, ExplicitFloatingSpecsMatchStdFormat) {
	constexpr double values[]{
		-123456.75, -1.25, -0.0, 0.0, 0.125, 1.25, 123456.75,
		std::numeric_limits<double>::infinity(),
		-std::numeric_limits<double>::infinity(),
	};
	for (const double value : values) {
		expectSameText(oa::format("{:.2f}", value), std::format("{:.2f}", value));
		expectSameText(oa::format("{:+018.6e}", value), std::format("{:+018.6e}", value));
		expectSameText(oa::format("{:#.12g}", value), std::format("{:#.12g}", value));
	}
	expectSameText(oa::format("{:.2f}", std::numeric_limits<double>::quiet_NaN()),
		std::format("{:.2f}", std::numeric_limits<double>::quiet_NaN()));
}

TEST(format, IntegerExtremaAndPointer) {
	EXPECT_STREQ(oa::format("{}", oa::Limits<oa::I64>::min()).cStr(),
		"-9223372036854775808");
	int value = 1;
	EXPECT_EQ(oa::format("{}", &value).find("0x"), 0U);
	EXPECT_STREQ(oa::format("{}", nullptr).cStr(), "0x0");
}

TEST(format, MalformedInputFailsClosed) {
	EXPECT_DEATH(static_cast<void>(oa::format("{")), "unmatched");
	EXPECT_DEATH(static_cast<void>(oa::format("{}")), "too few");
	EXPECT_DEATH(static_cast<void>(oa::format("plain", 1)), "more arguments");
	EXPECT_DEATH(static_cast<void>(oa::format("{:1048577}", 1)), "safety limit");
	EXPECT_DEATH(static_cast<void>(oa::format("{}", static_cast<const char*>(nullptr))),
		"null C string");
	EXPECT_DEATH(static_cast<void>(oa::format("{:q}", 1)), "integer format type");
	EXPECT_DEATH(static_cast<void>(oa::format("{:d}", "text")), "string format type");
	EXPECT_DEATH(static_cast<void>(oa::format("{:.*f}", 1.0)), "precision requires digits");
}

TEST(print, InvalidStreamReturnsStatus) {
	const oa::Status status = oa::print(static_cast<oa::PrintStream>(255), oa::StringView("x"));
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
	const oa::Status flushStatus = oa::flush(static_cast<oa::PrintStream>(255));
	EXPECT_EQ(flushStatus.getCode(), oa::StatusCode::InvalidArgument);
}

TEST(print, PythonStyleNewlineAndExplicitWrite) {
	bool passed = false;
	const std::string output = captureStdout(+[] {
		const oa::Status first = oa::print("value={}", 42);
		const oa::Status second = oa::write("tail");
		const oa::Status third = oa::print(oa::StringView("!"));
		if (first.isOk() and second.isOk() and third.isOk()) {
			// The callback cannot capture; preserve success in the output oracle.
			(void)oa::write(oa::StringView("ok"));
		}
	});
	passed = output == "value=42\ntail!\nok";
	EXPECT_TRUE(passed) << output;
}

TEST(print, ConcurrentRecordsDoNotInterleave) {
	const std::string output = captureStdout(+[] {
		std::vector<std::thread> threads;
		for (oa::U32 thread = 0; thread < 4U; ++thread) {
			threads.emplace_back([thread] {
				for (oa::U32 record = 0; record < 50U; ++record) {
					(void)oa::print("[thread={} record={}]", thread, record);
				}
			});
		}
		for (auto& thread : threads) thread.join();
	});
	std::size_t lineBegin = 0;
	std::size_t lines = 0;
	while (lineBegin < output.size()) {
		const std::size_t lineEnd = output.find('\n', lineBegin);
		ASSERT_NE(lineEnd, std::string::npos);
		const std::string line = output.substr(lineBegin, lineEnd - lineBegin);
		EXPECT_EQ(line.front(), '[');
		EXPECT_EQ(line.back(), ']');
		EXPECT_EQ(line.find('[', 1U), std::string::npos);
		EXPECT_EQ(line.find(']'), line.size() - 1U);
		++lines;
		lineBegin = lineEnd + 1U;
	}
	EXPECT_EQ(lines, 200U);
}

TEST(print, CarriageReturnAndExplicitFlush) {
	const std::string output = captureStdout(+[] {
		EXPECT_TRUE(oa::write("\rprogress={:.1f}%", 37.5).isOk());
		EXPECT_TRUE(oa::flush().isOk());
	});
	EXPECT_EQ(output, "\rprogress=37.5%");
}
