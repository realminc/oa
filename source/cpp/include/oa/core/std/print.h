#pragma once

#include <oa/core/status.h>
#include <oa/core/std/format.h>

namespace oa {

enum class PrintStream : oa::U8 {
	Out,
	Error,
};

namespace detail {
[[nodiscard]] oa::Status writePrint(
	oa::PrintStream inStream,
	oa::StringView inText,
	bool inNewline
) noexcept;
[[nodiscard]] oa::Status flushPrint(oa::PrintStream inStream) noexcept;
} // namespace detail

// Python-like record output. print adds exactly one newline; write does not.
// Each complete call is serialized against other OA print/write calls to the
// same stream. Status reports host I/O failures without exceptions.
template<oa::Usize N, typename... Args>
inline oa::Status print(
	oa::PrintStream inStream,
	const char (&inFormat)[N],
	Args&&... inArgs
) {
	const oa::String text = oa::format(inFormat, oa::forward<Args>(inArgs)...);
	return oa::detail::writePrint(inStream, text.view(), true);
}

template<oa::Usize N, typename... Args>
inline oa::Status print(const char (&inFormat)[N], Args&&... inArgs) {
	return oa::print(oa::PrintStream::Out, inFormat, oa::forward<Args>(inArgs)...);
}

template<oa::Usize N, typename... Args>
inline oa::Status write(
	oa::PrintStream inStream,
	const char (&inFormat)[N],
	Args&&... inArgs
) {
	const oa::String text = oa::format(inFormat, oa::forward<Args>(inArgs)...);
	return oa::detail::writePrint(inStream, text.view(), false);
}

template<oa::Usize N, typename... Args>
inline oa::Status write(const char (&inFormat)[N], Args&&... inArgs) {
	return oa::write(oa::PrintStream::Out, inFormat, oa::forward<Args>(inArgs)...);
}

inline oa::Status print(oa::PrintStream inStream, oa::StringView inText) noexcept {
	return oa::detail::writePrint(inStream, inText, true);
}

inline oa::Status print(oa::StringView inText) noexcept {
	return oa::detail::writePrint(oa::PrintStream::Out, inText, true);
}

inline oa::Status write(oa::PrintStream inStream, oa::StringView inText) noexcept {
	return oa::detail::writePrint(inStream, inText, false);
}

inline oa::Status write(oa::StringView inText) noexcept {
	return oa::detail::writePrint(oa::PrintStream::Out, inText, false);
}

// Explicitly publish buffered host output. Use after write("\r...") for an
// immediately visible same-line progress update; print does not auto-flush.
[[nodiscard]] inline oa::Status flush(
	oa::PrintStream inStream = oa::PrintStream::Out
) noexcept {
	return oa::detail::flushPrint(inStream);
}

} // namespace oa
