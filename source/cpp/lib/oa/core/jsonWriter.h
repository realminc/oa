#pragma once

// Private, allocation-aware JSON text writer. This keeps diagnostic and graph
// serialization independent from hosted iostreams while always emitting valid
// JSON strings and locale-independent decimal points.

#include <oa/core/std/format.h>
#include <oa/core/std/scalarMath.h>
#include <oa/core/std/string.h>
#include <oa/core/std/utility.h>

namespace oa::internal {

class JsonWriter {
public:
	JsonWriter() = default;

	JsonWriter& operator<<(oa::StringView inValue) {
		text_.append(inValue);
		return *this;
	}
	JsonWriter& operator<<(const oa::String& inValue) { return *this << inValue.view(); }
	JsonWriter& operator<<(const char* inValue) {
		if (inValue != nullptr) text_.append(inValue);
		return *this;
	}
	JsonWriter& operator<<(char inValue) {
		text_.pushBack(inValue);
		return *this;
	}
	JsonWriter& operator<<(unsigned int inValue) { return appendUnsigned_(inValue); }
	JsonWriter& operator<<(unsigned long inValue) { return appendUnsigned_(inValue); }
	JsonWriter& operator<<(unsigned long long inValue) { return appendUnsigned_(inValue); }
	JsonWriter& operator<<(int inValue) { return appendSigned_(inValue); }
	JsonWriter& operator<<(long inValue) { return appendSigned_(inValue); }
	JsonWriter& operator<<(long long inValue) { return appendSigned_(inValue); }
	JsonWriter& operator<<(float inValue) { return writeFloat(inValue, 9); }
	JsonWriter& operator<<(double inValue) { return writeFloat(inValue, 17); }

	JsonWriter& writeString(oa::StringView inValue) {
		static constexpr char Hex[] = "0123456789abcdef";
		text_.pushBack('"');
		for (const char character : inValue) {
			const auto value = static_cast<oa::U8>(character);
			switch (value) {
				case '"': text_.append("\\\""); break;
				case '\\': text_.append("\\\\"); break;
				case '\b': text_.append("\\b"); break;
				case '\f': text_.append("\\f"); break;
				case '\n': text_.append("\\n"); break;
				case '\r': text_.append("\\r"); break;
				case '\t': text_.append("\\t"); break;
				default:
					if (value < 0x20U) {
						text_.append("\\u00");
						text_.pushBack(Hex[(value >> 4U) & 0x0fU]);
						text_.pushBack(Hex[value & 0x0fU]);
					} else {
						text_.pushBack(character);
					}
			}
		}
		text_.pushBack('"');
		return *this;
	}

	JsonWriter& writeHexId(oa::U64 inValue) {
		static constexpr char Hex[] = "0123456789abcdef";
		text_.append("\"0x");
		for (oa::I32 shift = 60; shift >= 0; shift -= 4) {
			text_.pushBack(Hex[(inValue >> static_cast<oa::U32>(shift)) & 0x0fU]);
		}
		text_.pushBack('"');
		return *this;
	}

	JsonWriter& writeFloat(oa::F64 inValue, int inPrecision = 17) {
		if (not oa::isFinite(inValue)) {
			text_.append("null");
			return *this;
		}
		oa::String number;
		if (not oa::formatF64(inValue, number, inPrecision)) {
			text_.append("null");
			return *this;
		}
		text_.append(number.view());
		return *this;
	}

	[[nodiscard]] const oa::String& string() const noexcept { return text_; }
	[[nodiscard]] oa::String take() noexcept { return oa::move(text_); }

private:
	template <typename T>
	JsonWriter& appendUnsigned_(T inValue) {
		text_.append(oa::toString(static_cast<oa::U64>(inValue)).view());
		return *this;
	}

	template <typename T>
	JsonWriter& appendSigned_(T inValue) {
		text_.append(oa::toString(static_cast<oa::I64>(inValue)).view());
		return *this;
	}

	oa::String text_;
};

} // namespace oa::internal
