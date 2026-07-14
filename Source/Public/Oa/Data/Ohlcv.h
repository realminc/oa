// OA — Columnar OHLCV types (engine-agnostic; optional SQL store in database.h).

#pragma once

#include <Oa/Core/Time.h>
#include <Oa/Core/Types.h>

// Canonical OHLCV column names for OaDataFrame / CSV (pandas-style).
// Two-arg ctor: brace form would pick non-constexpr const char* overload.
inline constexpr OaStringView OaOhlcvColumnTime("Time", sizeof("Time") - 1);
inline constexpr OaStringView OaOhlcvColumnOpen("Open", sizeof("Open") - 1);
inline constexpr OaStringView OaOhlcvColumnHigh("High", sizeof("High") - 1);
inline constexpr OaStringView OaOhlcvColumnLow("Low", sizeof("Low") - 1);
inline constexpr OaStringView OaOhlcvColumnClose("Close", sizeof("Close") - 1);
inline constexpr OaStringView OaOhlcvColumnVolume("Volume", sizeof("Volume") - 1);
// Legacy trade export name for the time column (ToOhlcvData accepts either).
inline constexpr OaStringView OaOhlcvColumnTimeLegacy("OaTimestamp", sizeof("OaTimestamp") - 1);

struct OaRecordOhlcv {
	OaF64 Time = 0;
	OaF64 Open = 0;
	OaF64 High = 0;
	OaF64 Low = 0;
	OaF64 Close = 0;
	OaF64 Volume = 0;

	OaRecordOhlcv() = default;
	OaRecordOhlcv(OaF64 InTime, OaF64 InOpen, OaF64 InHigh, OaF64 InLow, OaF64 InClose, OaF64 InVolume)
		: Time(InTime), Open(InOpen), High(InHigh), Low(InLow), Close(InClose), Volume(InVolume) {}

	[[nodiscard]] OaTimestamp GetTimestamp() const { return OaTimestamp::FromDouble(Time); }
	[[nodiscard]] bool IsValid() const { return High >= Low && High >= Open && High >= Close; }
};

struct OaDataOhlcv {
	OaVec<OaF64> Time;
	OaVec<OaF64> Open;
	OaVec<OaF64> High;
	OaVec<OaF64> Low;
	OaVec<OaF64> Close;
	OaVec<OaF64> Volume;

	OaDataOhlcv() = default;
	explicit OaDataOhlcv(OaUsize InCapacity) { Reserve(InCapacity); }

	void Reserve(OaUsize InCapacity) {
		Time.Reserve(InCapacity);
		Open.Reserve(InCapacity);
		High.Reserve(InCapacity);
		Low.Reserve(InCapacity);
		Close.Reserve(InCapacity);
		Volume.Reserve(InCapacity);
	}

	void Clear() {
		Time.Clear();
		Open.Clear();
		High.Clear();
		Low.Clear();
		Close.Clear();
		Volume.Clear();
	}

	void PushBack(const OaRecordOhlcv& InRecord) {
		Time.PushBack(InRecord.Time);
		Open.PushBack(InRecord.Open);
		High.PushBack(InRecord.High);
		Low.PushBack(InRecord.Low);
		Close.PushBack(InRecord.Close);
		Volume.PushBack(InRecord.Volume);
	}

	[[nodiscard]] OaUsize Size() const { return Time.Size(); }
	[[nodiscard]] bool IsEmpty() const { return Time.Empty(); }

	[[nodiscard]] OaRecordOhlcv GetRecord(OaUsize InIdx) const {
		return OaRecordOhlcv(Time[InIdx], Open[InIdx], High[InIdx], Low[InIdx], Close[InIdx], Volume[InIdx]);
	}

	[[nodiscard]] const OaF64* GetTimePtr() const { return Time.Data(); }
	[[nodiscard]] const OaF64* GetOpenPtr() const { return Open.Data(); }
	[[nodiscard]] const OaF64* GetHighPtr() const { return High.Data(); }
	[[nodiscard]] const OaF64* GetLowPtr() const { return Low.Data(); }
	[[nodiscard]] const OaF64* GetClosePtr() const { return Close.Data(); }
	[[nodiscard]] const OaF64* GetVolumePtr() const { return Volume.Data(); }

	[[nodiscard]] OaF64* GetTimePtrMut() { return Time.Data(); }
	[[nodiscard]] OaF64* GetOpenPtrMut() { return Open.Data(); }
	[[nodiscard]] OaF64* GetHighPtrMut() { return High.Data(); }
	[[nodiscard]] OaF64* GetLowPtrMut() { return Low.Data(); }
	[[nodiscard]] OaF64* GetClosePtrMut() { return Close.Data(); }
	[[nodiscard]] OaF64* GetVolumePtrMut() { return Volume.Data(); }
};
