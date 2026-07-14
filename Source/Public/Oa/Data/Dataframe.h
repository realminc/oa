// OA — Columnar f64 series, time series, and dataframe (pandas-oriented).

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Data/Ohlcv.h>

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <new>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

inline constexpr OaUsize OaDataframeCacheLineSize = 64;
inline constexpr OaUsize OaDataframeSimdAlignment = 64;

template <typename T, OaUsize Alignment = OaDataframeSimdAlignment>
class OaAlignedAllocator {
public:
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;
	using is_always_equal = std::true_type;

	template <typename U>
	struct rebind {
		using other = OaAlignedAllocator<U, Alignment>;
	};

	constexpr OaAlignedAllocator() noexcept = default;
	constexpr OaAlignedAllocator(const OaAlignedAllocator&) noexcept = default;
	constexpr OaAlignedAllocator& operator=(const OaAlignedAllocator&) noexcept = default;

	template <typename U>
	constexpr OaAlignedAllocator(const OaAlignedAllocator<U, Alignment>& InOther) noexcept {
		(void)InOther;
	}

	[[nodiscard]] T* allocate(size_type n) {
		if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
			throw std::bad_array_new_length();
		}
		void* ptr = nullptr;
#ifdef _WIN32
		ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
		if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
			ptr = nullptr;
		}
#endif
		if (ptr == nullptr) {
			throw std::bad_alloc();
		}
		return static_cast<T*>(ptr);
	}

	void deallocate(T* InPtr, size_type /*InSize*/) noexcept {
#ifdef _WIN32
		_aligned_free(InPtr);
#else
		free(InPtr);
#endif
	}

	constexpr bool operator==(const OaAlignedAllocator&) const noexcept { return true; }
};

template <typename T>
using OaAlignedVec = std::vector<T, OaAlignedAllocator<T, OaDataframeSimdAlignment>>;

class OaSeries {
public:
	OaSeries() = default;
	explicit OaSeries(OaUsize InSize);
	OaSeries(const OaF64* InData, OaUsize InSize);
	OaSeries(std::initializer_list<OaF64> InValues);

	OaSeries(OaSeries&&) noexcept = default;
	OaSeries& operator=(OaSeries&&) noexcept = default;
	OaSeries(const OaSeries&) = default;
	OaSeries& operator=(const OaSeries&) = default;

	[[nodiscard]] OaUsize Size() const noexcept { return static_cast<OaUsize>(Data_.size()); }
	[[nodiscard]] bool IsEmpty() const noexcept { return Data_.empty(); }

	[[nodiscard]] OaF64 operator[](OaUsize InIdx) const { return Data_[InIdx]; }
	[[nodiscard]] OaF64& operator[](OaUsize InIdx) { return Data_[InIdx]; }
	[[nodiscard]] OaF64 At(OaUsize InIdx) const { return Data_.at(InIdx); }
	[[nodiscard]] OaF64& At(OaUsize InIdx) { return Data_.at(InIdx); }

	[[nodiscard]] OaF64 Front() const { return Data_.front(); }
	[[nodiscard]] OaF64 Back() const { return Data_.back(); }

	[[nodiscard]] const OaF64* Data() const noexcept { return Data_.data(); }
	[[nodiscard]] OaF64* Data() noexcept { return Data_.data(); }

	void Resize(OaUsize InSize);
	void Reserve(OaUsize InCapacity);
	void PushBack(OaF64 InValue);
	void Clear();

	void Fill(OaF64 InValue);
	void CopyFrom(const OaF64* InSrc, OaUsize InSize);

	[[nodiscard]] OaSeries Add(const OaSeries& Other) const;
	[[nodiscard]] OaSeries Sub(const OaSeries& Other) const;
	[[nodiscard]] OaSeries Mul(const OaSeries& Other) const;
	[[nodiscard]] OaSeries Div(const OaSeries& Other) const;
	[[nodiscard]] OaSeries Add(OaF64 InScalar) const;
	[[nodiscard]] OaSeries Sub(OaF64 InScalar) const;
	[[nodiscard]] OaSeries Mul(OaF64 InScalar) const;
	[[nodiscard]] OaSeries Div(OaF64 InScalar) const;

	OaSeries& AddInplace(const OaSeries& Other);
	OaSeries& SubInplace(const OaSeries& Other);
	OaSeries& MulInplace(const OaSeries& Other);
	OaSeries& DivInplace(const OaSeries& Other);
	OaSeries& AddInplace(OaF64 InScalar);
	OaSeries& SubInplace(OaF64 InScalar);
	OaSeries& MulInplace(OaF64 InScalar);
	OaSeries& DivInplace(OaF64 InScalar);

	[[nodiscard]] OaF64 Sum() const;
	[[nodiscard]] OaF64 Mean() const;
	[[nodiscard]] OaF64 Std() const;
	[[nodiscard]] OaF64 Var() const;
	[[nodiscard]] OaF64 Min() const;
	[[nodiscard]] OaF64 Max() const;
	[[nodiscard]] OaUsize ArgMin() const;
	[[nodiscard]] OaUsize ArgMax() const;

	[[nodiscard]] OaSeries RollingSum(OaUsize InWindow) const;
	[[nodiscard]] OaSeries RollingMean(OaUsize InWindow) const;
	[[nodiscard]] OaSeries RollingStd(OaUsize InWindow) const;
	[[nodiscard]] OaSeries RollingMin(OaUsize InWindow) const;
	[[nodiscard]] OaSeries RollingMax(OaUsize InWindow) const;
	[[nodiscard]] OaSeries Shift(OaI64 InPeriods, OaF64 InFillValue = 0.0) const;
	[[nodiscard]] OaSeries Diff(OaI64 InPeriods = 1) const;
	[[nodiscard]] OaSeries PctChange(OaI64 InPeriods = 1) const;

	[[nodiscard]] OaSeries Slice(OaUsize InStart, OaUsize InEnd) const;
	[[nodiscard]] OaSeries Head(OaUsize InN) const;
	[[nodiscard]] OaSeries Tail(OaUsize InN) const;

	[[nodiscard]] OaSeries operator+(const OaSeries& Other) const { return Add(Other); }
	[[nodiscard]] OaSeries operator-(const OaSeries& Other) const { return Sub(Other); }
	[[nodiscard]] OaSeries operator*(const OaSeries& Other) const { return Mul(Other); }
	[[nodiscard]] OaSeries operator/(const OaSeries& Other) const { return Div(Other); }

	OaSeries& operator+=(const OaSeries& Other) { return AddInplace(Other); }
	OaSeries& operator-=(const OaSeries& Other) { return SubInplace(Other); }
	OaSeries& operator*=(const OaSeries& Other) { return MulInplace(Other); }
	OaSeries& operator/=(const OaSeries& Other) { return DivInplace(Other); }

private:
	OaAlignedVec<OaF64> Data_;
};

// Time-aligned values: optional time index (unix seconds as f64) + one value column.
class OaTimeSeries {
public:
	OaTimeSeries() = default;

	[[nodiscard]] static OaResult<OaTimeSeries> Create(OaSeries InTimes, OaSeries InValues);
	[[nodiscard]] static OaTimeSeries FromOhlcvClose(const OaDataOhlcv& InData);

	[[nodiscard]] OaUsize Size() const;
	[[nodiscard]] bool HasTimeIndex() const { return !Times_.IsEmpty(); }

	[[nodiscard]] const OaSeries& Times() const { return Times_; }
	[[nodiscard]] const OaSeries& Values() const { return Values_; }
	[[nodiscard]] OaSeries& TimesMut() { return Times_; }
	[[nodiscard]] OaSeries& ValuesMut() { return Values_; }

	[[nodiscard]] OaTimeSeries Slice(OaUsize InStart, OaUsize InEnd) const;

private:
	OaSeries Times_;
	OaSeries Values_;
};

class OaDataFrame {
public:
	OaDataFrame() = default;
	explicit OaDataFrame(OaUsize InCapacity);

	[[nodiscard]] static OaDataFrame FromOhlcvData(const OaDataOhlcv& InData);
	[[nodiscard]] static OaDataFrame FromOhlcvRecords(const OaVec<OaRecordOhlcv>& InRecords);
	[[nodiscard]] static OaResult<OaDataFrame> FromCSV(const OaPath& InPath, char InDelimiter = ',');

	[[nodiscard]] bool HasColumn(OaStringView InName) const;
	[[nodiscard]] const OaSeries& GetColumn(OaStringView InName) const;
	[[nodiscard]] OaSeries& GetColumnMut(OaStringView InName);
	[[nodiscard]] const OaSeries& operator[](OaStringView InName) const { return GetColumn(InName); }
	[[nodiscard]] OaSeries& operator[](OaStringView InName) { return GetColumnMut(InName); }

	void AddColumn(OaStringView InName, OaSeries InSeries);
	void SetColumn(OaStringView InName, OaSeries InSeries);
	void RemoveColumn(OaStringView InName);
	[[nodiscard]] OaVec<OaString> GetColumnNames() const;

	[[nodiscard]] OaUsize GetNumRows() const;
	[[nodiscard]] OaUsize GetNumColumns() const { return Columns_.Size(); }
	[[nodiscard]] bool IsEmpty() const { return GetNumRows() == 0; }

	[[nodiscard]] OaDataFrame Slice(OaUsize InStart, OaUsize InEnd) const;
	[[nodiscard]] OaDataFrame Head(OaUsize InN = 5) const;
	[[nodiscard]] OaDataFrame Tail(OaUsize InN = 5) const;
	[[nodiscard]] OaDataFrame Select(const OaVec<OaString>& InColumns) const;

	[[nodiscard]] OaDataOhlcv ToOhlcvData() const;
	[[nodiscard]] OaStatus ToCsv(const OaPath& InPath, char InDelimiter = ',') const;
	[[nodiscard]] OaStatus ToParquet(const OaPath& InPath) const;
	[[nodiscard]] static OaResult<OaDataFrame> FromParquet(const OaPath& InPath);

	[[nodiscard]] OaUsize GetMemoryUsage() const;
	void Reserve(OaUsize InCapacity);

	[[nodiscard]] OaResult<OaAlignedVec<OaF64>> ToMatrix(const OaVec<OaString>& InColumns) const;
	[[nodiscard]] OaResult<OaAlignedVec<OaF32>> ToMatrixF32(const OaVec<OaString>& InColumns) const;

private:
	OaHashMap<OaString, OaSeries> Columns_;
	OaVec<OaString> ColumnOrder_;
};

[[nodiscard]] OaDataOhlcv OaResampleOhlcv(const OaDataOhlcv& InData, OaI32 InFactor);
