#include <Oa/Data/Dataframe.h>

#include <Oa/Core/FileIo.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

static void OaDfFillF64(OaF64* Data, OaF64 Value, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Data[i] = Value;
	}
}

static OaF64 OaDfSumF64(const OaF64* Data, OaUsize Size) {
	OaF64 s = 0;
	for (OaUsize i = 0; i < Size; ++i) {
		s += Data[i];
	}
	return s;
}

static OaF64 OaDfMinF64(const OaF64* Data, OaUsize Size) {
	if (Size == 0) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	OaF64 m = Data[0];
	for (OaUsize i = 1; i < Size; ++i) {
		m = std::min(m, Data[i]);
	}
	return m;
}

static OaF64 OaDfMaxF64(const OaF64* Data, OaUsize Size) {
	if (Size == 0) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	OaF64 m = Data[0];
	for (OaUsize i = 1; i < Size; ++i) {
		m = std::max(m, Data[i]);
	}
	return m;
}

static void OaDfAddVV(const OaF64* A, const OaF64* B, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] + B[i];
	}
}

static void OaDfSubVV(const OaF64* A, const OaF64* B, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] - B[i];
	}
}

static void OaDfMulVV(const OaF64* A, const OaF64* B, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] * B[i];
	}
}

static void OaDfDivVV(const OaF64* A, const OaF64* B, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] / B[i];
	}
}

static void OaDfSplitCsvRow(OaStringView InLine, char InDelim, OaVec<OaString>& OutRow) {
	OutRow.Clear();
	OaStringView rest = InLine;
	while (!rest.Empty()) {
		const OaUsize delimPos = rest.find(InDelim);
		OaStringView cell;
		if (delimPos == OaStringView::Npos) {
			cell = rest;
			rest = OaStringView();
		} else {
			cell = rest.SubStr(0, delimPos);
			rest.RemovePrefix(delimPos + 1);
		}
		OutRow.PushBack(OaString(cell));
	}
}

static OaF64 OaDfParseF64Cell(const OaString& InCell) {
	if (InCell.Empty()) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	char* endPtr = nullptr;
	errno = 0;
	const OaF64 val = std::strtod(InCell.c_str(), &endPtr);
	if (endPtr == InCell.c_str() || errno == ERANGE) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	while (*endPtr != '\0' && std::isspace(static_cast<unsigned char>(*endPtr)) != 0) {
		++endPtr;
	}
	if (*endPtr != '\0') {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	return val;
}

static void OaDfAddVS(const OaF64* A, OaF64 Scalar, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] + Scalar;
	}
}

static void OaDfMulVS(const OaF64* A, OaF64 Scalar, OaF64* Out, OaUsize Size) {
	for (OaUsize i = 0; i < Size; ++i) {
		Out[i] = A[i] * Scalar;
	}
}

OaSeries::OaSeries(OaUsize InSize) : Data_(InSize) {}

OaSeries::OaSeries(const OaF64* InData, OaUsize InSize) : Data_(InSize) {
	std::copy(InData, InData + InSize, Data_.begin());
}

OaSeries::OaSeries(std::initializer_list<OaF64> InValues) : Data_(InValues) {}

void OaSeries::Resize(OaUsize InSize) { Data_.resize(InSize); }

void OaSeries::Reserve(OaUsize InCapacity) { Data_.reserve(InCapacity); }

void OaSeries::PushBack(OaF64 InValue) { Data_.push_back(InValue); }

void OaSeries::Clear() { Data_.clear(); }

void OaSeries::Fill(OaF64 InValue) { OaDfFillF64(Data(), InValue, Size()); }

void OaSeries::CopyFrom(const OaF64* InSrc, OaUsize InSize) {
	Data_.resize(InSize);
	std::copy(InSrc, InSrc + InSize, Data_.begin());
}

OaSeries OaSeries::Add(const OaSeries& Other) const {
	const OaUsize n = std::min(Size(), Other.Size());
	OaSeries result(n);
	OaDfAddVV(Data(), Other.Data(), result.Data(), n);
	return result;
}

OaSeries OaSeries::Sub(const OaSeries& Other) const {
	const OaUsize n = std::min(Size(), Other.Size());
	OaSeries result(n);
	OaDfSubVV(Data(), Other.Data(), result.Data(), n);
	return result;
}

OaSeries OaSeries::Mul(const OaSeries& Other) const {
	const OaUsize n = std::min(Size(), Other.Size());
	OaSeries result(n);
	OaDfMulVV(Data(), Other.Data(), result.Data(), n);
	return result;
}

OaSeries OaSeries::Div(const OaSeries& Other) const {
	const OaUsize n = std::min(Size(), Other.Size());
	OaSeries result(n);
	OaDfDivVV(Data(), Other.Data(), result.Data(), n);
	return result;
}

OaSeries OaSeries::Add(OaF64 InScalar) const {
	OaSeries result(Size());
	OaDfAddVS(Data(), InScalar, result.Data(), Size());
	return result;
}

OaSeries OaSeries::Sub(OaF64 InScalar) const { return Add(-InScalar); }

OaSeries OaSeries::Mul(OaF64 InScalar) const {
	OaSeries result(Size());
	OaDfMulVS(Data(), InScalar, result.Data(), Size());
	return result;
}

OaSeries OaSeries::Div(OaF64 InScalar) const { return Mul(1.0 / InScalar); }

OaSeries& OaSeries::AddInplace(const OaSeries& Other) {
	const OaUsize n = std::min(Size(), Other.Size());
	OaDfAddVV(Data(), Other.Data(), Data(), n);
	return *this;
}

OaSeries& OaSeries::SubInplace(const OaSeries& Other) {
	const OaUsize n = std::min(Size(), Other.Size());
	OaDfSubVV(Data(), Other.Data(), Data(), n);
	return *this;
}

OaSeries& OaSeries::MulInplace(const OaSeries& Other) {
	const OaUsize n = std::min(Size(), Other.Size());
	OaDfMulVV(Data(), Other.Data(), Data(), n);
	return *this;
}

OaSeries& OaSeries::DivInplace(const OaSeries& Other) {
	const OaUsize n = std::min(Size(), Other.Size());
	OaDfDivVV(Data(), Other.Data(), Data(), n);
	return *this;
}

OaSeries& OaSeries::AddInplace(OaF64 InScalar) {
	OaDfAddVS(Data(), InScalar, Data(), Size());
	return *this;
}

OaSeries& OaSeries::SubInplace(OaF64 InScalar) { return AddInplace(-InScalar); }

OaSeries& OaSeries::MulInplace(OaF64 InScalar) {
	OaDfMulVS(Data(), InScalar, Data(), Size());
	return *this;
}

OaSeries& OaSeries::DivInplace(OaF64 InScalar) { return MulInplace(1.0 / InScalar); }

OaF64 OaSeries::Sum() const { return OaDfSumF64(Data(), Size()); }

OaF64 OaSeries::Mean() const {
	if (IsEmpty()) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	return Sum() / static_cast<OaF64>(Size());
}

OaF64 OaSeries::Var() const {
	if (Size() < 2) {
		return std::numeric_limits<OaF64>::quiet_NaN();
	}
	const OaF64 mean = Mean();
	OaF64 sumSq = 0;
	for (OaUsize i = 0; i < Size(); ++i) {
		const OaF64 d = Data_[i] - mean;
		sumSq += d * d;
	}
	return sumSq / static_cast<OaF64>(Size() - 1);
}

OaF64 OaSeries::Std() const { return std::sqrt(Var()); }

OaF64 OaSeries::Min() const { return OaDfMinF64(Data(), Size()); }

OaF64 OaSeries::Max() const { return OaDfMaxF64(Data(), Size()); }

OaUsize OaSeries::ArgMin() const {
	if (IsEmpty()) {
		return 0;
	}
	return static_cast<OaUsize>(
		std::distance(Data_.begin(), std::min_element(Data_.begin(), Data_.end())));
}

OaUsize OaSeries::ArgMax() const {
	if (IsEmpty()) {
		return 0;
	}
	return static_cast<OaUsize>(
		std::distance(Data_.begin(), std::max_element(Data_.begin(), Data_.end())));
}

OaSeries OaSeries::RollingSum(OaUsize InWindow) const {
	if (InWindow == 0 || InWindow > Size()) {
		return OaSeries(Size());
	}
	OaSeries result(Size());
	for (OaUsize i = 0; i < InWindow - 1; ++i) {
		result[i] = std::numeric_limits<OaF64>::quiet_NaN();
	}
	OaF64 windowSum = 0;
	for (OaUsize i = 0; i < InWindow; ++i) {
		windowSum += Data_[i];
	}
	result[InWindow - 1] = windowSum;
	for (OaUsize i = InWindow; i < Size(); ++i) {
		windowSum += Data_[i] - Data_[i - InWindow];
		result[i] = windowSum;
	}
	return result;
}

OaSeries OaSeries::RollingMean(OaUsize InWindow) const {
	OaSeries sum = RollingSum(InWindow);
	return sum.Div(static_cast<OaF64>(InWindow));
}

OaSeries OaSeries::RollingStd(OaUsize InWindow) const {
	if (InWindow < 2 || InWindow > Size()) {
		OaSeries result(Size());
		result.Fill(std::numeric_limits<OaF64>::quiet_NaN());
		return result;
	}
	OaSeries result(Size());
	for (OaUsize i = 0; i < InWindow - 1; ++i) {
		result[i] = std::numeric_limits<OaF64>::quiet_NaN();
	}
	for (OaUsize i = InWindow - 1; i < Size(); ++i) {
		OaF64 mean = 0;
		OaF64 m2 = 0;
		for (OaUsize j = 0; j < InWindow; ++j) {
			const OaF64 x = Data_[i - InWindow + 1 + j];
			const OaF64 delta = x - mean;
			mean += delta / static_cast<OaF64>(j + 1);
			const OaF64 delta2 = x - mean;
			m2 += delta * delta2;
		}
		result[i] = std::sqrt(m2 / static_cast<OaF64>(InWindow - 1));
	}
	return result;
}

OaSeries OaSeries::RollingMin(OaUsize InWindow) const {
	if (InWindow == 0 || InWindow > Size()) {
		return OaSeries(Size());
	}
	OaSeries result(Size());
	for (OaUsize i = 0; i < InWindow - 1; ++i) {
		result[i] = std::numeric_limits<OaF64>::quiet_NaN();
	}
	for (OaUsize i = InWindow - 1; i < Size(); ++i) {
		OaF64 minVal = Data_[i - InWindow + 1];
		for (OaUsize j = i - InWindow + 2; j <= i; ++j) {
			minVal = std::min(minVal, Data_[j]);
		}
		result[i] = minVal;
	}
	return result;
}

OaSeries OaSeries::RollingMax(OaUsize InWindow) const {
	if (InWindow == 0 || InWindow > Size()) {
		return OaSeries(Size());
	}
	OaSeries result(Size());
	for (OaUsize i = 0; i < InWindow - 1; ++i) {
		result[i] = std::numeric_limits<OaF64>::quiet_NaN();
	}
	for (OaUsize i = InWindow - 1; i < Size(); ++i) {
		OaF64 maxVal = Data_[i - InWindow + 1];
		for (OaUsize j = i - InWindow + 2; j <= i; ++j) {
			maxVal = std::max(maxVal, Data_[j]);
		}
		result[i] = maxVal;
	}
	return result;
}

OaSeries OaSeries::Shift(OaI64 InPeriods, OaF64 InFillValue) const {
	OaSeries result(Size());
	if (InPeriods >= 0) {
		const OaUsize shift = static_cast<OaUsize>(InPeriods);
		for (OaUsize i = 0; i < shift && i < Size(); ++i) {
			result[i] = InFillValue;
		}
		for (OaUsize i = shift; i < Size(); ++i) {
			result[i] = Data_[i - shift];
		}
	} else {
		const OaUsize shift = static_cast<OaUsize>(-InPeriods);
		for (OaUsize i = 0; i + shift < Size(); ++i) {
			result[i] = Data_[i + shift];
		}
		for (OaUsize i = Size() > shift ? Size() - shift : 0; i < Size(); ++i) {
			result[i] = InFillValue;
		}
	}
	return result;
}

OaSeries OaSeries::Diff(OaI64 InPeriods) const {
	OaSeries shifted = Shift(InPeriods, std::numeric_limits<OaF64>::quiet_NaN());
	return Sub(shifted);
}

OaSeries OaSeries::PctChange(OaI64 InPeriods) const {
	OaSeries shifted = Shift(InPeriods, std::numeric_limits<OaF64>::quiet_NaN());
	OaSeries diff = Sub(shifted);
	return diff.Div(shifted);
}

OaSeries OaSeries::Slice(OaUsize InStart, OaUsize InEnd) const {
	InStart = std::min(InStart, Size());
	InEnd = std::min(InEnd, Size());
	if (InStart >= InEnd) {
		return OaSeries();
	}
	return OaSeries(Data() + InStart, InEnd - InStart);
}

OaSeries OaSeries::Head(OaUsize InN) const { return Slice(0, InN); }

OaSeries OaSeries::Tail(OaUsize InN) const {
	if (InN >= Size()) {
		return *this;
	}
	return Slice(Size() - InN, Size());
}

OaResult<OaTimeSeries> OaTimeSeries::Create(OaSeries InTimes, OaSeries InValues) {
	if (!InTimes.IsEmpty() && InTimes.Size() != InValues.Size()) {
		return OaStatus::InvalidArgument("OaTimeSeries: times and values length mismatch");
	}
	OaTimeSeries ts;
	ts.Times_ = std::move(InTimes);
	ts.Values_ = std::move(InValues);
	return ts;
}

OaTimeSeries OaTimeSeries::FromOhlcvClose(const OaDataOhlcv& InData) {
	OaTimeSeries ts;
	ts.Times_ = OaSeries(InData.Time.Data(), InData.Time.Size());
	ts.Values_ = OaSeries(InData.Close.Data(), InData.Close.Size());
	return ts;
}

OaUsize OaTimeSeries::Size() const {
	if (!Times_.IsEmpty()) {
		return Times_.Size();
	}
	return Values_.Size();
}

OaTimeSeries OaTimeSeries::Slice(OaUsize InStart, OaUsize InEnd) const {
	OaTimeSeries out;
	out.Values_ = Values_.Slice(InStart, InEnd);
	if (!Times_.IsEmpty()) {
		out.Times_ = Times_.Slice(InStart, InEnd);
	}
	return out;
}

OaDataFrame::OaDataFrame(OaUsize InCapacity) { (void)InCapacity; }

OaDataFrame OaDataFrame::FromOhlcvData(const OaDataOhlcv& InData) {
	OaDataFrame df;
	df.AddColumn(OaOhlcvColumnTime, OaSeries(InData.Time.Data(), InData.Time.Size()));
	df.AddColumn(OaOhlcvColumnOpen, OaSeries(InData.Open.Data(), InData.Open.Size()));
	df.AddColumn(OaOhlcvColumnHigh, OaSeries(InData.High.Data(), InData.High.Size()));
	df.AddColumn(OaOhlcvColumnLow, OaSeries(InData.Low.Data(), InData.Low.Size()));
	df.AddColumn(OaOhlcvColumnClose, OaSeries(InData.Close.Data(), InData.Close.Size()));
	df.AddColumn(OaOhlcvColumnVolume, OaSeries(InData.Volume.Data(), InData.Volume.Size()));
	return df;
}

OaDataFrame OaDataFrame::FromOhlcvRecords(const OaVec<OaRecordOhlcv>& InRecords) {
	OaDataOhlcv data(InRecords.Size());
	for (const auto& rec : InRecords) {
		data.PushBack(rec);
	}
	return FromOhlcvData(data);
}

OaResult<OaDataFrame> OaDataFrame::FromCSV(const OaPath& InPath, char InDelimiter) {
	auto lines = OaFileIo::ReadLines(InPath);
	if (!lines.IsOk()) {
		return lines.GetStatus();
	}
	OaDataFrame df;
	const OaVec<OaString>& L = lines.GetValue();
	if (L.Empty()) {
		return df;
	}
	OaVec<OaString> headers;
	OaVec<OaString> row;
	bool isHeader = true;
	for (const OaString& line : L) {
		OaDfSplitCsvRow(line.View(), InDelimiter, row);
		if (isHeader) {
			headers = row;
			for (const auto& h : headers) {
				df.AddColumn(h, OaSeries());
			}
			isHeader = false;
		} else {
			for (OaUsize i = 0; i < row.Size() && i < headers.Size(); ++i) {
				const OaF64 val = OaDfParseF64Cell(row[i]);
				df.GetColumnMut(headers[i]).PushBack(val);
			}
		}
	}
	return df;
}

bool OaDataFrame::HasColumn(OaStringView InName) const {
	return Columns_.Contains(OaString(InName));
}

const OaSeries& OaDataFrame::GetColumn(OaStringView InName) const {
	const auto it = Columns_.Find(OaString(InName));
	if (it == Columns_.End()) {
		static OaSeries empty;
		return empty;
	}
	return it->second;
}

OaSeries& OaDataFrame::GetColumnMut(OaStringView InName) {
	OaString key(InName);
	auto it = Columns_.Find(key);
	if (it != Columns_.End()) {
		return it->second;
	}
	auto placed = Columns_.Emplace(std::move(key), OaSeries());
	return placed.first->second;
}

void OaDataFrame::AddColumn(OaStringView InName, OaSeries InSeries) {
	OaString name(InName);
	if (!Columns_.Contains(name)) {
		ColumnOrder_.PushBack(name);
		Columns_.Emplace(std::move(name), std::move(InSeries));
		return;
	}
	Columns_.Find(name)->second = std::move(InSeries);
}

void OaDataFrame::SetColumn(OaStringView InName, OaSeries InSeries) {
	AddColumn(InName, std::move(InSeries));
}

void OaDataFrame::RemoveColumn(OaStringView InName) {
	OaString name(InName);
	Columns_.Erase(name);
	ColumnOrder_.Erase(
		std::remove(ColumnOrder_.Begin(), ColumnOrder_.End(), name),
		ColumnOrder_.End());
}

OaVec<OaString> OaDataFrame::GetColumnNames() const { return ColumnOrder_; }

OaUsize OaDataFrame::GetNumRows() const {
	if (Columns_.Empty()) {
		return 0;
	}
	return Columns_.Begin()->second.Size();
}

OaDataFrame OaDataFrame::Slice(OaUsize InStart, OaUsize InEnd) const {
	OaDataFrame result;
	for (const auto& name : ColumnOrder_) {
		result.AddColumn(name, GetColumn(name).Slice(InStart, InEnd));
	}
	return result;
}

OaDataFrame OaDataFrame::Head(OaUsize InN) const { return Slice(0, InN); }

OaDataFrame OaDataFrame::Tail(OaUsize InN) const {
	const OaUsize rows = GetNumRows();
	if (InN >= rows) {
		return *this;
	}
	return Slice(rows - InN, rows);
}

OaDataFrame OaDataFrame::Select(const OaVec<OaString>& InColumns) const {
	OaDataFrame result;
	for (const auto& name : InColumns) {
		if (HasColumn(name)) {
			result.AddColumn(name, GetColumn(name));
		}
	}
	return result;
}

static void OaDfCopyTimeColumn(const OaDataFrame& InDf, OaVec<OaF64>& OutTime) {
	if (InDf.HasColumn(OaOhlcvColumnTime)) {
		const OaSeries& s = InDf.GetColumn(OaOhlcvColumnTime);
		OutTime.Assign(s.Data(), s.Data() + s.Size());
	} else if (InDf.HasColumn(OaOhlcvColumnTimeLegacy)) {
		const OaSeries& s = InDf.GetColumn(OaOhlcvColumnTimeLegacy);
		OutTime.Assign(s.Data(), s.Data() + s.Size());
	}
}

OaDataOhlcv OaDataFrame::ToOhlcvData() const {
	OaDataOhlcv data;
	OaDfCopyTimeColumn(*this, data.Time);
	if (HasColumn(OaOhlcvColumnOpen)) {
		const OaSeries& s = GetColumn(OaOhlcvColumnOpen);
		data.Open.Assign(s.Data(), s.Data() + s.Size());
	}
	if (HasColumn(OaOhlcvColumnHigh)) {
		const OaSeries& s = GetColumn(OaOhlcvColumnHigh);
		data.High.Assign(s.Data(), s.Data() + s.Size());
	}
	if (HasColumn(OaOhlcvColumnLow)) {
		const OaSeries& s = GetColumn(OaOhlcvColumnLow);
		data.Low.Assign(s.Data(), s.Data() + s.Size());
	}
	if (HasColumn(OaOhlcvColumnClose)) {
		const OaSeries& s = GetColumn(OaOhlcvColumnClose);
		data.Close.Assign(s.Data(), s.Data() + s.Size());
	}
	if (HasColumn(OaOhlcvColumnVolume)) {
		const OaSeries& s = GetColumn(OaOhlcvColumnVolume);
		data.Volume.Assign(s.Data(), s.Data() + s.Size());
	}
	return data;
}

OaStatus OaDataFrame::ToCsv(const OaPath& InPath, char InDelimiter) const {
	OaString file;
	for (OaUsize i = 0; i < ColumnOrder_.Size(); ++i) {
		if (i > 0) {
			file.PushBack(InDelimiter);
		}
		file += ColumnOrder_[i];
	}
	file.PushBack('\n');
	const OaUsize rows = GetNumRows();
	char numBuf[64];
	for (OaUsize rowIdx = 0; rowIdx < rows; ++rowIdx) {
		for (OaUsize col = 0; col < ColumnOrder_.Size(); ++col) {
			if (col > 0) {
				file.PushBack(InDelimiter);
			}
			const OaF64 val = GetColumn(ColumnOrder_[col])[rowIdx];
			std::snprintf(numBuf, sizeof(numBuf), "%.17g", val);
			file += numBuf;
		}
		file.PushBack('\n');
	}
	return OaFileIo::WriteText(InPath, file.View());
}

OaStatus OaDataFrame::ToParquet(const OaPath& InPath) const {
	(void)InPath;
	return OaStatus::Unimplemented(
		"OaDataFrame::ToParquet requires Apache Arrow/Parquet (not built in liboa by default)");
}

OaResult<OaDataFrame> OaDataFrame::FromParquet(const OaPath& InPath) {
	(void)InPath;
	return OaStatus::Unimplemented(
		"OaDataFrame::FromParquet requires Apache Arrow/Parquet (not built in liboa by default)");
}

OaUsize OaDataFrame::GetMemoryUsage() const {
	OaUsize total = 0;
	for (const auto& pair : Columns_) {
		total += pair.second.Size() * sizeof(OaF64);
		total += pair.first.Capacity();
	}
	return total;
}

void OaDataFrame::Reserve(OaUsize InCapacity) {
	for (auto& pair : Columns_) {
		pair.second.Reserve(InCapacity);
	}
}

OaResult<OaAlignedVec<OaF64>> OaDataFrame::ToMatrix(const OaVec<OaString>& InColumns) const {
	if (InColumns.Empty()) {
		return OaStatus::InvalidArgument("No columns specified");
	}
	const OaUsize rows = GetNumRows();
	const OaUsize cols = InColumns.Size();
	OaAlignedVec<OaF64> matrix(rows * cols);
	for (OaUsize c = 0; c < cols; ++c) {
		if (!HasColumn(InColumns[c])) {
			return OaStatus::NotFound(OaString("Column not found: ") + InColumns[c]);
		}
		const OaSeries& col = GetColumn(InColumns[c]);
		std::copy(col.Data(), col.Data() + rows, matrix.data() + (c * rows));
	}
	return matrix;
}

OaResult<OaAlignedVec<OaF32>> OaDataFrame::ToMatrixF32(const OaVec<OaString>& InColumns) const {
	auto result64 = ToMatrix(InColumns);
	if (!result64.IsOk()) {
		return result64.GetStatus();
	}
	const auto& data64 = result64.GetValue();
	OaAlignedVec<OaF32> data32(data64.size());
	for (OaUsize i = 0; i < data64.size(); ++i) {
		data32[i] = static_cast<OaF32>(data64[i]);
	}
	return data32;
}

OaDataOhlcv OaResampleOhlcv(const OaDataOhlcv& InData, OaI32 InFactor) {
	if (InFactor <= 1) {
		return InData;
	}
	const OaI64 len = static_cast<OaI64>(InData.Time.Size());
	OaDataOhlcv out;
	out.Time.Resize(static_cast<OaUsize>(len));
	out.Open.Resize(static_cast<OaUsize>(len));
	out.High.Resize(static_cast<OaUsize>(len));
	out.Low.Resize(static_cast<OaUsize>(len));
	out.Close.Resize(static_cast<OaUsize>(len));
	out.Volume.Resize(static_cast<OaUsize>(len));
	for (OaI64 i = 0; i < len; ++i) {
		const OaI64 barStart = (i / InFactor) * InFactor;
		const OaI64 barEnd = std::min(barStart + InFactor - 1, i);
		out.Time[static_cast<OaUsize>(i)] = InData.Time[static_cast<OaUsize>(barStart)];
		out.Open[static_cast<OaUsize>(i)] = InData.Open[static_cast<OaUsize>(barStart)];
		out.Close[static_cast<OaUsize>(i)] = InData.Close[static_cast<OaUsize>(barEnd)];
		OaF64 high = InData.High[static_cast<OaUsize>(barStart)];
		OaF64 low = InData.Low[static_cast<OaUsize>(barStart)];
		OaF64 vol = 0;
		for (OaI64 j = barStart; j <= barEnd; ++j) {
			high = std::max(high, InData.High[static_cast<OaUsize>(j)]);
			low = std::min(low, InData.Low[static_cast<OaUsize>(j)]);
			vol += InData.Volume[static_cast<OaUsize>(j)];
		}
		out.High[static_cast<OaUsize>(i)] = high;
		out.Low[static_cast<OaUsize>(i)] = low;
		out.Volume[static_cast<OaUsize>(i)] = vol;
	}
	return out;
}
