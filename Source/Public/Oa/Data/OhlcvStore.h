// OA — Abstract OHLCV time-series persistence (ticker / contract / interval keys).

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Data/DatabaseTypes.h>
#include <Oa/Data/Ohlcv.h>

class OaIOhlcvStore {
public:
	virtual ~OaIOhlcvStore() = default;

	[[nodiscard]] virtual OaStatus Open(const OaDatabaseConfig& InConfig) = 0;
	[[nodiscard]] virtual OaStatus Open(const OaString& InPath, bool InReadOnly = false) = 0;
	virtual void Close() = 0;

	[[nodiscard]] virtual bool IsOpen() const noexcept = 0;
	[[nodiscard]] virtual bool IsReadOnly() const noexcept = 0;
	[[nodiscard]] virtual const OaString& GetPath() const noexcept = 0;

	[[nodiscard]] virtual OaResult<OaDataOhlcv> LoadData(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		OaOpt<OaTimestamp> InStart = std::nullopt,
		OaOpt<OaTimestamp> InEnd = std::nullopt
	) = 0;

	[[nodiscard]] virtual OaResult<OaVec<OaRecordOhlcv>> LoadRecords(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		OaOpt<OaTimestamp> InStart = std::nullopt,
		OaOpt<OaTimestamp> InEnd = std::nullopt
	) = 0;

	[[nodiscard]] virtual OaStatus SaveData(
		const OaDataOhlcv& InData,
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		bool InOverwrite = false
	) = 0;

	[[nodiscard]] virtual OaStatus SaveRecords(
		const OaVec<OaRecordOhlcv>& InRecords,
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		bool InOverwrite = false
	) = 0;

	[[nodiscard]] virtual OaResult<OaVec<OaContractTicker>> GetTickerCombinations() = 0;
	[[nodiscard]] virtual OaResult<OaVec<OaDataSummary>> GetAvailableData() = 0;
	[[nodiscard]] virtual OaResult<OaOpt<OaF64>> GetEarliestTimestamp(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) = 0;
	[[nodiscard]] virtual OaResult<OaOpt<OaF64>> GetLatestTimestamp(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) = 0;
	[[nodiscard]] virtual OaResult<OaI64> GetRecordCount(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) = 0;
	[[nodiscard]] virtual OaResult<OaVec<OaF64>> GetAllTimestamps(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) = 0;

	[[nodiscard]] virtual OaStatus DeleteData(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) = 0;
};
