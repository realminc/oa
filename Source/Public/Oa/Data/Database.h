// OA — Optional embedded OHLCV store (DuckDB only in Source/Private/Oa/Data/Database.cpp when OA_WITH_DATABASE).

#pragma once

#include <Oa/Data/DatabaseTypes.h>
#include <Oa/Data/OhlcvStore.h>
#include <Oa/Data/SqlStore.h>

#include <shared_mutex>

class OaDatabase final : public OaIOhlcvStore, public OaISqlStore {
public:
	OaDatabase();
	~OaDatabase() override;

	OaDatabase(const OaDatabase&) = delete;
	OaDatabase& operator=(const OaDatabase&) = delete;
	OaDatabase(OaDatabase&& InOther) noexcept;
	OaDatabase& operator=(OaDatabase&& InOther) noexcept;

	[[nodiscard]] OaStatus Open(const OaDatabaseConfig& InConfig) override;
	[[nodiscard]] OaStatus Open(const OaString& InPath, bool InReadOnly = false) override;
	void Close() override;

	[[nodiscard]] bool IsOpen() const noexcept override { return Open_; }
	[[nodiscard]] bool IsReadOnly() const noexcept override { return ReadOnly_; }
	[[nodiscard]] const OaString& GetPath() const noexcept override { return Path_; }

	[[nodiscard]] OaResult<OaDataOhlcv> LoadData(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		OaOpt<OaTimestamp> InStart = std::nullopt,
		OaOpt<OaTimestamp> InEnd = std::nullopt
	) override;

	[[nodiscard]] OaResult<OaVec<OaRecordOhlcv>> LoadRecords(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		OaOpt<OaTimestamp> InStart = std::nullopt,
		OaOpt<OaTimestamp> InEnd = std::nullopt
	) override;

	[[nodiscard]] OaStatus SaveData(
		const OaDataOhlcv& InData,
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		bool InOverwrite = false
	) override;

	[[nodiscard]] OaStatus SaveRecords(
		const OaVec<OaRecordOhlcv>& InRecords,
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval,
		bool InOverwrite = false
	) override;

	[[nodiscard]] OaResult<OaVec<OaContractTicker>> GetTickerCombinations() override;
	[[nodiscard]] OaResult<OaVec<OaDataSummary>> GetAvailableData() override;
	[[nodiscard]] OaResult<OaOpt<OaF64>> GetEarliestTimestamp(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) override;
	[[nodiscard]] OaResult<OaOpt<OaF64>> GetLatestTimestamp(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) override;
	[[nodiscard]] OaResult<OaI64> GetRecordCount(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) override;
	[[nodiscard]] OaResult<OaVec<OaF64>> GetAllTimestamps(
		OaStringView InTicker,
		OaStringView InContractType,
		OaStringView InInterval
	) override;

	[[nodiscard]] OaStatus DeleteData(OaStringView InTicker, OaStringView InContractType, OaStringView InInterval) override;
	[[nodiscard]] OaStatus CompactStorage() override;
	[[nodiscard]] OaStatus ExecuteSQL(OaStringView InQuery) override;

private:
	void* Db_ = nullptr;
	void* Conn_ = nullptr;
	OaString Path_;
	bool Open_ = false;
	bool ReadOnly_ = false;
	mutable std::shared_mutex Mutex_;

	[[nodiscard]] OaStatus InitSchema();
};

[[nodiscard]] OaResult<OaUniquePtr<OaDatabase>> OaOpenDatabase(const OaString& InPath, bool InReadOnly = false);
[[nodiscard]] OaResult<OaUniquePtr<OaDatabase>> OaOpenDatabaseReadOnly(const OaString& InPath);
