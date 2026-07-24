#include <Oa/Data/Database.h>

#include <Oa/Core/Filesystem.h>

#include <fmt/format.h>

#if defined(OA_HAS_EMBEDDED_SQL_STORE)
#include <chrono>
#include <duckdb.h>
#endif

#if defined(OA_HAS_EMBEDDED_SQL_STORE)

static constexpr OaStringView kCreateTableSQL = R"sql(
	CREATE TABLE IF NOT EXISTS ohlcv (
		Timestamp DOUBLE NOT NULL,
		Ticker VARCHAR NOT NULL,
		ContractType VARCHAR NOT NULL,
		Interval VARCHAR NOT NULL,
		Open DOUBLE NOT NULL,
		High DOUBLE NOT NULL,
		Low DOUBLE NOT NULL,
		Close DOUBLE NOT NULL,
		Volume DOUBLE NOT NULL,
		CreatedAt DOUBLE NOT NULL
	);
)sql";

static constexpr OaStringView kCreateIndexSQL = R"sql(
	CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
	ON ohlcv(Ticker, ContractType, Interval, Timestamp);
)sql";

class OaDuckdbResult {
public:
	OaDuckdbResult() = default;
	~OaDuckdbResult() { duckdb_destroy_result(&Result_); }

	OaDuckdbResult(const OaDuckdbResult&) = delete;
	OaDuckdbResult& operator=(const OaDuckdbResult&) = delete;
	OaDuckdbResult(OaDuckdbResult&&) = delete;
	OaDuckdbResult& operator=(OaDuckdbResult&&) = delete;

	duckdb_result* Get() { return &Result_; }

	[[nodiscard]] bool HasError() const { return duckdb_result_error(&Result_) != nullptr; }

	[[nodiscard]] OaString GetError() const {
		const char* err = duckdb_result_error(&Result_);
		return err ? OaString(err) : OaString();
	}

	[[nodiscard]] idx_t RowCount() const { return duckdb_row_count(&Result_); }

	[[nodiscard]] OaF64 GetDouble(idx_t InRow, idx_t InCol) const {
		return duckdb_value_double(&Result_, InCol, InRow);
	}

	[[nodiscard]] OaI64 GetInt64(idx_t InRow, idx_t InCol) const {
		return duckdb_value_int64(&Result_, InCol, InRow);
	}

	[[nodiscard]] OaString GetString(idx_t InRow, idx_t InCol) const {
		char* val = duckdb_value_varchar(&Result_, InCol, InRow);
		OaString out = val ? OaString(val) : OaString();
		duckdb_free(val);
		return out;
	}

	[[nodiscard]] bool IsNull(idx_t InRow, idx_t InCol) const {
		return duckdb_value_is_null(&Result_, InCol, InRow);
	}

private:
	duckdb_result Result_{};
};

#endif

OaDatabase::OaDatabase() = default;

OaDatabase::~OaDatabase() { Close(); }

OaDatabase::OaDatabase(OaDatabase&& InOther) noexcept
	: Db_(InOther.Db_)
	, Conn_(InOther.Conn_)
	, Path_(std::move(InOther.Path_))
	, Open_(InOther.Open_)
	, ReadOnly_(InOther.ReadOnly_) {
	InOther.Db_ = nullptr;
	InOther.Conn_ = nullptr;
	InOther.Open_ = false;
}

OaDatabase& OaDatabase::operator=(OaDatabase&& InOther) noexcept {
	if (this != &InOther) {
		Close();
		Db_ = InOther.Db_;
		Conn_ = InOther.Conn_;
		Path_ = std::move(InOther.Path_);
		Open_ = InOther.Open_;
		ReadOnly_ = InOther.ReadOnly_;
		InOther.Db_ = nullptr;
		InOther.Conn_ = nullptr;
		InOther.Open_ = false;
	}
	return *this;
}

#if !defined(OA_HAS_EMBEDDED_SQL_STORE)

static OaStatus OaDatabaseStubStatus() {
	return OaStatus::Unimplemented(
		"OaDatabase requires OA_WITH_DATABASE=ON and vcpkg feature database (duckdb)"
	);
}

OaStatus OaDatabase::Open(const OaDatabaseConfig& /*InConfig*/) { return OaDatabaseStubStatus(); }

OaStatus OaDatabase::Open(const OaString& /*InPath*/, bool /*InReadOnly*/) { return OaDatabaseStubStatus(); }

void OaDatabase::Close() {
	Db_ = nullptr;
	Conn_ = nullptr;
	Path_.clear();
	Open_ = false;
	ReadOnly_ = false;
}

OaResult<OaDataOhlcv> OaDatabase::LoadData(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/,
	OaOpt<OaTimestamp> /*InStart*/,
	OaOpt<OaTimestamp> /*InEnd*/
) {
	return OaResult<OaDataOhlcv>(OaDatabaseStubStatus());
}

OaResult<OaVec<OaRecordOhlcv>> OaDatabase::LoadRecords(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/,
	OaOpt<OaTimestamp> /*InStart*/,
	OaOpt<OaTimestamp> /*InEnd*/
) {
	return OaResult<OaVec<OaRecordOhlcv>>(OaDatabaseStubStatus());
}

OaStatus OaDatabase::SaveData(
	const OaDataOhlcv& /*InData*/,
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/,
	bool /*InOverwrite*/
) {
	return OaDatabaseStubStatus();
}

OaStatus OaDatabase::SaveRecords(
	const OaVec<OaRecordOhlcv>& /*InRecords*/,
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/,
	bool /*InOverwrite*/
) {
	return OaDatabaseStubStatus();
}

OaResult<OaVec<OaContractTicker>> OaDatabase::GetTickerCombinations() {
	return OaResult<OaVec<OaContractTicker>>(OaDatabaseStubStatus());
}

OaResult<OaVec<OaDataSummary>> OaDatabase::GetAvailableData() {
	return OaResult<OaVec<OaDataSummary>>(OaDatabaseStubStatus());
}

OaResult<OaOpt<OaF64>> OaDatabase::GetEarliestTimestamp(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/
) {
	return OaResult<OaOpt<OaF64>>(OaDatabaseStubStatus());
}

OaResult<OaOpt<OaF64>> OaDatabase::GetLatestTimestamp(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/
) {
	return OaResult<OaOpt<OaF64>>(OaDatabaseStubStatus());
}

OaResult<OaI64> OaDatabase::GetRecordCount(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/
) {
	return OaResult<OaI64>(OaDatabaseStubStatus());
}

OaResult<OaVec<OaF64>> OaDatabase::GetAllTimestamps(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/
) {
	return OaResult<OaVec<OaF64>>(OaDatabaseStubStatus());
}

OaStatus OaDatabase::DeleteData(
	OaStringView /*InTicker*/,
	OaStringView /*InContractType*/,
	OaStringView /*InInterval*/
) {
	return OaDatabaseStubStatus();
}

OaStatus OaDatabase::CompactStorage() { return OaDatabaseStubStatus(); }

OaStatus OaDatabase::ExecuteSQL(OaStringView /*InQuery*/) { return OaDatabaseStubStatus(); }

OaStatus OaDatabase::InitSchema() { return OaDatabaseStubStatus(); }

#else

OaStatus OaDatabase::Open(const OaDatabaseConfig& InConfig) {
	std::unique_lock Lock(Mutex_);

	if (Open_) {
		return OaStatus::Error("OaDatabase already open");
	}

	OaPath DbPath(InConfig.Path);
	if (const OaPath parent = DbPath.ParentPath(); !parent.Empty()) {
		OA_RETURN_IF_ERROR(OaFilesystem::CreateDirectories(parent));
	}

	duckdb_config cfg;
	if (duckdb_create_config(&cfg) != DuckDBSuccess) {
		return OaStatus::Error("Failed to create DuckDB config");
	}

	const OaI32 threads = InConfig.Threads > 0 ? InConfig.Threads : 4;
	duckdb_set_config(cfg, "threads", fmt::format("{}", threads).c_str());
	duckdb_set_config(cfg, "enable_external_access", "true");
	duckdb_set_config(cfg, "access_mode", InConfig.ReadOnly ? "READ_ONLY" : "READ_WRITE");

	duckdb_database db = nullptr;
	char* errOpen = nullptr;
	if (duckdb_open_ext(InConfig.Path.c_str(), &db, cfg, &errOpen) != DuckDBSuccess) {
		OaString msg = errOpen ? OaString(errOpen) : OaString("Unknown error");
		duckdb_free(errOpen);
		duckdb_destroy_config(&cfg);
		return OaStatus::Error(fmt::format("Failed to open database: {}", msg));
	}
	duckdb_destroy_config(&cfg);

	duckdb_connection conn = nullptr;
	if (duckdb_connect(db, &conn) != DuckDBSuccess) {
		duckdb_close(&db);
		return OaStatus::Error("Failed to create database connection");
	}

	Db_ = db;
	Conn_ = conn;
	Path_ = InConfig.Path;
	Open_ = true;
	ReadOnly_ = InConfig.ReadOnly;

	if (!ReadOnly_) {
		OaStatus InitSt = InitSchema();
		if (!InitSt.IsOk()) {
			duckdb_disconnect(&conn);
			duckdb_close(&db);
			Db_ = nullptr;
			Conn_ = nullptr;
			Open_ = false;
			return InitSt;
		}
	}

	return OaStatus::Ok();
}

OaStatus OaDatabase::Open(const OaString& InPath, bool InReadOnly) {
	OaDatabaseConfig Cfg;
	Cfg.Path = InPath;
	Cfg.ReadOnly = InReadOnly;
	return Open(Cfg);
}

void OaDatabase::Close() {
	std::unique_lock Lock(Mutex_);

	if (Conn_) {
		duckdb_connection c = static_cast<duckdb_connection>(Conn_);
		duckdb_disconnect(&c);
		Conn_ = nullptr;
	}
	if (Db_) {
		duckdb_database d = static_cast<duckdb_database>(Db_);
		duckdb_close(&d);
		Db_ = nullptr;
	}
	Open_ = false;
	Path_.clear();
}

OaStatus OaDatabase::InitSchema() {
	OaDuckdbResult R;
	if (duckdb_query(static_cast<duckdb_connection>(Conn_), OaString(kCreateTableSQL).c_str(), R.Get()) != DuckDBSuccess) {
		return OaStatus::Error(fmt::format("Failed to create table: {}", R.GetError()));
	}
	OaDuckdbResult Idx;
	if (duckdb_query(static_cast<duckdb_connection>(Conn_), OaString(kCreateIndexSQL).c_str(), Idx.Get()) != DuckDBSuccess) {
		return OaStatus::Error(fmt::format("Failed to create index: {}", Idx.GetError()));
	}
	return OaStatus::Ok();
}

OaResult<OaDataOhlcv> OaDatabase::LoadData(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval,
	OaOpt<OaTimestamp> InStart,
	OaOpt<OaTimestamp> InEnd
) {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaDataOhlcv>(OaStatus::Error("OaDatabase not open"));
	}

	OaString Sql =
		"SELECT Timestamp, Open, High, Low, Close, Volume FROM ohlcv "
		"WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3";
	if (InStart.has_value()) {
		Sql += fmt::format(" AND Timestamp >= {:.9f}", InStart->ToSeconds());
	}
	if (InEnd.has_value()) {
		Sql += fmt::format(" AND Timestamp <= {:.9f}", InEnd->ToSeconds());
	}
	Sql += " ORDER BY Timestamp ASC";

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(static_cast<duckdb_connection>(Conn_), Sql.c_str(), &Stmt) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaDataOhlcv>(OaStatus::Error(E));
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("bind failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaDataOhlcv>(OaStatus::Error(E));
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaDataOhlcv>(OaStatus::Error(E));
	}
	duckdb_destroy_prepare(&Stmt);

	const idx_t n = Res.RowCount();
	OaDataOhlcv Data;
	Data.Time.resize(n);
	Data.Open.resize(n);
	Data.High.resize(n);
	Data.Low.resize(n);
	Data.Close.resize(n);
	Data.Volume.resize(n);

	for (idx_t i = 0; i < n; ++i) {
		Data.Time[i] = Res.GetDouble(i, 0);
		Data.Open[i] = Res.GetDouble(i, 1);
		Data.High[i] = Res.GetDouble(i, 2);
		Data.Low[i] = Res.GetDouble(i, 3);
		Data.Close[i] = Res.GetDouble(i, 4);
		Data.Volume[i] = Res.GetDouble(i, 5);
	}

	return OaResult<OaDataOhlcv>(std::move(Data));
}

OaResult<OaVec<OaRecordOhlcv>> OaDatabase::LoadRecords(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval,
	OaOpt<OaTimestamp> InStart,
	OaOpt<OaTimestamp> InEnd
) {
	auto Dr = LoadData(InTicker, InContractType, InInterval, InStart, InEnd);
	if (!Dr.IsOk()) {
		return OaResult<OaVec<OaRecordOhlcv>>(Dr.GetStatus());
	}
	const OaDataOhlcv& D = *Dr;
	OaVec<OaRecordOhlcv> Recs;
	Recs.reserve(D.Size());
	for (OaUsize i = 0; i < D.Size(); ++i) {
		Recs.push_back(D.GetRecord(i));
	}
	return OaResult<OaVec<OaRecordOhlcv>>(std::move(Recs));
}

OaStatus OaDatabase::SaveData(
	const OaDataOhlcv& InData,
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval,
	bool InOverwrite
) {
	std::unique_lock Lock(Mutex_);

	if (!Open_) {
		return OaStatus::Error("OaDatabase not open");
	}
	if (ReadOnly_) {
		return OaStatus::Error("Cannot save to read-only database");
	}
	if (InData.IsEmpty()) {
		return OaStatus::Ok();
	}

	duckdb_connection Conn = static_cast<duckdb_connection>(Conn_);

	if (InOverwrite) {
		duckdb_prepared_statement Del = nullptr;
		if (duckdb_prepare(Conn, "DELETE FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3", &Del)
			!= DuckDBSuccess) {
			const char* pe = duckdb_prepare_error(Del);
			OaString E = pe ? OaString(pe) : OaString("prepare delete failed");
			duckdb_destroy_prepare(&Del);
			return OaStatus::Error(E);
		}
		OaString T(InTicker);
		OaString C(InContractType);
		OaString I(InInterval);
		if (duckdb_bind_varchar(Del, 1, T.c_str()) != DuckDBSuccess
			|| duckdb_bind_varchar(Del, 2, C.c_str()) != DuckDBSuccess
			|| duckdb_bind_varchar(Del, 3, I.c_str()) != DuckDBSuccess) {
			duckdb_destroy_prepare(&Del);
			return OaStatus::Error("bind delete failed");
		}
		OaDuckdbResult DelRes;
		if (duckdb_execute_prepared(Del, DelRes.Get()) != DuckDBSuccess) {
			OaString E = DelRes.GetError();
			duckdb_destroy_prepare(&Del);
			return OaStatus::Error(E);
		}
		duckdb_destroy_prepare(&Del);
	}

	duckdb_appender App = nullptr;
	if (duckdb_appender_create(Conn, nullptr, "ohlcv", &App) != DuckDBSuccess) {
		return OaStatus::Error("Failed to create appender");
	}

	const OaF64 CreatedAt = static_cast<OaF64>(
		std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
	);

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);

	for (OaUsize j = 0; j < InData.Size(); ++j) {
		duckdb_append_double(App, InData.Time[j]);
		duckdb_append_varchar(App, T.c_str());
		duckdb_append_varchar(App, C.c_str());
		duckdb_append_varchar(App, I.c_str());
		duckdb_append_double(App, InData.Open[j]);
		duckdb_append_double(App, InData.High[j]);
		duckdb_append_double(App, InData.Low[j]);
		duckdb_append_double(App, InData.Close[j]);
		duckdb_append_double(App, InData.Volume[j]);
		duckdb_append_double(App, CreatedAt);
		if (duckdb_appender_end_row(App) != DuckDBSuccess) {
			duckdb_appender_destroy(&App);
			return OaStatus::Error("Failed to append row");
		}
	}

	if (duckdb_appender_flush(App) != DuckDBSuccess) {
		const char* ae = duckdb_appender_error(App);
		OaString Msg = ae ? OaString(ae) : OaString("Unknown appender error");
		duckdb_appender_destroy(&App);
		return OaStatus::Error(fmt::format("Failed to flush appender: {}", Msg));
	}
	duckdb_appender_destroy(&App);
	return OaStatus::Ok();
}

OaStatus OaDatabase::SaveRecords(
	const OaVec<OaRecordOhlcv>& InRecords,
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval,
	bool InOverwrite
) {
	OaDataOhlcv Col(InRecords.size());
	for (const OaRecordOhlcv& R : InRecords) {
		Col.PushBack(R);
	}
	return SaveData(Col, InTicker, InContractType, InInterval, InOverwrite);
}

OaResult<OaVec<OaContractTicker>> OaDatabase::GetTickerCombinations() {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaVec<OaContractTicker>>(OaStatus::Error("OaDatabase not open"));
	}

	OaDuckdbResult Res;
	if (duckdb_query(
			static_cast<duckdb_connection>(Conn_),
			"SELECT DISTINCT Ticker, ContractType FROM ohlcv ORDER BY Ticker, ContractType",
			Res.Get()
		) != DuckDBSuccess) {
		return OaResult<OaVec<OaContractTicker>>(OaStatus::Error(Res.GetError()));
	}

	OaVec<OaContractTicker> Out;
	for (idx_t i = 0; i < Res.RowCount(); ++i) {
		OaContractTicker Row;
		Row.OaTicker = Res.GetString(i, 0);
		Row.ContractType = Res.GetString(i, 1);
		Out.push_back(std::move(Row));
	}
	return OaResult<OaVec<OaContractTicker>>(std::move(Out));
}

OaResult<OaVec<OaDataSummary>> OaDatabase::GetAvailableData() {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaVec<OaDataSummary>>(OaStatus::Error("OaDatabase not open"));
	}

	static const char* Q = R"sql(
		SELECT
			Ticker,
			ContractType,
			Interval,
			COUNT(*) AS RecordCount,
			MIN(Timestamp) AS Earliest,
			MAX(Timestamp) AS Latest
		FROM ohlcv
		GROUP BY Ticker, ContractType, Interval
		ORDER BY Ticker, ContractType, Interval
	)sql";

	OaDuckdbResult Res;
	if (duckdb_query(static_cast<duckdb_connection>(Conn_), Q, Res.Get()) != DuckDBSuccess) {
		return OaResult<OaVec<OaDataSummary>>(OaStatus::Error(Res.GetError()));
	}

	OaVec<OaDataSummary> Out;
	for (idx_t i = 0; i < Res.RowCount(); ++i) {
		OaDataSummary S;
		S.OaTicker = Res.GetString(i, 0);
		S.ContractType = Res.GetString(i, 1);
		S.Interval = Res.GetString(i, 2);
		S.RecordCount = Res.GetInt64(i, 3);
		S.EarliestTimestamp = Res.GetDouble(i, 4);
		S.LatestTimestamp = Res.GetDouble(i, 5);
		Out.push_back(std::move(S));
	}
	return OaResult<OaVec<OaDataSummary>>(std::move(Out));
}

OaResult<OaOpt<OaF64>> OaDatabase::GetEarliestTimestamp(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval
) {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaOpt<OaF64>>(OaStatus::Error("OaDatabase not open"));
	}

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(
			static_cast<duckdb_connection>(Conn_),
			"SELECT MIN(Timestamp) FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3",
			&Stmt
		) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error(E));
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error("bind failed"));
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error(E));
	}
	duckdb_destroy_prepare(&Stmt);

	if (Res.RowCount() == 0 || Res.IsNull(0, 0)) {
		return OaResult<OaOpt<OaF64>>(OaOpt<OaF64>(std::nullopt));
	}
	return OaResult<OaOpt<OaF64>>(OaOpt<OaF64>(Res.GetDouble(0, 0)));
}

OaResult<OaOpt<OaF64>> OaDatabase::GetLatestTimestamp(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval
) {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaOpt<OaF64>>(OaStatus::Error("OaDatabase not open"));
	}

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(
			static_cast<duckdb_connection>(Conn_),
			"SELECT MAX(Timestamp) FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3",
			&Stmt
		) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error(E));
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error("bind failed"));
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaOpt<OaF64>>(OaStatus::Error(E));
	}
	duckdb_destroy_prepare(&Stmt);

	if (Res.RowCount() == 0 || Res.IsNull(0, 0)) {
		return OaResult<OaOpt<OaF64>>(OaOpt<OaF64>(std::nullopt));
	}
	return OaResult<OaOpt<OaF64>>(OaOpt<OaF64>(Res.GetDouble(0, 0)));
}

OaResult<OaI64> OaDatabase::GetRecordCount(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval
) {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaI64>(OaStatus::Error("OaDatabase not open"));
	}

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(
			static_cast<duckdb_connection>(Conn_),
			"SELECT COUNT(*) FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3",
			&Stmt
		) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaI64>(OaStatus::Error(E));
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaI64>(OaStatus::Error("bind failed"));
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaI64>(OaStatus::Error(E));
	}
	duckdb_destroy_prepare(&Stmt);

	return OaResult<OaI64>(Res.GetInt64(0, 0));
}

OaResult<OaVec<OaF64>> OaDatabase::GetAllTimestamps(
	OaStringView InTicker,
	OaStringView InContractType,
	OaStringView InInterval
) {
	std::shared_lock Lock(Mutex_);

	if (!Open_) {
		return OaResult<OaVec<OaF64>>(OaStatus::Error("OaDatabase not open"));
	}

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(
			static_cast<duckdb_connection>(Conn_),
			"SELECT Timestamp FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3 ORDER BY Timestamp ASC",
			&Stmt
		) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaVec<OaF64>>(OaStatus::Error(E));
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaVec<OaF64>>(OaStatus::Error("bind failed"));
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaResult<OaVec<OaF64>>(OaStatus::Error(E));
	}
	duckdb_destroy_prepare(&Stmt);

	OaVec<OaF64> Ts;
	const idx_t n = Res.RowCount();
	Ts.reserve(static_cast<OaUsize>(n));
	for (idx_t i = 0; i < n; ++i) {
		Ts.push_back(Res.GetDouble(i, 0));
	}
	return OaResult<OaVec<OaF64>>(std::move(Ts));
}

OaStatus OaDatabase::DeleteData(OaStringView InTicker, OaStringView InContractType, OaStringView InInterval) {
	std::unique_lock Lock(Mutex_);

	if (!Open_) {
		return OaStatus::Error("OaDatabase not open");
	}
	if (ReadOnly_) {
		return OaStatus::Error("Cannot delete from read-only database");
	}

	duckdb_prepared_statement Stmt = nullptr;
	if (duckdb_prepare(
			static_cast<duckdb_connection>(Conn_),
			"DELETE FROM ohlcv WHERE Ticker = $1 AND ContractType = $2 AND Interval = $3",
			&Stmt
		) != DuckDBSuccess) {
		const char* pe = duckdb_prepare_error(Stmt);
		OaString E = pe ? OaString(pe) : OaString("prepare failed");
		duckdb_destroy_prepare(&Stmt);
		return OaStatus::Error(E);
	}

	OaString T(InTicker);
	OaString C(InContractType);
	OaString I(InInterval);
	if (duckdb_bind_varchar(Stmt, 1, T.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 2, C.c_str()) != DuckDBSuccess
		|| duckdb_bind_varchar(Stmt, 3, I.c_str()) != DuckDBSuccess) {
		duckdb_destroy_prepare(&Stmt);
		return OaStatus::Error("bind failed");
	}

	OaDuckdbResult Res;
	if (duckdb_execute_prepared(Stmt, Res.Get()) != DuckDBSuccess) {
		OaString E = Res.GetError();
		duckdb_destroy_prepare(&Stmt);
		return OaStatus::Error(E);
	}
	duckdb_destroy_prepare(&Stmt);
	return OaStatus::Ok();
}

OaStatus OaDatabase::CompactStorage() {
	std::unique_lock Lock(Mutex_);

	if (!Open_) {
		return OaStatus::Error("OaDatabase not open");
	}
	if (ReadOnly_) {
		return OaStatus::Error("Cannot checkpoint read-only database");
	}

	OaDuckdbResult Res;
	if (duckdb_query(static_cast<duckdb_connection>(Conn_), "CHECKPOINT", Res.Get()) != DuckDBSuccess) {
		return OaStatus::Error(fmt::format("CHECKPOINT failed: {}", Res.GetError()));
	}
	return OaStatus::Ok();
}

OaStatus OaDatabase::ExecuteSQL(OaStringView InQuery) {
	std::unique_lock Lock(Mutex_);

	if (!Open_) {
		return OaStatus::Error("OaDatabase not open");
	}

	OaDuckdbResult Res;
	if (duckdb_query(static_cast<duckdb_connection>(Conn_), OaString(InQuery).c_str(), Res.Get()) != DuckDBSuccess) {
		return OaStatus::Error(fmt::format("SQL failed: {}", Res.GetError()));
	}
	return OaStatus::Ok();
}

#endif

OaResult<OaUniquePtr<OaDatabase>> OaOpenDatabase(const OaString& InPath, bool InReadOnly) {
	OaUniquePtr<OaDatabase> Db = OaMakeUniquePtr<OaDatabase>();
	OaStatus St = Db->Open(InPath, InReadOnly);
	if (!St.IsOk()) {
		return OaResult<OaUniquePtr<OaDatabase>>(std::move(St));
	}
	return OaResult<OaUniquePtr<OaDatabase>>(std::move(Db));
}

OaResult<OaUniquePtr<OaDatabase>> OaOpenDatabaseReadOnly(const OaString& InPath) {
	return OaOpenDatabase(InPath, true);
}
