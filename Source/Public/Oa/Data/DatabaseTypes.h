// OA — POD types shared by OHLCV store interfaces and OaDatabase (no heavy includes).

#pragma once

#include <Oa/Core/Types.h>

struct OaDataSummary {
	OaString OaTicker;
	OaString ContractType;
	OaString Interval;
	OaI64 RecordCount = 0;
	OaF64 EarliestTimestamp = 0;
	OaF64 LatestTimestamp = 0;
	OaF64 Coverage = 0;
};

struct OaContractTicker {
	OaString OaTicker;
	OaString ContractType;
};

struct OaDatabaseConfig {
	OaString Path = "var/db/financial.db";
	bool ReadOnly = false;
	OaI32 CacheMegabytes = 256;
	OaI32 Threads = 0;
};
