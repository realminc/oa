// OA — High-level storage family tags for documentation and future factory helpers.

#pragma once

#include <Oa/Core/Types.h>

enum class OaStoreKind : OaU8 {
	KvOrdered = 0,
	SqlEmbedded = 1,
	Composite = 2,
};

// KvOrdered: byte-key LSM-style stores (e.g. RocksDB) — chain ledger/state/WAL.
// SqlEmbedded: single-file or in-process SQL (e.g. DuckDB) — analytics, OHLCV.
// Composite: multiple backends under one facade (e.g. cluster shared DB + per-validator WAL).
