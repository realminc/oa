// OA — Abstract embedded SQL session surface (maintenance + ad-hoc queries). DuckDB implements via OaDatabase.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaISqlStore {
public:
	virtual ~OaISqlStore() = default;

	[[nodiscard]] virtual OaStatus CompactStorage() = 0;
	[[nodiscard]] virtual OaStatus ExecuteSQL(OaStringView InQuery) = 0;
};
