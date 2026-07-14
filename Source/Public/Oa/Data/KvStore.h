// OA — Abstract ordered key-value store (byte keys/values). Chain RocksDB maps here.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaIKeyValueStore {
public:
	virtual ~OaIKeyValueStore() = default;

	[[nodiscard]] virtual OaStatus Open(OaStringView InPath, bool InReadOnly = false) = 0;
	virtual void Close() = 0;
	[[nodiscard]] virtual bool IsOpen() const noexcept = 0;

	[[nodiscard]] virtual OaStatus Get(OaSpan<const OaU8> InKey, OaVec<OaU8>& OutValue) = 0;
	[[nodiscard]] virtual OaStatus Put(OaSpan<const OaU8> InKey, OaSpan<const OaU8> InValue) = 0;
	[[nodiscard]] virtual OaStatus Delete(OaSpan<const OaU8> InKey) = 0;
};
