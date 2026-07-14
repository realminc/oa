#include <Oa/Runtime/Gemm/Cache.h>
#include <Oa/Core/Log.h>

#include <fstream>

// ---------------------------------------------------------------------------
// OaBlasKernelCache
// ---------------------------------------------------------------------------

OaGemmKernel OaBlasKernelCache::Lookup(OaU32 InM, OaU32 InN, OaU32 InK) const noexcept {
	std::lock_guard<std::mutex> lock(Mutex_);
	auto it = Map_.Find({ InM, InN, InK });
	return it != Map_.End() ? it->second : OaGemmKernel::Auto;
}

void OaBlasKernelCache::Store(OaU32 InM, OaU32 InN, OaU32 InK, OaGemmKernel InKernel) {
	std::lock_guard<std::mutex> lock(Mutex_);
	Map_.Insert({ { InM, InN, InK }, InKernel });
}

void OaBlasKernelCache::Clear() noexcept {
	std::lock_guard<std::mutex> lock(Mutex_);
	while (Map_.Begin() != Map_.End()) {
		Map_.Erase(Map_.Begin()->first);
	}
}

OaU32 OaBlasKernelCache::Size() const noexcept{
	std::lock_guard<std::mutex> lock(Mutex_);
	OaU32 count = 0;
	for (auto it = Map_.Begin(); it != Map_.End(); ++it) {
		++count;
	}
	return count;
}

// File format: [magic(4)] [count(4)] [M(4) N(4) K(4) kernel(1) pad(3)] * count
static constexpr OaU32 kFileMagic = 0x4F424C53; // 'OBLS'

OaStatus OaBlasKernelCache::SaveToFile(OaStringView InPath) const
{
	std::lock_guard<std::mutex> lock(Mutex_);
	std::ofstream f(InPath.Data(), std::ios::binary);
	if (!f) {
		OaString msg = OaString("OaBlasKernelCache: cannot open '") + InPath.Data() + "' for write";
		return OaStatus::Error(msg);
	}

	OaU32 magic = kFileMagic;
	OaU32 count = Size();
	f.write(reinterpret_cast<const char*>(&magic), 4);
	f.write(reinterpret_cast<const char*>(&count), 4);

	for (auto it = Map_.Begin(); it != Map_.End(); ++it) {
		const auto &key = it->first;
		OaU8 k = static_cast<OaU8>(it->second);
		f.write(reinterpret_cast<const char*>(&key.M), 4);
		f.write(reinterpret_cast<const char*>(&key.N), 4);
		f.write(reinterpret_cast<const char*>(&key.K), 4);
		f.write(reinterpret_cast<const char*>(&k), 1);
		OaU8 pad[3] = {};
		f.write(reinterpret_cast<const char*>(pad), 3);
	}

	OA_LOG_INFO(OaLogComponent::Core, "OaBlasKernelCache: saved {} entries to '{}'", count, InPath);
	return OaStatus::Ok();
}

OaStatus OaBlasKernelCache::LoadFromFile(OaStringView InPath) {
	std::ifstream f(InPath.Data(), std::ios::binary);
	if (!f) {
		OaString msg = OaString("OaBlasKernelCache: cannot open '") + InPath.Data() + "' for read";
		return OaStatus::Error(msg);
	}

	OaU32 magic = 0;
	OaU32 count = 0;
	f.read(reinterpret_cast<char*>(&magic), 4);
	f.read(reinterpret_cast<char*>(&count), 4);
	if (magic != kFileMagic) {
		char buf[128];
		std::snprintf(buf, sizeof(buf), "OaBlasKernelCache: bad magic in '%s' (got 0x%x)",
		              InPath.Data(), magic);
		return OaStatus::Error(OaString(buf));
	}

	std::lock_guard<std::mutex> lock(Mutex_);
	Clear();

	for (OaU32 i = 0; i < count; i++) {
		OaGemmShapeKey key;
		OaU8 k = 0;
		OaU8 pad[3];
		f.read(reinterpret_cast<char*>(&key.M), 4);
		f.read(reinterpret_cast<char*>(&key.N), 4);
		f.read(reinterpret_cast<char*>(&key.K), 4);
		f.read(reinterpret_cast<char*>(&k), 1);
		f.read(reinterpret_cast<char*>(pad), 3);
		Map_.Insert({ key, static_cast<OaGemmKernel>(k) });
	}

	OA_LOG_INFO(OaLogComponent::Core, "OaBlasKernelCache: loaded {} entries from '{}'", count, InPath);
	return OaStatus::Ok();
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------

OaBlasKernelCache &OaGetGemmKernelCache() noexcept {
	static OaBlasKernelCache g_Cache;
	return g_Cache;
}
