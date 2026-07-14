#include <Oa/Core/Thread.h>
#include <Oa/Core/Log.h>

#include <cstdio>
#include <cstring>
#include <thread>

#ifdef OA_PLATFORM_LINUX
#include <dirent.h>
#include <unistd.h>
#endif

static OaI32 ReadSysfsInt(const char* InPath) {
#ifdef OA_PLATFORM_LINUX
	FILE* f = fopen(InPath, "r");
	if (!f) return -1;
	OaI32 val = -1;
	if (fscanf(f, "%d", &val) != 1) val = -1;
	fclose(f);
	return val;
#else
	(void)InPath;
	return -1;
#endif
}

static OaI32 CountOnlineCpus() {
#ifdef OA_PLATFORM_LINUX
	OaI32 count = 0;
	for (OaI32 i = 0; i < 1024; ++i) {
		char path[256];
		snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", i);
		FILE* f = fopen(path, "r");
		if (!f) {
			if (i == 0) { ++count; continue; }
			break;
		}
		OaI32 online = 0;
		if (fscanf(f, "%d", &online) == 1 && online) ++count;
		fclose(f);
	}
	if (count == 0) count = static_cast<OaI32>(std::thread::hardware_concurrency());
	return count;
#else
	return static_cast<OaI32>(std::thread::hardware_concurrency());
#endif
}

OaCpuTopology OaCpuTopology::Detect() {
	OaCpuTopology topo;
	topo.NumLogicalCores = static_cast<OaI32>(std::thread::hardware_concurrency());
	if (topo.NumLogicalCores <= 0) topo.NumLogicalCores = 1;

#ifdef OA_PLATFORM_LINUX
	OaI32 maxCpu = CountOnlineCpus();
	if (maxCpu <= 0) maxCpu = topo.NumLogicalCores;

	OaI32 maxFreqGlobal = 0;
	OaI32 maxPkg = 0;
	OaI32 maxNuma = 0;

	topo.Cores.Resize(maxCpu);

	for (OaI32 i = 0; i < maxCpu; ++i) {
		auto& core = topo.Cores[i];
		core.Id = i;

		char path[256];

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
		core.MaxFreqKhz = ReadSysfsInt(path);

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
		OaI32 pkg = ReadSysfsInt(path);
		core.PackageId = (pkg >= 0) ? pkg : 0;

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/core_id", i);
		// core_id used for counting physical cores later

		// NUMA node: check which node directory contains this cpu
		core.NumaNode = 0;
		for (OaI32 n = 0; n < 16; ++n) {
			snprintf(path, sizeof(path),
				"/sys/devices/system/node/node%d/cpu%d", n, i);
			FILE* f = fopen(path, "r");
			if (f) { core.NumaNode = n; fclose(f); break; }
		}

		if (core.MaxFreqKhz > maxFreqGlobal) maxFreqGlobal = core.MaxFreqKhz;
		if (core.PackageId > maxPkg) maxPkg = core.PackageId;
		if (core.NumaNode > maxNuma) maxNuma = core.NumaNode;
	}

	topo.NumPackages = maxPkg + 1;
	topo.NumNumaNodes = maxNuma + 1;

	// P/E core classification: 80% threshold of max frequency
	if (maxFreqGlobal > 0) {
		OaI32 threshold = maxFreqGlobal * 80 / 100;
		bool hasSplit = false;
		for (auto& core : topo.Cores) {
			if (core.MaxFreqKhz >= threshold) {
				core.Type = OaCoreType::Performance;
			} else if (core.MaxFreqKhz > 0) {
				core.Type = OaCoreType::Efficiency;
				hasSplit = true;
			} else {
				core.Type = OaCoreType::Performance;
			}
		}
		if (!hasSplit) {
			for (auto& core : topo.Cores)
				core.Type = OaCoreType::Performance;
		}
	} else {
		for (auto& core : topo.Cores)
			core.Type = OaCoreType::Performance;
	}

	// Count physical cores via unique (package_id, core_id) pairs
	OaVec<OaU64> seen;
	for (OaI32 i = 0; i < maxCpu; ++i) {
		char path[256];
		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/core_id", i);
		OaI32 coreId = ReadSysfsInt(path);
		if (coreId < 0) coreId = i;
		OaU64 key = (static_cast<OaU64>(topo.Cores[i].PackageId) << 32)
			| static_cast<OaU64>(coreId);
		bool found = false;
		for (auto k : seen) { if (k == key) { found = true; break; } }
		if (!found) seen.PushBack(key);
	}
	topo.NumPhysicalCores = static_cast<OaI32>(seen.Size());

#else
	// Non-Linux fallback: all cores are Performance, no topology info
	topo.Cores.resize(topo.NumLogicalCores);
	for (OaI32 i = 0; i < topo.NumLogicalCores; ++i) {
		topo.Cores[i].Id = i;
		topo.Cores[i].Type = OaCoreType::Performance;
	}
	topo.NumPhysicalCores = topo.NumLogicalCores;
#endif

	return topo;
}

OaVec<OaI32> OaCpuTopology::GetPcoreIds() const {
	OaVec<OaI32> ids;
	for (auto& c : Cores) {
		if (c.Type == OaCoreType::Performance) ids.PushBack(c.Id);
	}
	return ids;
}

OaVec<OaI32> OaCpuTopology::GetEcoreIds() const {
	OaVec<OaI32> ids;
	for (auto& c : Cores) {
		if (c.Type == OaCoreType::Efficiency) ids.PushBack(c.Id);
	}
	return ids;
}

OaVec<OaI32> OaCpuTopology::GetCoresOnNuma(OaI32 InNode) const {
	OaVec<OaI32> ids;
	for (auto& c : Cores) {
		if (c.NumaNode == InNode) ids.PushBack(c.Id);
	}
	return ids;
}

void OaCpuTopology::Print() const {
	OA_LOG_INFO(OaLogComponent::Core,
		"CPU: %d logical, %d physical, %d NUMA, %d pkg",
		NumLogicalCores, NumPhysicalCores, NumNumaNodes, NumPackages);

	OaI32 pCount = 0, eCount = 0;
	for (auto& c : Cores) {
		if (c.Type == OaCoreType::Performance) ++pCount;
		else if (c.Type == OaCoreType::Efficiency) ++eCount;
	}

	if (eCount > 0) {
		OA_LOG_INFO(OaLogComponent::Core,
			"  P-cores: %d, E-cores: %d", pCount, eCount);
	}

	if (NumNumaNodes > 1) {
		for (OaI32 n = 0; n < NumNumaNodes; ++n) {
			auto ids = GetCoresOnNuma(n);
			OA_LOG_INFO(OaLogComponent::Core,
				"  NUMA %d: %d cores", n, static_cast<OaI32>(ids.Size()));
		}
	}
}
