#include <oa/core/thread.h>
#include <oa/core/log.h>

#include <stdio.h>
#include <string.h>

#ifdef OA_PLATFORM_LINUX
#include <dirent.h>
#include <unistd.h>
#endif

static oa::I32 readSysfsInt(const char* inPath) {
#ifdef OA_PLATFORM_LINUX
	FILE* f = fopen(inPath, "r");
	if (!f) return -1;
	oa::I32 val = -1;
	if (fscanf(f, "%d", &val) != 1) val = -1;
	fclose(f);
	return val;
#else
	(void)inPath;
	return -1;
#endif
}

static oa::I32 countOnlineCpus() {
#ifdef OA_PLATFORM_LINUX
	oa::I32 count = 0;
	for (oa::I32 i = 0; i < 1024; ++i) {
		char path[256];
		snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/online", i);
		FILE* f = fopen(path, "r");
		if (!f) {
			if (i == 0) { ++count; continue; }
			break;
		}
		oa::I32 online = 0;
		if (fscanf(f, "%d", &online) == 1 && online) ++count;
		fclose(f);
	}
	if (count == 0) count = static_cast<oa::I32>(oa::Thread::hardwareConcurrency());
	return count;
#else
	return static_cast<oa::I32>(oa::Thread::hardwareConcurrency());
#endif
}

oa::CpuTopology oa::CpuTopology::detect() {
	oa::CpuTopology topo;
	topo.numLogicalCores = static_cast<oa::I32>(oa::Thread::hardwareConcurrency());
	if (topo.numLogicalCores <= 0) topo.numLogicalCores = 1;

#ifdef OA_PLATFORM_LINUX
	oa::I32 maxCpu = countOnlineCpus();
	if (maxCpu <= 0) maxCpu = topo.numLogicalCores;

	oa::I32 maxFreqGlobal = 0;
	oa::I32 maxPkg = 0;
	oa::I32 maxNuma = 0;

	topo.cores.resize(maxCpu);

	for (oa::I32 i = 0; i < maxCpu; ++i) {
		auto& core = topo.cores[i];
		core.id = i;

		char path[256];

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
		core.maxFreqKhz = readSysfsInt(path);

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
		oa::I32 pkg = readSysfsInt(path);
		core.packageId = (pkg >= 0) ? pkg : 0;

		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/core_id", i);
		// core_id used for counting physical cores later

		// NUMA node: check which node directory contains this cpu
		core.numaNode = 0;
		for (oa::I32 n = 0; n < 16; ++n) {
			snprintf(path, sizeof(path),
				"/sys/devices/system/node/node%d/cpu%d", n, i);
			FILE* f = fopen(path, "r");
			if (f) { core.numaNode = n; fclose(f); break; }
		}

		if (core.maxFreqKhz > maxFreqGlobal) maxFreqGlobal = core.maxFreqKhz;
		if (core.packageId > maxPkg) maxPkg = core.packageId;
		if (core.numaNode > maxNuma) maxNuma = core.numaNode;
	}

	topo.numPackages = maxPkg + 1;
	topo.numNumaNodes = maxNuma + 1;

	// P/E core classification: 80% threshold of max frequency
	if (maxFreqGlobal > 0) {
		oa::I32 threshold = maxFreqGlobal * 80 / 100;
		bool hasSplit = false;
		for (auto& core : topo.cores) {
			if (core.maxFreqKhz >= threshold) {
				core.type = oa::CoreType::Performance;
			} else if (core.maxFreqKhz > 0) {
				core.type = oa::CoreType::Efficiency;
				hasSplit = true;
			} else {
				core.type = oa::CoreType::Performance;
			}
		}
		if (!hasSplit) {
			for (auto& core : topo.cores)
				core.type = oa::CoreType::Performance;
		}
	} else {
		for (auto& core : topo.cores)
			core.type = oa::CoreType::Performance;
	}

	// Count physical cores via unique (package_id, core_id) pairs
	oa::Vec<oa::U64> seen;
	for (oa::I32 i = 0; i < maxCpu; ++i) {
		char path[256];
		snprintf(path, sizeof(path),
			"/sys/devices/system/cpu/cpu%d/topology/core_id", i);
		oa::I32 coreId = readSysfsInt(path);
		if (coreId < 0) coreId = i;
		oa::U64 key = (static_cast<oa::U64>(topo.cores[i].packageId) << 32)
			| static_cast<oa::U64>(coreId);
		bool found = false;
		for (auto k : seen) { if (k == key) { found = true; break; } }
		if (!found) seen.pushBack(key);
	}
	topo.numPhysicalCores = static_cast<oa::I32>(seen.size());

#else
	// Non-Linux fallback: all cores are Performance, no topology info
	topo.cores.resize(topo.numLogicalCores);
	for (oa::I32 i = 0; i < topo.numLogicalCores; ++i) {
		topo.cores[i].id = i;
		topo.cores[i].type = oa::CoreType::Performance;
	}
	topo.numPhysicalCores = topo.numLogicalCores;
#endif

	return topo;
}

oa::Vec<oa::I32> oa::CpuTopology::getPcoreIds() const {
	oa::Vec<oa::I32> ids;
	for (auto& c : cores) {
		if (c.type == oa::CoreType::Performance) ids.pushBack(c.id);
	}
	return ids;
}

oa::Vec<oa::I32> oa::CpuTopology::getEcoreIds() const {
	oa::Vec<oa::I32> ids;
	for (auto& c : cores) {
		if (c.type == oa::CoreType::Efficiency) ids.pushBack(c.id);
	}
	return ids;
}

oa::Vec<oa::I32> oa::CpuTopology::getCoresOnNuma(oa::I32 inNode) const {
	oa::Vec<oa::I32> ids;
	for (auto& c : cores) {
		if (c.numaNode == inNode) ids.pushBack(c.id);
	}
	return ids;
}

void oa::CpuTopology::print() const {
	OaLogInfo(oa::LogComponent::Core,
		"CPU: %d logical, %d physical, %d NUMA, %d pkg",
		numLogicalCores, numPhysicalCores, numNumaNodes, numPackages);

	oa::I32 pCount = 0, eCount = 0;
	for (auto& c : cores) {
		if (c.type == oa::CoreType::Performance) ++pCount;
		else if (c.type == oa::CoreType::Efficiency) ++eCount;
	}

	if (eCount > 0) {
		OaLogInfo(oa::LogComponent::Core,
			"  P-cores: %d, E-cores: %d", pCount, eCount);
	}

	if (numNumaNodes > 1) {
		for (oa::I32 n = 0; n < numNumaNodes; ++n) {
			auto ids = getCoresOnNuma(n);
			OaLogInfo(oa::LogComponent::Core,
				"  NUMA %d: %d cores", n, static_cast<oa::I32>(ids.size()));
		}
	}
}
