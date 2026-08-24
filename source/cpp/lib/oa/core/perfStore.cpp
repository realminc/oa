// OA CORE - persistent Performance Record store

#include <oa/core/perfStore.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/time.h>
#include <cstring>
#include <cstdio>

static constexpr oa::U32 kMagic   = 0x4F415046U;
static constexpr oa::U32 kVersion = 1U;

struct PerfFileHeader {
    oa::U32 magic;
    oa::U32 version;
    oa::U32 entryCount;
    oa::U32 reserved[5];
};
static_assert(sizeof(PerfFileHeader) == 32, "header must be 32 bytes");

oa::Status oa::PerfStore::load(const char* inPath) {
    if (inPath != nullptr and inPath[0] != '\0') {
        path_ = oa::String(oa::StringView(inPath));
    } else {
        path_ = oa::String((oa::Paths::var("perf") / "OaPerf.dat").string());
    }

    records_.clear();

    if (not oa::Filesystem::exists(oa::Path(path_))) {
        return oa::Status::ok();
    }

    auto rawResult = oa::Filesystem::readBinary(oa::Path(path_));
    if (not rawResult.isOk()) {
        return rawResult.getStatus();
    }
    const auto& raw = rawResult.getValue();

    if (raw.size() < sizeof(PerfFileHeader)) {
        return oa::Status::ok();
    }

    PerfFileHeader hdr{};
    oa::memcpy(&hdr, raw.data(), sizeof(hdr));

    if (hdr.magic != kMagic or hdr.version != kVersion) {
        return oa::Status::ok();
    }

    oa::Usize expectedSize = sizeof(PerfFileHeader)
        + (static_cast<oa::Usize>(hdr.entryCount) * sizeof(oa::PerfRecord));
    if (raw.size() < expectedSize) {
        return oa::Status::ok();
    }

    records_.reserve(hdr.entryCount);
    const oa::U8* ptr = raw.data() + sizeof(PerfFileHeader);
    for (oa::U32 i = 0; i < hdr.entryCount; ++i) {
        oa::PerfRecord rec{};
        oa::memcpy(&rec, ptr, sizeof(rec));
        records_.pushBack(rec);
        ptr += sizeof(rec);
    }

    return oa::Status::ok();
}

oa::Status oa::PerfStore::append(const oa::PerfRecord& inRecord) {
    records_.pushBack(inRecord);
    return flush_();
}

const oa::PerfRecord* oa::PerfStore::findLatest(
    const char* inGpuName,
    const char* inMetricName
) const {
    const oa::PerfRecord* best = nullptr;
    for (const auto& rec : records_) {
        if (std::strncmp(rec.gpuName,    inGpuName,    64) == 0 and
            std::strncmp(rec.metricName, inMetricName, 64) == 0) {
            if (best == nullptr or rec.timestampNs > best->timestampNs) {
                best = &rec;
            }
        }
    }
    return best;
}

void oa::PerfStore::printComparison(
    const char* inMetricName,
    const oa::PerfRecord& inCurrent,
    const char* inGpuName
) const {
    const oa::PerfRecord* prev = findLatest(inGpuName, inMetricName);
    if (prev == nullptr or prev->sampleCount == 0) {
        printf("  (no previous run for %s)\n", inMetricName);
        return;
    }

    oa::F64 delta = 0.0;
    if (prev->mean > 0.0) {
        delta = (inCurrent.mean - prev->mean) / prev->mean * 100.0;
    }

    char  sign     = delta >= 0.0 ? '+' : '-';
    oa::F64 absDelta = delta < 0.0 ? -delta : delta;

    oa::Datetime dt = oa::Datetime::fromUnixSeconds(prev->timestampNs / 1'000'000'000LL);
    oa::String   ts = dt.format("%Y-%m-%dT%H:%M");

    const char* trend;
    if (absDelta < 1.0) {
        trend = "\xe2\x9c\x93"; // UTF-8 checkmark ✓
    } else if (delta < 0.0) {
        trend = "REGRESSION";
    } else {
        trend = "improved";
    }

    printf("  Previous run: %.3f  %c%.1f%%  %s  (%s)\n",
        prev->mean, sign, absDelta, trend, ts.cStr());
}

oa::Status oa::PerfStore::flush_() const {
    oa::Status mkdirStatus = oa::Filesystem::createDirectories(oa::Paths::var("perf"));
    if (not mkdirStatus.isOk()) {
        return mkdirStatus;
    }

    PerfFileHeader hdr{};
    hdr.magic      = kMagic;
    hdr.version    = kVersion;
    hdr.entryCount = static_cast<oa::U32>(records_.size());

    oa::Usize totalSize = sizeof(PerfFileHeader)
        + (records_.size() * sizeof(oa::PerfRecord));

    oa::Vec<oa::U8> buf;
    buf.resize(totalSize, 0U);

    oa::memcpy(buf.data(), &hdr, sizeof(hdr));
    oa::U8* ptr = buf.data() + sizeof(hdr);
    for (const auto& rec : records_) {
        oa::memcpy(ptr, &rec, sizeof(rec));
        ptr += sizeof(rec);
    }

    return oa::Filesystem::writeBinary(oa::Path(path_),
        oa::Span<const oa::U8>(buf.data(), buf.size()));
}
