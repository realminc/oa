// OA CORE - Persistent Performance Record Store

#include <Oa/Core/PerfStore.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Time.h>
#include <cstring>
#include <cstdio>

static constexpr OaU32 kMagic   = 0x4F415046U;
static constexpr OaU32 kVersion = 1U;

struct OaPerfFileHeader {
    OaU32 Magic;
    OaU32 Version;
    OaU32 EntryCount;
    OaU32 Reserved[5];
};
static_assert(sizeof(OaPerfFileHeader) == 32, "header must be 32 bytes");

OaStatus OaPerfStore::Load(const char* InPath) {
    if (InPath != nullptr and InPath[0] != '\0') {
        Path_ = OaString(OaStringView(InPath));
    } else {
        Path_ = OaString((OaFileIo::GetVarDir("perf") / "OaPerf.dat").String());
    }

    Records_.Clear();

    if (not OaFileIo::Exists(OaPath(Path_))) {
        return OaStatus::Ok();
    }

    auto rawResult = OaFileIo::ReadBinary(OaPath(Path_));
    if (not rawResult.IsOk()) {
        return rawResult.GetStatus();
    }
    const auto& raw = rawResult.GetValue();

    if (raw.Size() < sizeof(OaPerfFileHeader)) {
        return OaStatus::Ok();
    }

    OaPerfFileHeader hdr{};
    OaMemcpy(&hdr, raw.Data(), sizeof(hdr));

    if (hdr.Magic != kMagic or hdr.Version != kVersion) {
        return OaStatus::Ok();
    }

    OaUsize expectedSize = sizeof(OaPerfFileHeader)
        + (static_cast<OaUsize>(hdr.EntryCount) * sizeof(OaPerfRecord));
    if (raw.Size() < expectedSize) {
        return OaStatus::Ok();
    }

    Records_.Reserve(hdr.EntryCount);
    const OaU8* ptr = raw.Data() + sizeof(OaPerfFileHeader);
    for (OaU32 i = 0; i < hdr.EntryCount; ++i) {
        OaPerfRecord rec{};
        OaMemcpy(&rec, ptr, sizeof(rec));
        Records_.PushBack(rec);
        ptr += sizeof(rec);
    }

    return OaStatus::Ok();
}

OaStatus OaPerfStore::Append(const OaPerfRecord& InRecord) {
    Records_.PushBack(InRecord);
    return Flush();
}

const OaPerfRecord* OaPerfStore::FindLatest(
    const char* InGpuName,
    const char* InMetricName
) const {
    const OaPerfRecord* best = nullptr;
    for (const auto& rec : Records_) {
        if (std::strncmp(rec.GpuName,    InGpuName,    64) == 0 and
            std::strncmp(rec.MetricName, InMetricName, 64) == 0) {
            if (best == nullptr or rec.TimestampNs > best->TimestampNs) {
                best = &rec;
            }
        }
    }
    return best;
}

void OaPerfStore::PrintComparison(
    const char* InMetricName,
    const OaPerfRecord& InCurrent,
    const char* InGpuName
) const {
    const OaPerfRecord* prev = FindLatest(InGpuName, InMetricName);
    if (prev == nullptr or prev->SampleCount == 0) {
        printf("  (no previous run for %s)\n", InMetricName);
        return;
    }

    OaF64 delta = 0.0;
    if (prev->Mean > 0.0) {
        delta = (InCurrent.Mean - prev->Mean) / prev->Mean * 100.0;
    }

    char  sign     = delta >= 0.0 ? '+' : '-';
    OaF64 absDelta = delta < 0.0 ? -delta : delta;

    OaDatetime dt = OaDatetime::FromUnixSeconds(prev->TimestampNs / 1'000'000'000LL);
    OaString   ts = dt.Format("%Y-%m-%dT%H:%M");

    const char* trend;
    if (absDelta < 1.0) {
        trend = "\xe2\x9c\x93"; // UTF-8 checkmark ✓
    } else if (delta < 0.0) {
        trend = "REGRESSION";
    } else {
        trend = "improved";
    }

    printf("  Previous run: %.3f  %c%.1f%%  %s  (%s)\n",
        prev->Mean, sign, absDelta, trend, ts.CStr());
}

OaStatus OaPerfStore::Flush() const {
    OaStatus mkdirStatus = OaFileIo::CreateDirectories(OaFileIo::GetVarDir("perf"));
    if (not mkdirStatus.IsOk()) {
        return mkdirStatus;
    }

    OaPerfFileHeader hdr{};
    hdr.Magic      = kMagic;
    hdr.Version    = kVersion;
    hdr.EntryCount = static_cast<OaU32>(Records_.Size());

    OaUsize totalSize = sizeof(OaPerfFileHeader)
        + (Records_.Size() * sizeof(OaPerfRecord));

    OaVec<OaU8> buf;
    buf.Resize(totalSize, 0U);

    OaMemcpy(buf.Data(), &hdr, sizeof(hdr));
    OaU8* ptr = buf.Data() + sizeof(hdr);
    for (const auto& rec : Records_) {
        OaMemcpy(ptr, &rec, sizeof(rec));
        ptr += sizeof(rec);
    }

    return OaFileIo::WriteBinary(OaPath(Path_),
        OaSpan<const OaU8>(buf.Data(), buf.Size()));
}
