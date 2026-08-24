// datasetctl — pack / inspect / unpack Realm .oad (dataset archive) files
//
//   datasetctl pack -o out.oad --train train.txt [--val val.txt] [--test test.txt]
//   datasetctl info path.oad
//   datasetctl unpack path.oad -o outdir

#include <oa/core/cli.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>
#include <oa/core/types.h>
#include <data/datasetArchive.h>

#include <cstdio>

static oa::String formatBytes(oa::U64 inBytes) {
	char buf[64];
	if (inBytes >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(inBytes) / 1'000'000'000);
	} else if (inBytes >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(inBytes) / 1'000'000);
	} else if (inBytes >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(inBytes) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(inBytes));
	}
	return buf;
}

static int cmdInfo(const oa::String& inPath) {
	oa::DatasetArchive f;
	if (!f.tryOpen(inPath)) {
		OA_CLI("Error: not a valid .oad v1 file: %s", inPath.cStr());
		return 1;
	}
	const auto& h = f.header();
	OA_CLI_RAW("\n");
	OA_CLI("  file:     %s", inPath.cStr());
	auto sz = oa::Filesystem::getFileSize(oa::Path(inPath));
	if (sz.isOk()) {
		OA_CLI("  size:     %s", formatBytes(sz.getValue()).cStr());
	}
	OA_CLI("  format:   OAD v%u.%u", static_cast<unsigned>(h.versionMajor), static_cast<unsigned>(h.versionMinor));
	OA_CLI("  Train:    %s @ offset %llu", formatBytes(h.trainBytes).cStr(),
		static_cast<unsigned long long>(h.trainOffset));
	OA_CLI("  Val:      %s @ offset %llu", formatBytes(h.valBytes).cStr(),
		static_cast<unsigned long long>(h.valOffset));
	OA_CLI("  Test:     %s @ offset %llu", formatBytes(h.testBytes).cStr(),
		static_cast<unsigned long long>(h.testOffset));
	OA_CLI_RAW("\n");
	return 0;
}

static int cmdPack(
	const oa::String& outPath,
	const oa::String& trainPath,
	const oa::String& valPath,
	const oa::String& testPath
) {
	if (!oa::Filesystem::isFile(oa::Path(trainPath))) {
		OA_CLI("Error: train file not found: %s", trainPath.cStr());
		return 1;
	}
	auto trainR = oa::Filesystem::readBinary(oa::Path(trainPath));
	if (!trainR.isOk()) {
		OA_CLI("Error: read train: %s", trainPath.cStr());
		return 1;
	}
	auto& train = trainR.getValue();

	oa::Vec<oa::U8> val;
	if (!valPath.empty()) {
		if (!oa::Filesystem::isFile(oa::Path(valPath))) {
			OA_CLI("Error: val file not found: %s", valPath.cStr());
			return 1;
		}
		auto valR = oa::Filesystem::readBinary(oa::Path(valPath));
		if (!valR.isOk()) {
			OA_CLI("Error: read val: %s", valPath.cStr());
			return 1;
		}
		val = oa::move(valR).getValue();
	}

	oa::Vec<oa::U8> test;
	if (!testPath.empty()) {
		if (!oa::Filesystem::isFile(oa::Path(testPath))) {
			OA_CLI("Error: test file not found: %s", testPath.cStr());
			return 1;
		}
		auto testR = oa::Filesystem::readBinary(oa::Path(testPath));
		if (!testR.isOk()) {
			OA_CLI("Error: read test: %s", testPath.cStr());
			return 1;
		}
		test = oa::move(testR).getValue();
	}

	oa::Span<const oa::U8> ts(train.data(), train.size());
	oa::Span<const oa::U8> vs(val.data(), val.size());
	oa::Span<const oa::U8> xs(test.data(), test.size());
	auto st = oa::writeDatasetArchive(oa::Path(outPath), ts, vs, xs);
	if (st.isError()) {
		OA_CLI("Error: %s", st.getMessage().cStr());
		return 1;
	}
	OA_CLI("Wrote %s (%s train, %s val, %s test)", outPath.cStr(), formatBytes(train.size()).cStr(),
		formatBytes(val.size()).cStr(), formatBytes(test.size()).cStr());
	return 0;
}

static int cmdUnpack(const oa::String& inPath, const oa::String& outDir) {
	oa::DatasetArchive f;
	if (!f.tryOpen(inPath)) {
		OA_CLI("Error: not a valid .oad v1 file: %s", inPath.cStr());
		return 1;
	}
	(void)oa::Filesystem::createDirectories(oa::Path(outDir));

	auto writeSplit = [&](const char* name, oa::Span<const oa::U8> span) -> int {
		if (span.empty()) return 0;
		oa::Path p = oa::Path(outDir) / name;
		auto wst = oa::Filesystem::writeBinary(p, span);
		if (wst.isError()) {
			OA_CLI("Error: write %s: %s", p.string().cStr(), wst.getMessage().cStr());
			return 1;
		}
		return 0;
	};

	if (writeSplit("train.bin", f.trainSpan())) return 1;
	if (writeSplit("val.bin", f.valSpan())) return 1;
	if (writeSplit("test.bin", f.testSpan())) return 1;

	OA_CLI("Unpacked to %s/ (train.bin, val.bin, test.bin as present)", outDir.cStr());
	return 0;
}

struct DatasetctlConfig {
	oa::String outPath;
	oa::String outDir = ".";
	oa::String trainPath;
	oa::String valPath;
	oa::String testPath;
	oa::String inputPath;
};

class DatasetctlCli : public oa::Cli<DatasetctlConfig> {
public:
	DatasetctlCli() : oa::Cli<DatasetctlConfig>("datasetctl", "Realm dataset archive (.oad) tool") {
		setEpilog(
			"Examples:\n"
			"  datasetctl pack -o corpus.oad --train train.txt\n"
			"  datasetctl pack -o corpus.oad --train train.txt --val val.txt --test test.txt\n"
			"  datasetctl info corpus.oad\n"
			"  datasetctl unpack corpus.oad -o ./extracted\n"
		);

		auto* pack = addSubcommand("pack", "Pack train/val/test byte files into one .oad");
		pack->add_option("-o,--output", cfg_.outPath, "output .oad path")->required();
		pack->add_option("--train", cfg_.trainPath, "training corpus (required)")->required();
		pack->add_option("--val", cfg_.valPath, "Validation corpus (optional)");
		pack->add_option("--test", cfg_.testPath, "Test corpus (optional)");

		auto* info = addSubcommand("info", "show .oad header and split sizes");
		info->add_option("path", cfg_.inputPath, "Path to .oad")->required();

		auto* unpack = addSubcommand("unpack", "Write train.bin / val.bin / test.bin to a directory");
		unpack->add_option("path", cfg_.inputPath, "Path to .oad")->required();
		unpack->add_option("-o,--out", cfg_.outDir, "output directory")->required();

		requireSubcommand(1, 1);
	}
};

int main(int argc, char** argv) {
	DatasetctlCli cli;
	if (!cli.parse(argc, argv)) return 1;

	const auto& cfg = cli.getConfig();
	auto cmd = cli.getSubcommand();

	if (cmd == "pack")
		return cmdPack(cfg.outPath, cfg.trainPath, cfg.valPath, cfg.testPath);
	if (cmd == "info") return cmdInfo(cfg.inputPath);
	if (cmd == "unpack") return cmdUnpack(cfg.inputPath, cfg.outDir);

	OA_CLI("Error: unknown command '%s'", cmd.cStr());
	return 1;
}
