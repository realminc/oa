// datasetctl — pack / inspect / unpack Realm .oad (dataset archive) files
//
//   datasetctl pack -o out.oad --train train.txt [--val val.txt] [--test test.txt]
//   datasetctl info path.oad
//   datasetctl unpack path.oad -o outdir

#include <Oa/Core/Cli.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Types.h>
#include <Oa/Ml/Oad.h>

#include <cstdio>

static OaString FormatBytes(OaU64 InBytes) {
	char buf[64];
	if (InBytes >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(InBytes) / 1'000'000'000);
	} else if (InBytes >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(InBytes) / 1'000'000);
	} else if (InBytes >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(InBytes) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(InBytes));
	}
	return buf;
}

static int CmdInfo(const OaString& InPath) {
	OaOadFile f;
	if (!f.TryOpen(InPath)) {
		OA_CLI("Error: not a valid .oad v1 file: %s", InPath.c_str());
		return 1;
	}
	const auto& h = f.Header();
	OA_CLI_RAW("\n");
	OA_CLI("  File:     %s", InPath.c_str());
	auto sz = OaFilesystem::GetFileSize(OaPath(InPath));
	if (sz.IsOk()) {
		OA_CLI("  Size:     %s", FormatBytes(sz.GetValue()).c_str());
	}
	OA_CLI("  Format:   OAD v%u.%u", static_cast<unsigned>(h.VersionMajor), static_cast<unsigned>(h.VersionMinor));
	OA_CLI("  Train:    %s @ offset %llu", FormatBytes(h.TrainBytes).c_str(),
		static_cast<unsigned long long>(h.TrainOffset));
	OA_CLI("  Val:      %s @ offset %llu", FormatBytes(h.ValBytes).c_str(),
		static_cast<unsigned long long>(h.ValOffset));
	OA_CLI("  Test:     %s @ offset %llu", FormatBytes(h.TestBytes).c_str(),
		static_cast<unsigned long long>(h.TestOffset));
	OA_CLI_RAW("\n");
	return 0;
}

static int CmdPack(
	const OaString& OutPath,
	const OaString& TrainPath,
	const OaString& ValPath,
	const OaString& TestPath
) {
	if (!OaFilesystem::IsFile(OaPath(TrainPath))) {
		OA_CLI("Error: train file not found: %s", TrainPath.c_str());
		return 1;
	}
	auto trainR = OaFilesystem::ReadBinary(OaPath(TrainPath));
	if (!trainR.IsOk()) {
		OA_CLI("Error: read train: %s", TrainPath.c_str());
		return 1;
	}
	auto& train = trainR.GetValue();

	OaVec<OaU8> val;
	if (!ValPath.empty()) {
		if (!OaFilesystem::IsFile(OaPath(ValPath))) {
			OA_CLI("Error: val file not found: %s", ValPath.c_str());
			return 1;
		}
		auto valR = OaFilesystem::ReadBinary(OaPath(ValPath));
		if (!valR.IsOk()) {
			OA_CLI("Error: read val: %s", ValPath.c_str());
			return 1;
		}
		val = std::move(valR).GetValue();
	}

	OaVec<OaU8> test;
	if (!TestPath.empty()) {
		if (!OaFilesystem::IsFile(OaPath(TestPath))) {
			OA_CLI("Error: test file not found: %s", TestPath.c_str());
			return 1;
		}
		auto testR = OaFilesystem::ReadBinary(OaPath(TestPath));
		if (!testR.IsOk()) {
			OA_CLI("Error: read test: %s", TestPath.c_str());
			return 1;
		}
		test = std::move(testR).GetValue();
	}

	OaSpan<const OaU8> ts(train.Data(), train.Size());
	OaSpan<const OaU8> vs(val.Data(), val.Size());
	OaSpan<const OaU8> xs(test.Data(), test.Size());
	auto st = OaOadWriteFile(OaPath(OutPath), ts, vs, xs);
	if (st.IsError()) {
		OA_CLI("Error: %s", st.GetMessage().c_str());
		return 1;
	}
	OA_CLI("Wrote %s (%s train, %s val, %s test)", OutPath.c_str(), FormatBytes(train.Size()).c_str(),
		FormatBytes(val.Size()).c_str(), FormatBytes(test.Size()).c_str());
	return 0;
}

static int CmdUnpack(const OaString& InPath, const OaString& OutDir) {
	OaOadFile f;
	if (!f.TryOpen(InPath)) {
		OA_CLI("Error: not a valid .oad v1 file: %s", InPath.c_str());
		return 1;
	}
	(void)OaFilesystem::CreateDirectories(OaPath(OutDir));

	auto writeSplit = [&](const char* Name, OaSpan<const OaU8> Span) -> int {
		if (Span.empty()) return 0;
		OaPath p = OaPath(OutDir) / Name;
		auto wst = OaFilesystem::WriteBinary(p, Span);
		if (wst.IsError()) {
			OA_CLI("Error: write %s: %s", p.string().c_str(), wst.GetMessage().c_str());
			return 1;
		}
		return 0;
	};

	if (writeSplit("train.bin", f.TrainSpan())) return 1;
	if (writeSplit("val.bin", f.ValSpan())) return 1;
	if (writeSplit("test.bin", f.TestSpan())) return 1;

	OA_CLI("Unpacked to %s/ (train.bin, val.bin, test.bin as present)", OutDir.c_str());
	return 0;
}

struct DatasetctlConfig {
	OaString OutPath;
	OaString OutDir = ".";
	OaString TrainPath;
	OaString ValPath;
	OaString TestPath;
	OaString InputPath;
};

class DatasetctlCli : public OaCli<DatasetctlConfig> {
public:
	DatasetctlCli() : OaCli("datasetctl", "Realm dataset archive (.oad) tool") {
		SetEpilog(
			"Examples:\n"
			"  datasetctl pack -o corpus.oad --train train.txt\n"
			"  datasetctl pack -o corpus.oad --train train.txt --val val.txt --test test.txt\n"
			"  datasetctl info corpus.oad\n"
			"  datasetctl unpack corpus.oad -o ./extracted\n"
		);

		auto* pack = AddSubcommand("pack", "Pack train/val/test byte files into one .oad");
		pack->add_option("-o,--output", Cfg_.OutPath, "Output .oad path")->required();
		pack->add_option("--train", Cfg_.TrainPath, "Training corpus (required)")->required();
		pack->add_option("--val", Cfg_.ValPath, "Validation corpus (optional)");
		pack->add_option("--test", Cfg_.TestPath, "Test corpus (optional)");

		auto* info = AddSubcommand("info", "Show .oad header and split sizes");
		info->add_option("path", Cfg_.InputPath, "Path to .oad")->required();

		auto* unpack = AddSubcommand("unpack", "Write train.bin / val.bin / test.bin to a directory");
		unpack->add_option("path", Cfg_.InputPath, "Path to .oad")->required();
		unpack->add_option("-o,--out", Cfg_.OutDir, "Output directory")->required();

		RequireSubcommand(1, 1);
	}
};

int main(int argc, char** argv) {
	DatasetctlCli cli;
	if (!cli.Parse(argc, argv)) return 1;

	const auto& cfg = cli.GetConfig();
	auto cmd = cli.GetSubcommand();

	if (cmd == "pack")
		return CmdPack(cfg.OutPath, cfg.TrainPath, cfg.ValPath, cfg.TestPath);
	if (cmd == "info") return CmdInfo(cfg.InputPath);
	if (cmd == "unpack") return CmdUnpack(cfg.InputPath, cfg.OutDir);

	OA_CLI("Error: unknown command '%s'", cmd.c_str());
	return 1;
}
