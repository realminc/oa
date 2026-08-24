#include <oa/crypto/keccak.h>
#include <oa/network/satelliteSession.h>
#include <oa/runtime/engine.h>

#include "nlpCoordinator.h"
#include "nlpStep.h"

#include <charconv>
#include <cstdio>
#include <cstring>

namespace {

oa::Status satelliteMac(
	oa::Span<const oa::Byte> inKey,
	oa::Span<const oa::Byte> inData,
	oa::StringView inCustom,
	oa::Array<oa::Byte, 32>& outMac)
{
	return oa::kmac256(
		inKey.data(), inKey.size(), inData.data(), inData.size(),
		reinterpret_cast<const oa::Byte*>(inCustom.data()), inCustom.size(),
		outMac.data(), outMac.size());
}

struct Endpoint {
	oa::String host;
	oa::U16 port = 0;
};

void usage() {
	std::fprintf(stderr,
		"usage:\n"
		"  oa-satellite serve --listen 127.0.0.1:9100 --auth-key-hex <64 hex> [--once]\n"
		"  oa-satellite probe 127.0.0.1:9100 --auth-key-hex <64 hex>\n"
		"  oa-satellite benchmark --workload nlp-byte-transformer-step-v1 "
		"--mode standalone --checkpoint <path>\n"
		"  oa-satellite benchmark --peer 127.0.0.1:9100 --auth-key-hex <64 hex> "
		"--workload nlp-byte-transformer-step-v1 --mode split-batch "
		"--checkpoint <path>\n");
}

oa::Result<Endpoint> parseEndpoint(oa::StringView inText) {
	const auto separator = inText.stdView().rfind(':');
	if (separator == std::string_view::npos or separator == 0U
		or separator + 1U >= inText.size())
	{
		return oa::Status::invalidArgument("endpoint must be host:port");
	}
	Endpoint endpoint;
	endpoint.host = oa::String(inText.data(), separator);
	unsigned port = 0;
	const char* begin = inText.data() + separator + 1U;
	const char* end = inText.data() + inText.size();
	const auto parsed = std::from_chars(begin, end, port);
	if (parsed.ec != std::errc{} or parsed.ptr != end or port == 0U or port > 65535U) {
		return oa::Status::invalidArgument("endpoint port is invalid");
	}
	endpoint.port = static_cast<oa::U16>(port);
	return endpoint;
}

oa::I32 hexNibble(char inChar) {
	if (inChar >= '0' and inChar <= '9') return inChar - '0';
	if (inChar >= 'a' and inChar <= 'f') return 10 + inChar - 'a';
	if (inChar >= 'A' and inChar <= 'F') return 10 + inChar - 'A';
	return -1;
}

oa::Result<oa::SatelliteSecret> parseSecret(oa::StringView inHex) {
	if (inHex.size() != 64U) {
		return oa::Status::invalidArgument("authentication key must contain 64 hex digits");
	}
	oa::Array<oa::Byte, 32> bytes{};
	for (oa::Usize i = 0; i < bytes.size(); ++i) {
		const oa::I32 high = hexNibble(inHex[i * 2U]);
		const oa::I32 low = hexNibble(inHex[i * 2U + 1U]);
		if (high < 0 or low < 0) {
			return oa::Status::invalidArgument("authentication key is not hexadecimal");
		}
		bytes[i] = static_cast<oa::Byte>((high << 4U) | low);
	}
	auto secret = oa::SatelliteSecret::fromBytes(bytes);
	volatile oa::Byte* clear = bytes.data();
	for (oa::Usize i = 0; i < bytes.size(); ++i) clear[i] = 0U;
	return secret;
}

const char* findOption(int inArgc, char** inArgv, const char* inName) {
	for (int i = 2; i + 1 < inArgc; ++i) {
		if (std::strcmp(inArgv[i], inName) == 0) return inArgv[i + 1];
	}
	return nullptr;
}

bool hasFlag(int inArgc, char** inArgv, const char* inName) {
	for (int i = 2; i < inArgc; ++i) {
		if (std::strcmp(inArgv[i], inName) == 0) return true;
	}
	return false;
}

int serve(int inArgc, char** inArgv) {
	const char* endpointText = findOption(inArgc, inArgv, "--listen");
	const char* keyText = findOption(inArgc, inArgv, "--auth-key-hex");
	if (endpointText == nullptr or keyText == nullptr) {
		usage();
		return 2;
	}
	auto endpoint = parseEndpoint(endpointText);
	if (endpoint.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n", endpoint.getStatus().toString().cStr());
		return 2;
	}
	if (endpoint->host != "127.0.0.1") {
		std::fprintf(stderr,
			"oa-satellite: non-loopback serving is disabled until an encrypted pairing channel ships\n");
		return 2;
	}
	auto secret = parseSecret(keyText);
	if (secret.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n", secret.getStatus().toString().cStr());
		return 2;
	}
	auto listener = oa::TcpListener::bind(endpoint->host, endpoint->port, 8);
	if (listener.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n", listener.getStatus().toString().cStr());
		return 1;
	}
	oa::EngineConfig engineConfig;
	engineConfig.appName = "oa-satellite";
	engineConfig.presentationMode = oa::PresentationMode::None;
	engineConfig.precision = oa::Precision::FP32;
	engineConfig.numericMode = oa::NumericMode::Stable;
	auto engineResult = oa::Engine::create(engineConfig);
	if (engineResult.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	auto engine = oa::move(*engineResult);
	oa::SatelliteSessionConfig config(oa::move(*secret), satelliteMac);
	config.deviceName = oa::String(engine->deviceName());
	config.namedOperations.pushBack(oa::String(oa::SatelliteNlpStepOperation));
	config.namedOperations.pushBack(oa::String(oa::SatelliteNlpGradientOperation));
	config.namedWork = oa::satelliteExecuteNlpWork;
	oa::SatelliteServerSession server(*engine, oa::move(config));
	const bool once = hasFlag(inArgc, inArgv, "--once");
	std::printf("oa-satellite: listening on %s:%u\n",
		endpoint->host.cStr(), static_cast<unsigned>(listener->port()));
	std::fflush(stdout);
	int exitCode = 0;
	do {
		auto accepted = listener->accept();
		if (accepted.isError()) {
			std::fprintf(stderr, "oa-satellite: %s\n",
				accepted.getStatus().toString().cStr());
			exitCode = 1;
			break;
		}
		const auto status = server.serve(oa::move(*accepted));
		if (status.isError() and once) {
			std::fprintf(stderr, "oa-satellite: %s\n", status.toString().cStr());
			exitCode = 1;
			break;
		}
	} while (not once);
	const auto close = engine->close();
	if (close.isError()) {
		std::fprintf(stderr, "oa-satellite: engine close failed: %s\n",
			close.toString().cStr());
		return 1;
	}
	return exitCode;
}

int probe(int inArgc, char** inArgv) {
	if (inArgc < 3) {
		usage();
		return 2;
	}
	const char* keyText = findOption(inArgc, inArgv, "--auth-key-hex");
	if (keyText == nullptr) {
		usage();
		return 2;
	}
	auto endpoint = parseEndpoint(inArgv[2]);
	auto secret = parseSecret(keyText);
	if (endpoint.isError() or secret.isError()) {
		const auto& status = endpoint.isError() ? endpoint.getStatus() : secret.getStatus();
		std::fprintf(stderr, "oa-satellite: %s\n", status.toString().cStr());
		return 2;
	}
	oa::SatelliteSessionConfig config(oa::move(*secret), satelliteMac);
	auto connected = oa::SatelliteClientSession::connect(
		endpoint->host, endpoint->port, oa::move(config));
	if (connected.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n",
			connected.getStatus().toString().cStr());
		return 1;
	}
	auto client = oa::move(*connected);
	std::printf(
		"protocol=%u epoch=%llu device=%s max_payload=%u max_resident=%llu "
		"max_objects=%u max_inflight=%u\n",
		static_cast<unsigned>(oa::SatelliteProtocol::kVersion),
		static_cast<unsigned long long>(client.probe().sessionEpoch),
		client.probe().deviceName.cStr(),
		static_cast<unsigned>(client.probe().limits.maxPayloadBytes),
		static_cast<unsigned long long>(client.probe().limits.maxResidentBytes),
		static_cast<unsigned>(client.probe().limits.maxObjects),
		static_cast<unsigned>(client.probe().limits.maxInflight));
	const auto close = client.close();
	if (close.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n", close.toString().cStr());
		return 1;
	}
	return 0;
}

void printBenchmarkReport(
	const oa::SatelliteNlpCoordinatorReport& inReport,
	const char* inWorkload,
	const char* inMode,
	const char* inRemoteDevice)
{
	const oa::F64 wallPerStep = inReport.completeWallMs
		/ static_cast<oa::F64>(inReport.completedSteps);
	std::printf(
		"workload=%s mode=%s split=%s steps=%u optimizer_step=%llu "
		"initial_loss=%.6f final_loss=%.6f accuracy=%.4f "
		"wall_ms=%.3f wall_ms_per_step=%.3f local_kernel_selections=%llu "
		"local_kernel_fallbacks=%llu remote_kernel_selections=%llu "
		"remote_kernel_fallbacks=%llu checkpoint_round_trip=%s "
		"generation_quality=%s remote_device=%s\n",
		inWorkload, inMode,
		std::strcmp(inMode, "split-batch") == 0 ? "32/32" : "64/0",
		static_cast<unsigned>(inReport.completedSteps),
		static_cast<unsigned long long>(inReport.optimizerStep),
		static_cast<double>(inReport.initialLoss),
		static_cast<double>(inReport.finalLoss),
		static_cast<double>(inReport.finalAccuracy), inReport.completeWallMs,
		wallPerStep,
		static_cast<unsigned long long>(inReport.localKernelSelections),
		static_cast<unsigned long long>(inReport.localKernelFallbacks),
		static_cast<unsigned long long>(inReport.remoteKernelSelections),
		static_cast<unsigned long long>(inReport.remoteKernelFallbacks),
		inReport.checkpointRoundTrip ? "PASS" : "FAIL",
		inReport.generationQualityPassed ? "PASS" : "FAIL", inRemoteDevice);
	std::printf("generated=%.*s\n",
		static_cast<int>(inReport.generated.size()), inReport.generated.data());
}

int benchmark(int inArgc, char** inArgv) {
	const char* endpointText = findOption(inArgc, inArgv, "--peer");
	const char* keyText = findOption(inArgc, inArgv, "--auth-key-hex");
	const char* workload = findOption(inArgc, inArgv, "--workload");
	const char* mode = findOption(inArgc, inArgv, "--mode");
	const char* checkpoint = findOption(inArgc, inArgv, "--checkpoint");
	if (workload == nullptr or mode == nullptr or checkpoint == nullptr
		or std::strcmp(workload, oa::SatelliteNlpStepOperation.data()) != 0
		or (std::strcmp(mode, "standalone") != 0
			and std::strcmp(mode, "split-batch") != 0)
		or checkpoint[0] == '\0')
	{
		usage();
		return 2;
	}
	oa::EngineConfig engineConfig;
	engineConfig.appName = "oa-satellite-benchmark";
	engineConfig.presentationMode = oa::PresentationMode::None;
	engineConfig.precision = oa::Precision::FP32;
	engineConfig.numericMode = oa::NumericMode::Stable;
	auto engineResult = oa::Engine::create(engineConfig);
	if (engineResult.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n",
			engineResult.getStatus().toString().cStr());
		return 1;
	}
	auto engine = oa::move(*engineResult);
	oa::SatelliteNlpCoordinatorConfig coordinatorConfig;
	coordinatorConfig.checkpointPath = checkpoint;
	if (std::strcmp(mode, "standalone") == 0) {
		auto report = oa::satelliteRunStandaloneNlp(*engine, coordinatorConfig);
		if (report.isError()) {
			std::fprintf(stderr, "oa-satellite: %s\n",
				report.getStatus().toString().cStr());
			(void)engine->close();
			return 1;
		}
		printBenchmarkReport(*report, workload, mode, "none");
		const auto engineClose = engine->close();
		if (engineClose.isError()) {
			std::fprintf(stderr, "oa-satellite: close failed: %s\n",
				engineClose.toString().cStr());
			return 1;
		}
		return 0;
	}
	if (endpointText == nullptr or keyText == nullptr) {
		usage();
		(void)engine->close();
		return 2;
	}
	auto endpoint = parseEndpoint(endpointText);
	auto secret = parseSecret(keyText);
	if (endpoint.isError() or secret.isError()) {
		const auto& status = endpoint.isError() ? endpoint.getStatus() : secret.getStatus();
		std::fprintf(stderr, "oa-satellite: %s\n", status.toString().cStr());
		(void)engine->close();
		return 2;
	}
	oa::SatelliteSessionConfig sessionConfig(oa::move(*secret), satelliteMac);
	sessionConfig.deviceName = oa::String(engine->deviceName());
	sessionConfig.ioTimeoutMs = 10000U;
	auto connected = oa::SatelliteClientSession::connect(
		endpoint->host, endpoint->port, oa::move(sessionConfig));
	if (connected.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n",
			connected.getStatus().toString().cStr());
		(void)engine->close();
		return 1;
	}
	auto client = oa::move(*connected);
	coordinatorConfig.overlapRemote = true;
	auto report = oa::satelliteRunSplitBatchNlp(
		*engine, client, coordinatorConfig);
	if (report.isError()) {
		std::fprintf(stderr, "oa-satellite: %s\n",
			report.getStatus().toString().cStr());
		(void)client.close();
		(void)engine->close();
		return 1;
	}
	printBenchmarkReport(
		*report, workload, mode, client.probe().deviceName.cStr());
	const auto close = client.close();
	const auto engineClose = engine->close();
	if (close.isError() or engineClose.isError()) {
		const auto& status = close.isError() ? close : engineClose;
		std::fprintf(stderr, "oa-satellite: close failed: %s\n",
			status.toString().cStr());
		return 1;
	}
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		usage();
		return 2;
	}
	if (std::strcmp(argv[1], "serve") == 0) return serve(argc, argv);
	if (std::strcmp(argv[1], "probe") == 0) return probe(argc, argv);
	if (std::strcmp(argv[1], "benchmark") == 0) return benchmark(argc, argv);
	usage();
	return 2;
}
