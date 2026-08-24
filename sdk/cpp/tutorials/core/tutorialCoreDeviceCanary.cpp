// OA device-admission canary. The JSON report is suitable for direct capture
// into an oa.evidence bundle or a scheduler admission record.

#include <oa/runtime/canary.h>
#include <oa/runtime/engine.h>

#include <cstdlib>
#include <cstdio>
#include <utility>

int main() {
	oa::EngineConfig config;
	config.appName = "oa::DeviceCanary";
	config.precision = oa::Precision::FP32;
	config.numericMode = oa::NumericMode::Deterministic;
	if (const char* validation = std::getenv("OA_VK_VALIDATION");
		validation != nullptr and validation[0] == '1')
	{
		config.enableValidation = true;
	}

	auto created = oa::Engine::create(config);
	if (not created.isOk()) {
		std::fprintf(stderr, "device canary engine creation failed: %s\n",
			created.getStatus().toString().cStr());
		return 2;
	}
	auto engine = std::move(created).getValue();
	oa::DeviceCanaryReport report;
	const auto status = oa::DeviceCanary::run(*engine, report);
	const auto json = report.debugReportJson();
	std::fwrite(json.data(), 1, json.size(), stdout);
	if (not status.isOk()) {
		std::fprintf(stderr, "device canary failed: %s\n",
			status.toString().cStr());
		return status.getCode() == oa::StatusCode::DataLoss ? 3 : 2;
	}
	return 0;
}
