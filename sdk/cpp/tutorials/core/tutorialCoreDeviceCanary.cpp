// OA device-admission canary. The JSON report is suitable for direct capture
// into an oa.evidence bundle or a scheduler admission record.

#include <oa/runtime/canary.h>
#include <oa/runtime/engine.h>

#include <stdlib.h>
int main() {
	oa::EngineConfig config;
	config.appName = "oa::DeviceCanary";
	config.precision = oa::Precision::FP32;
	config.numericMode = oa::NumericMode::Deterministic;
	if (const char* validation = ::getenv("OA_VK_VALIDATION");
		validation != nullptr and validation[0] == '1')
	{
		config.enableValidation = true;
	}

	auto created = oa::Engine::create(config);
	if (not created.isOk()) {
		oa::print(oa::PrintStream::Error, "device canary engine creation failed: {}",
			created.getStatus().toString().cStr());
		return 2;
	}
	auto engine = oa::move(created).getValue();
	oa::DeviceCanaryReport report;
	const auto status = oa::DeviceCanary::run(*engine, report);
	const auto json = report.debugReportJson();
	const auto outputStatus = oa::write(json.view());
	if (not outputStatus.isOk()) {
		oa::print(oa::PrintStream::Error, "device canary output failed: {}",
			outputStatus.toString().cStr());
		return 2;
	}
	if (not status.isOk()) {
		oa::print(oa::PrintStream::Error, "device canary failed: {}",
			status.toString().cStr());
		return status.getCode() == oa::StatusCode::DataLoss ? 3 : 2;
	}
	return 0;
}
