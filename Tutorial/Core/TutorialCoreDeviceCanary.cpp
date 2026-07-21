// OA device-admission canary. The JSON report is suitable for direct capture
// into an oa.evidence bundle or a scheduler admission record.

#include <Oa/Runtime/Canary.h>
#include <Oa/Runtime/Engine.h>

#include <cstdlib>
#include <cstdio>
#include <utility>

int main() {
	OaEngineConfig config;
	config.AppName = "OaDeviceCanary";
	config.Precision = OaPrecision::FP32;
	config.NumericMode = OaNumericMode::Deterministic;
	if (const char* validation = std::getenv("OA_VK_VALIDATION");
		validation != nullptr and validation[0] == '1')
	{
		config.EnableValidation = true;
	}

	auto created = OaEngine::Create(config);
	if (not created.IsOk()) {
		std::fprintf(stderr, "device canary engine creation failed: %s\n",
			created.GetStatus().ToString().CStr());
		return 2;
	}
	auto engine = std::move(created).GetValue();
	OaDeviceCanaryReport report;
	const auto status = OaDeviceCanary::Run(*engine, report);
	const auto json = report.DebugReportJson();
	std::fwrite(json.Data(), 1, json.Size(), stdout);
	if (not status.IsOk()) {
		std::fprintf(stderr, "device canary failed: %s\n",
			status.ToString().CStr());
		return status.GetCode() == OaStatusCode::DataLoss ? 3 : 2;
	}
	return 0;
}
