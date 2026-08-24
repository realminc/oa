#include <oa/oa.h>

int main() {
	oa::EngineConfig config;
	config.appName = "OaInstalledConsumer";

	const auto factory = &oa::Engine::create;
	return factory == nullptr;
}
