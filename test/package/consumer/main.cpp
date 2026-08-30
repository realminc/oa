#include <oa/oa.h>

int main() {
	oa::EngineConfig config;
	config.appName = "OaInstalledConsumer";
	oa::Scene scene;
	scene.meshes.pushBack({
		oa::SceneMeshId{1U}, "cube", oa::FnMesh::createCube()});
	scene.nodes.pushBack({
		oa::SceneNodeId{1U}, "cube", {}, oa::SceneMeshId{1U},
		oa::vlm::Mat4::identity(), true});
	auto compiledScene = oa::FnScene::compile(scene);
	if (compiledScene.isError() or compiledScene->indices.size() != 36U) return 1;

	const auto factory = &oa::Engine::create;
	return factory == nullptr;
}
