#include <oa/oa.h>

int main() {
	oa::EngineConfig config;
	config.appName = "OaInstalledConsumer";
	oa::Scene scene;
	scene.meshes.pushBack({
		oa::SceneMeshId{1U}, "cube", oa::FnMesh::createCube()});
	scene.nodes.pushBack({
		oa::SceneNodeId{1U}, "cube", {}, oa::SceneMeshId{1U},
		oa::Transform{}, true});
	auto compiledScene = oa::FnScene::compile(scene);
	if (compiledScene.isError() or compiledScene->indices.size() != 36U) return 1;

	const oa::Transform animationFrames[] = {
		oa::Transform{{0.0F, 0.0F, 0.0F}},
		oa::Transform{{2.0F, 0.0F, 0.0F}},
	};
	auto clip = oa::FnAnim::createClip(7U, 1U, 1.0F, animationFrames);
	if (clip.isError()) return 2;
	auto pose = clip->sample(0.5);
	if (pose.isError()
		or pose->getLocalTransform(0U).getPosition().x != 1.0F) {
		return 3;
	}
	const oa::Keyframe<oa::F32> opacityKeys[] = {
		{.time = oa::Duration{}, .value = 0.0F},
		{.time = oa::Duration::fromSeconds(1), .value = 1.0F},
	};
	auto opacityCurve = oa::FnAnim::createCurve(
		oa::AnimInterpolation::Linear, opacityKeys);
	if (opacityCurve.isError()) return 4;
	auto opacity = opacityCurve->sample(oa::Duration::fromMilliseconds(500));
	if (opacity.isError() or *opacity != 0.5F) return 5;
	auto bakedOpacity = oa::FnAnim::bake(
		*opacityCurve, oa::Duration{}, 2.0F, 3U);
	if (bakedOpacity.isError()
		or bakedOpacity->size() != 3U
		or (*bakedOpacity)[2] != 1.0F) {
		return 6;
	}

	const auto factory = &oa::Engine::create;
	return factory == nullptr;
}
