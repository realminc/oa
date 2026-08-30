#include <oa/render/fnScene.h>

#include <gtest/gtest.h>

namespace {

oa::MeshData triangleMesh() {
	oa::MeshData mesh;
	const oa::F32 inverseSqrtTwo = 0.7071067811865475F;
	mesh.vertices = {
		{{0.0F, 0.0F, 0.0F}, {inverseSqrtTwo, inverseSqrtTwo, 0.0F}, {}, {1.0F, 0.0F, 0.0F, 1.0F}},
		{{1.0F, 0.0F, 0.0F}, {inverseSqrtTwo, inverseSqrtTwo, 0.0F}, {}, {0.0F, 1.0F, 0.0F, 1.0F}},
		{{0.0F, 1.0F, 0.0F}, {inverseSqrtTwo, inverseSqrtTwo, 0.0F}, {}, {0.0F, 0.0F, 1.0F, 1.0F}},
	};
	mesh.indices = {0U, 1U, 2U};
	oa::FnMesh::computeBounds(mesh);
	return mesh;
}

oa::Scene baseScene() {
	oa::Scene scene;
	scene.meshes.pushBack({oa::SceneMeshId{10U}, "triangle", triangleMesh()});
	scene.nodes.pushBack({
		oa::SceneNodeId{20U},
		"triangle instance",
		{},
		oa::SceneMeshId{10U},
		oa::vlm::Mat4::identity(),
		true,
	});
	return scene;
}

} // namespace

TEST(Scene, ResolvesOutOfOrderHierarchyAndCompilesStableNodeOrder) {
	oa::Scene scene = baseScene();
	scene.nodes[0].parent = oa::SceneNodeId{30U};
	scene.nodes[0].localTransform = oa::vlm::translation(
		oa::vlm::Vec3{1.0F, 0.0F, 0.0F});
	scene.nodes.pushBack({
		oa::SceneNodeId{30U},
		"parent declared after child",
		{},
		{},
		oa::vlm::translation(oa::vlm::Vec3{2.0F, 3.0F, 4.0F}),
		true,
	});

	auto compiled = oa::FnScene::compile(scene);
	ASSERT_TRUE(compiled.isOk()) << compiled.getStatus().toString().cStr();
	ASSERT_EQ(compiled->vertices.size(), 3U);
	EXPECT_FLOAT_EQ(compiled->vertices[0].position.x, 3.0F);
	EXPECT_FLOAT_EQ(compiled->vertices[0].position.y, 3.0F);
	EXPECT_FLOAT_EQ(compiled->vertices[0].position.z, 4.0F);
	EXPECT_FLOAT_EQ(compiled->bounds.min.x, 3.0F);
	EXPECT_FLOAT_EQ(compiled->bounds.max.x, 4.0F);
}

TEST(Scene, UsesInverseTransposeForNonUniformNormalTransform) {
	oa::Scene scene = baseScene();
	scene.nodes[0].localTransform = oa::vlm::scaleMatrix(
		oa::vlm::Vec3{2.0F, 1.0F, 1.0F});

	auto compiled = oa::FnScene::compile(scene);
	ASSERT_TRUE(compiled.isOk()) << compiled.getStatus().toString().cStr();
	const oa::vlm::Vec3 normal = compiled->vertices[0].normal;
	EXPECT_NEAR(normal.x, 0.4472135955F, 1.0e-5F);
	EXPECT_NEAR(normal.y, 0.8944271910F, 1.0e-5F);
	EXPECT_NEAR(normal.z, 0.0F, 1.0e-5F);
}

TEST(Mesh, TransformUsesOneCheckedVlmAffineContract) {
	oa::MeshData mesh = triangleMesh();
	const oa::vlm::Mat4 transform = oa::vlm::composeTrs(
		{3.0F, 4.0F, 5.0F}, oa::vlm::Quat::identity(),
		{2.0F, 1.0F, 1.0F});
	ASSERT_TRUE(oa::FnMesh::transform(mesh, transform).isOk());
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		mesh.vertices[1].position, oa::vlm::Vec3{5.0F, 4.0F, 5.0F}));
	EXPECT_TRUE(oa::vlm::approximatelyEqual(
		mesh.vertices[0].normal,
		oa::vlm::normalize(oa::vlm::Vec3{0.5F, 1.0F, 0.0F})));
	EXPECT_TRUE(mesh.boundsDirty);
}

TEST(Mesh, TransformFailureLeavesMeshUnchanged) {
	oa::MeshData mesh = triangleMesh();
	const oa::MeshData original = mesh;
	const oa::Status status = oa::FnMesh::transform(
		mesh,
		oa::vlm::scaleMatrix(oa::vlm::Vec3{1.0F, 0.0F, 1.0F}));
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
	ASSERT_EQ(mesh.vertices.size(), original.vertices.size());
	for (oa::Usize index = 0U; index < mesh.vertices.size(); ++index) {
		EXPECT_EQ(mesh.vertices[index].position, original.vertices[index].position);
		EXPECT_EQ(mesh.vertices[index].normal, original.vertices[index].normal);
	}
	EXPECT_EQ(mesh.boundsDirty, original.boundsDirty);
}

TEST(Scene, RejectsDuplicateMissingAndCyclicIdentities) {
	{
		oa::Scene scene = baseScene();
		scene.nodes.pushBack(scene.nodes.front());
		EXPECT_EQ(
			oa::FnScene::validate(scene).getCode(),
			oa::StatusCode::InvalidArgument);
	}
	{
		oa::Scene scene = baseScene();
		scene.nodes[0].mesh = oa::SceneMeshId{999U};
		EXPECT_EQ(
			oa::FnScene::validate(scene).getCode(),
			oa::StatusCode::InvalidArgument);
	}
	{
		oa::Scene scene = baseScene();
		scene.nodes[0].parent = oa::SceneNodeId{30U};
		scene.nodes.pushBack({
			oa::SceneNodeId{30U}, "cycle", oa::SceneNodeId{20U}, {},
			oa::vlm::Mat4::identity(), true});
		EXPECT_EQ(
			oa::FnScene::validate(scene).getCode(),
			oa::StatusCode::InvalidArgument);
	}
}

TEST(Scene, EnforcesInheritedVisibilityAndConfiguredCapacity) {
	oa::Scene scene = baseScene();
	scene.nodes.pushBack({
		oa::SceneNodeId{30U}, "second", {}, oa::SceneMeshId{10U},
		oa::vlm::translation(oa::vlm::Vec3{4.0F, 0.0F, 0.0F}), true});

	oa::FnScene::CompileConfig limited;
	limited.maxVertexCount = 5U;
	limited.maxIndexCount = 6U;
	EXPECT_EQ(
		oa::FnScene::compile(scene, limited).getStatus().getCode(),
		oa::StatusCode::ResourceExhausted);

	scene.nodes[0].parent = oa::SceneNodeId{40U};
	scene.nodes.pushBack({
		oa::SceneNodeId{40U}, "hidden root", {}, {},
		oa::vlm::Mat4::identity(), false});
	auto compiled = oa::FnScene::compile(scene);
	ASSERT_TRUE(compiled.isOk()) << compiled.getStatus().toString().cStr();
	EXPECT_EQ(compiled->vertices.size(), 3U);
	EXPECT_FLOAT_EQ(compiled->vertices[0].position.x, 4.0F);
}

TEST(Scene, RejectsSingularTransformsAndZeroNormals) {
	{
		oa::Scene scene = baseScene();
		scene.nodes[0].localTransform = oa::vlm::scaleMatrix(
			oa::vlm::Vec3{1.0F, 0.0F, 1.0F});
		EXPECT_EQ(
			oa::FnScene::compile(scene).getStatus().getCode(),
			oa::StatusCode::InvalidArgument);
	}
	{
		oa::Scene scene = baseScene();
		scene.meshes[0].data.vertices[2].normal = {};
		EXPECT_EQ(
			oa::FnScene::validate(scene).getCode(),
			oa::StatusCode::InvalidArgument);
	}
}

TEST(Scene, FindsMutableNodeByStableIdentity) {
	oa::Scene scene = baseScene();
	oa::SceneNode* node = oa::FnScene::findNode(scene, oa::SceneNodeId{20U});
	ASSERT_NE(node, nullptr);
	node->visible = false;
	EXPECT_FALSE(scene.nodes[0].visible);
	EXPECT_EQ(oa::FnScene::findNode(scene, oa::SceneNodeId{999U}), nullptr);
}
