#include <Oa/Ui/Node.h>

OaNodeGraph::OaNodeGraph(OaNodeGraph&&) noexcept = default;
OaNodeGraph& OaNodeGraph::operator=(OaNodeGraph&&) noexcept = default;
OaNodeGraph::~OaNodeGraph() { Destroy(); }

void OaNodeGraph::Init(const OaNodeGraphConfig& /*InConfig*/) {}
void OaNodeGraph::Destroy() {}

OaUiNodeId OaNodeGraph::AddNode(OaUiNode /*InNode*/) { return 0; }
void OaNodeGraph::RemoveNode(OaUiNodeId /*InId*/) {}

bool OaNodeGraph::Connect(OaUiPinId /*InFrom*/, OaUiPinId /*InTo*/) { return false; }
void OaNodeGraph::Disconnect(OaUiPinId /*InFrom*/, OaUiPinId /*InTo*/) {}

OaUiNode* OaNodeGraph::FindNode(OaUiNodeId /*InId*/) noexcept { return nullptr; }
const OaUiNode* OaNodeGraph::FindNode(OaUiNodeId /*InId*/) const noexcept { return nullptr; }

OaSpan<const OaUiNode> OaNodeGraph::Nodes() const noexcept { return {}; }
OaSpan<const OaUiNodeEdge> OaNodeGraph::Edges() const noexcept { return {}; }

OaStatus OaNodeGraph::Compile(OaEngine& /*InRt*/, OaUiComputeGraph& /*OutGraph*/) {
	return OaStatus::Ok();
}

void OaNodeGraph::Tick(OaF32 /*InDeltaMs*/) noexcept {}

void OaNodeGraph::RecordRender(VkCommandBuffer /*InCmd*/, OaU32 /*InDstBindlessIdx*/,
	const OaNodeCanvas& /*InCanvas*/) {}

bool OaNodeGraph::HandleEvent(const OaUiEvent& /*InEvent*/, OaNodeCanvas& /*InOutCanvas*/) {
	return false;
}

OaStatus OaNodeGraph::SaveJson(OaStringView /*InPath*/) const { return OaStatus::Ok(); }
OaStatus OaNodeGraph::LoadJson(OaStringView /*InPath*/) { return OaStatus::Ok(); }
