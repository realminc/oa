// OaUi — Node graph editor.
//
// Every OaUiNode IS a real OaComputeGraph dispatch node.  Editing the graph
// via the canvas edits the actual compute DAG — there is no separate "visual
// representation" that compiles down to a graph.
//
// OaNodeGraph::Compile(InRt, OutGraph) walks the node topology, resolves
// buffer handles from connected pins, and calls OutGraph.Add(...) for each
// node in topological order.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Ui/Canvas.h>   // VlmVec2
#include <vulkan/vulkan.h>

class OaComputeEngine;
class OaVkBuffer;
class OaUiComputeGraph;   // forward — defined in Runtime/Graph.h


// ─── IDs ──────────────────────────────────────────────────────────────────────

using OaUiNodeId = OaU32;
using OaUiPinId  = OaU64;  // packed: (node_id << 16) | (pin_index << 1) | is_output


// ─── OaNodeStatus ─────────────────────────────────────────────────────────────

enum class OaNodeStatus : OaU8 {
	Uncompiled = 0,
	Compiling  = 1,
	Ready      = 2,
	Error      = 3,
};


// ─── OaUiPin ────────────────────────────────────────────────────────────────

struct OaTensorMeta {
	OaVec<OaI32>  Shape;
	OaU8          Dtype = 0;   // OaScalarType
};

struct OaUiPin {
	OaUiPinId  Id;
	OaString     Name;
	OaVkBuffer*  Buffer    = nullptr;  // live buffer (filled at compile time)
	OaTensorMeta Meta;
	bool         IsOutput  = false;
	bool         Connected = false;
};


// ─── OaUiNode ───────────────────────────────────────────────────────────────

// Category controls title bar color (Realm Design System palette).
enum class OaNodeCategory : OaU8 {
	Data       = 0,   // #6366f1 indigo
	Norm       = 1,   // #22d3ee cyan
	Linear     = 2,   // #38bdf8 sky
	Activation = 3,   // #30d158 green
	Loss       = 4,   // #f59e0b amber
	State      = 5,   // #a855f7 purple
	Control    = 6,   // #ec4899 pink
	Optim      = 7,   // #ff453a red
	Custom     = 8,   // user-supplied color
};

struct OaUiNode {
	OaUiNodeId          Id;
	OaString              ShaderName;
	VlmVec2                Position;      // world space
	VlmVec2                Size;          // computed from pin count
	OaVec<OaUiPin>      Inputs;
	OaVec<OaUiPin>      Outputs;
	OaVec<OaU8>           PushData;      // editable push-constant bytes
	OaNodeStatus          Status    = OaNodeStatus::Uncompiled;
	OaNodeCategory        Category  = OaNodeCategory::Custom;
	OaU32                 TitleColor = 0x6366F1FFU;  // packed RGBA8
};


// ─── OaUiNodeEdge ───────────────────────────────────────────────────────────────

struct OaUiNodeEdge {
	OaUiPinId From;           // source output pin
	OaUiPinId To;             // destination input pin
	OaF32       FlowPhase     = 0.0F;    // animated flow (increments per frame)
	OaF32       ThroughputMBps = 0.0F;  // measured at last execution
};


// ─── OaNodeGraph ──────────────────────────────────────────────────────────────

struct OaNodeGraphConfig {
	bool AutoLayoutOnLoad = true;
};

class OaNodeGraph {
public:
	OaNodeGraph() = default;
	OaNodeGraph(const OaNodeGraph&)            = delete;
	OaNodeGraph& operator=(const OaNodeGraph&) = delete;
	OaNodeGraph(OaNodeGraph&&) noexcept;
	OaNodeGraph& operator=(OaNodeGraph&&) noexcept;
	~OaNodeGraph();

	void Init(const OaNodeGraphConfig& InConfig = {});
	void Destroy();

	// ── Graph mutation ────────────────────────────────────────────────────────

	[[nodiscard]] OaUiNodeId AddNode(OaUiNode InNode);
	void RemoveNode(OaUiNodeId InId);

	// Connect output pin → input pin.  Returns false if types mismatch.
	[[nodiscard]] bool Connect(OaUiPinId InFrom, OaUiPinId InTo);
	void Disconnect(OaUiPinId InFrom, OaUiPinId InTo);

	// ── Query ─────────────────────────────────────────────────────────────────

	[[nodiscard]] OaUiNode*       FindNode(OaUiNodeId InId) noexcept;
	[[nodiscard]] const OaUiNode* FindNode(OaUiNodeId InId) const noexcept;

	[[nodiscard]] OaSpan<const OaUiNode> Nodes() const noexcept;
	[[nodiscard]] OaSpan<const OaUiNodeEdge> Edges() const noexcept;

	// ── Compile: produce the OaComputeGraph from the node topology ────────────
	// Calls OutGraph.Reset(), topological sort, then OutGraph.Add() per node.
	[[nodiscard]] OaStatus Compile(OaComputeEngine& InRt, OaUiComputeGraph& OutGraph);

	// ── Render ────────────────────────────────────────────────────────────────

	// Advance flow phase animations.
	void Tick(OaF32 InDeltaMs) noexcept;

	// Record bezier_edge + sdf_rect node body dispatch commands.
	void RecordRender(VkCommandBuffer InCmd, OaU32 InDstBindlessIdx, const OaNodeCanvas& InCanvas);

	// ── Canvas interaction (called from OaUi::RouteEvent) ─────────────────────

	// Returns true if the event was consumed.
	[[nodiscard]] bool HandleEvent(const struct OaUiEvent& InEvent, OaNodeCanvas& InOutCanvas);

	// ── Serialization ─────────────────────────────────────────────────────────

	[[nodiscard]] OaStatus SaveJson(OaStringView InPath) const;
	[[nodiscard]] OaStatus LoadJson(OaStringView InPath);

private:
	struct Impl;
	OaUniquePtr<Impl> Impl_;
};
