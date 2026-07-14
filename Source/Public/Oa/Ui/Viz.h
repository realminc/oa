// OaUi — Tensor visualization widgets.
//
// All widgets dispatch compute shaders that READ from OaVkBuffer / bindless
// images directly.  No CPU readback.  No copies.
//
// Dispatch model:
//   OaHeatmap::Dispatch(InCmd, InBuffer, InCfg)
//     → tensor_heatmap.slang  ceil(cols/16) × ceil(rows/16) workgroups
//   OaChart::Dispatch(InCmd, InData, InCount, InCfg)
//     → chart_line.slang      ceil(width/256) workgroups
//   OaBranchViz::Dispatch(InCmd, InBuffer, InCfg)
//     → branch_viz.slang      ceil(width/256) workgroups

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Ui/Canvas.h>   // OaPixelRect
#include <vulkan/vulkan.h>

class OaComputeEngine;
struct OaVkBuffer;


// ─── OaColormapId ─────────────────────────────────────────────────────────────

enum class OaColormapId : OaU8 {
	Plasma   = 0,
	Viridis  = 1,
	Coolwarm = 2,
	Grays    = 3,
	Realm    = 4,  // 8-color Realm Design System palette
};


// ─── OaHeatmap ────────────────────────────────────────────────────────────────

struct OaHeatmapConfig {
	OaI32        Rows      = 0;
	OaI32        Cols      = 0;
	OaF32        VMin      = -1.0F;
	OaF32        VMax      =  1.0F;
	OaColormapId Colormap  = OaColormapId::Coolwarm;
	OaU32        DstIdx    = 0;   // bindless index of destination storage image
};

class OaHeatmap {
public:
	// Dispatch the tensor_heatmap compute shader.
	// InBuffer: float32 device buffer of at least Rows×Cols elements.
	static void Dispatch(
		VkCommandBuffer         InCmd,
		const OaVkBuffer&       InBuffer,
		const OaHeatmapConfig&  InCfg);
};


// ─── OaChart ──────────────────────────────────────────────────────────────────

enum class OaChartKind : OaU8 {
	Line      = 0,
	Histogram = 1,
	Scatter   = 2,
};

struct OaChartConfig {
	OaChartKind Kind      = OaChartKind::Line;
	OaF32       YMin      = 0.0F;
	OaF32       YMax      = 1.0F;
	bool        AutoScale = true;
	OaU32       Color     = 0x6366F1FFU;  // packed RGBA8 — Accent
	OaU32       DstIdx    = 0;
	OaI32       DstWidth  = 0;
	OaI32       DstHeight = 0;
};

class OaChart {
public:
	// CPU float array → chart (uploads to a transient mapped buffer each call).
	static void Dispatch(
		VkCommandBuffer        InCmd,
		OaComputeEngine&     InRt,
		const OaF32*           InData,
		OaI32                  InCount,
		const OaChartConfig&   InCfg);

	// Ring-buffer variant (reads InData[InOffset % InCount .. ] wrapping).
	static void DispatchRing(
		VkCommandBuffer        InCmd,
		OaComputeEngine&     InRt,
		const OaF32*           InData,
		OaI32                  InCount,
		OaI32                  InOffset,
		const OaChartConfig&   InCfg);
};


// ─── OaBranchViz ──────────────────────────────────────────────────────────────
// REALM-P specific: 4-branch cTEM state oscilloscope waveforms.

struct OaBranchVizConfig {
	OaI32 BranchCount = 4;
	OaI32 RingLen     = 512;   // samples per branch ring buffer
	OaU32 DstIdx      = 0;
	OaI32 DstWidth    = 0;
	OaI32 DstHeight   = 0;
};

class OaBranchViz {
public:
	// InBuffer: float32 ring of BranchCount × RingLen elements (branch-major).
	static void Dispatch(
		VkCommandBuffer          InCmd,
		const OaVkBuffer&        InBuffer,
		const OaBranchVizConfig& InCfg);
};


// ─── OaTokenGrid ──────────────────────────────────────────────────────────────
// Byte-sequence heatmap: each cell = one byte, colored by attribute buffer.

enum class OaTokenGridAttr : OaU8 {
	BranchWinner = 0,   // which ISG branch won
	RdgDepth     = 1,   // RDG iteration count (saturating to 7)
	Confidence   = 2,   // per-token cross-entropy loss (low=confident)
	AttentionSum = 3,   // sum of attention weights onto this token
};

struct OaTokenGridConfig {
	OaI32             SeqLen    = 0;
	OaTokenGridAttr   Attr      = OaTokenGridAttr::BranchWinner;
	OaF32             CellSize  = 12.0F;
	OaU32             DstIdx    = 0;
};

class OaTokenGrid {
public:
	// InTokenBuf: OaU8 sequence buffer (device).
	// InAttrBuf:  matching OaU8 attribute buffer (device).
	static void Dispatch(
		VkCommandBuffer          InCmd,
		const OaVkBuffer&        InTokenBuf,
		const OaVkBuffer&        InAttrBuf,
		const OaTokenGridConfig& InCfg);
};
