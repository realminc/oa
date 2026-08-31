// Ui — immediate-API widget layer with GPU-retained rendering.
//
// call pattern (every frame):
//   ui.beginFrame(delta_ms);
//   ui.beginPanel("Train", {20, 20, 400, 600});
//     if (ui.button("run")) { ... }
//     ui.sliderF32("LR", &lr, 1e-5F, 1e-3F);
//     ui.plotLine("loss", loss_data, count);
//   ui.endPanel();
//   ui.endFrame();
//
// commands are rebuilt each frame while pipelines, atlases, and bounded upload
// rings remain persistent. resource reuse follows explicit completion events.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/ui/style.h>
#include <oa/ui/event.h>
#include <oa/ui/motion.h>
#include <oa/ui/canvas.h>
#include <oa/ui/text.h>
#include <oa/runtime/event.h>
#include <vulkan/vulkan.h>

namespace oa {
class DetectionBuffer;
class Engine;
class Matrix;
class Texture;
}
namespace oavk { class Buffer; }
namespace oa {

class GlyphBuffer;
class TextAtlas;
class ImagePlanes;


// ─── layout primitives ────────────────────────────────────────────────────────

enum class UiDirection : oa::U8 {
	Column = 0,
	Row    = 1,
};

enum class UiAlign : oa::U8 {
	Start   = 0,
	Center  = 1,
	End     = 2,
	Stretch = 3,
};

enum class UiTextDirection : oa::U8 {
	LeftToRight = 0,
	BottomToTop = 1,
};

enum class UiChevronDirection : oa::U8 {
	Previous = 0,
	Next = 1,
};

// Small GPU-vector glyphs used by icon-only controls. These are UI semantics,
// not an asset pack: every icon is rendered by the compositor and remains
// theme-, scale-, focus- and accessibility-aware.
enum class UiIcon : oa::U8 {
	Previous = 0,
	Next,
	Play,
	Pause,
	Menu,
	Minimize,
	Maximize,
	Restore,
	Close,
	Fullscreen,
	ExitFullscreen,
	Volume,
	Muted,
	Settings,
};

enum class UiIconButtonStyle : oa::U8 {
	Overlay = 0,
	Header,
	WindowControl,
};

enum class UiSizingKind : oa::U8 {
	Fill  = 0,  // expand to fill parent
	Fixed = 1,  // fixed pixel size
	Hug   = 2,  // shrink-wrap content
};

struct UiSizing {
	UiSizingKind kind  = UiSizingKind::Fill;
	oa::F32         value = 0.0F;  // pixel size when kind == Fixed

	[[nodiscard]] static constexpr UiSizing fill()          noexcept { return {.kind = UiSizingKind::Fill,  .value = 0.0F}; }
	[[nodiscard]] static constexpr UiSizing fixed(oa::F32 inPx) noexcept { return {.kind = UiSizingKind::Fixed, .value = inPx}; }
	[[nodiscard]] static constexpr UiSizing hug()           noexcept { return {.kind = UiSizingKind::Hug,   .value = 0.0F}; }
};

struct UiEdge {
	oa::F32 top    = 0.0F;
	oa::F32 right  = 0.0F;
	oa::F32 bottom = 0.0F;
	oa::F32 left   = 0.0F;

	constexpr UiEdge() = default;
	explicit constexpr UiEdge(oa::F32 inAll) : top(inAll), right(inAll), bottom(inAll), left(inAll) {}
	constexpr UiEdge(oa::F32 inV, oa::F32 inH) : top(inV), right(inH), bottom(inV), left(inH) {}
};

struct UiLayout {
	UiDirection direction = UiDirection::Column;
	UiAlign     align     = UiAlign::Stretch;
	UiAlign     justify   = UiAlign::Start;
	oa::F32        gap       = 4.0F;
	UiEdge      padding   = UiEdge{8.0F};
	UiSizing    width     = UiSizing::fill();
	UiSizing    height    = UiSizing::hug();
};

struct UiScrollConfig {
	// Positive pixel movement produced by one discrete wheel tick.
	oa::I32 wheelStep = 48;
	oa::I32 scrollbarWidth = 10;
	oa::I32 scrollbarGap = 4;
	bool showScrollbar = true;
};

struct UiScrollRegion {
	PixelRect viewport;
	// Pixel-snapped panel coordinates after applying the retained offset.
	// Manual rows begin at content.y + layout.padding.top.
	PixelRect content;
	oa::I32 offsetY = 0;
	oa::I32 maxOffsetY = 0;
};

struct UiVirtualRange {
	oa::I32 first = 0;
	oa::I32 onePastLast = 0;

	[[nodiscard]] constexpr bool empty() const noexcept {
		return first >= onePastLast;
	}
};

struct UiSplitConfig {
	// Row divides left/right; Column divides top/bottom.
	UiDirection direction = UiDirection::Row;
	oa::I32 handleSize = 8;
	oa::I32 minimumFirst = 120;
	oa::I32 minimumSecond = 120;
	// fraction of the distributable extent per unmodified arrow press.
	oa::F32 keyboardStep = 0.02F;
};

struct UiSplitRegion {
	PixelRect first;
	PixelRect handle;
	PixelRect second;
	bool changed = false;
};

struct UiTreeRowConfig {
	oa::I32 depth = 0;
	oa::I32 indent = 16;
	oa::I32 disclosureWidth = 18;
	bool hasChildren = false;
	bool open = false;
	bool selected = false;
	bool enabled = true;
};

struct UiTreeRowResult {
	PixelRect row;
	PixelRect disclosure;
	PixelRect label;
	bool activated = false;
	bool openChanged = false;
	bool open = false;
};

struct UiPropertyRowConfig {
	oa::F32 labelFraction = 0.38F;
	oa::I32 gap = 8;
	oa::I32 paddingX = 6;
	bool alternate = false;
};

struct UiPropertyRegion {
	PixelRect row;
	PixelRect label;
	PixelRect value;
};

struct UiTabItem {
	// Stable within the tab bar and independent of the visible label.
	oa::StringView id;
	oa::StringView label;
	bool dirty = false;
	bool closable = true;
	bool enabled = true;
};

struct UiTabBarState {
	// Both fields are caller-owned presentation state and can be serialized with
	// the downstream workspace. Item order remains caller-owned as well.
	oa::I32 selected = -1;
	oa::I32 firstVisible = 0;
};

struct UiTabBarConfig {
	oa::I32 minimumTabWidth = 88;
	oa::I32 maximumTabWidth = 220;
	oa::I32 overflowButtonWidth = 22;
	oa::I32 closeWidth = 20;
	bool reorderable = true;
};

struct UiTabBarResult {
	PixelRect bar;
	PixelRect tabs;
	oa::I32 firstVisible = 0;
	oa::I32 onePastLast = 0;
	oa::I32 activatedIndex = -1;
	oa::I32 closeRequestedIndex = -1;
	oa::I32 moveFromIndex = -1;
	oa::I32 moveToIndex = -1;
	bool selectionChanged = false;
};

struct UiPopupConfig {
	// Desired popup content box. Placement is resolved against the frame
	// viewport and flips above the anchor when there is not enough room below.
	oa::I32 width = 220;
	oa::I32 height = 240;
	oa::I32 gap = 4;
	UiEdge padding = UiEdge{4.0F};
};

struct UiDropdownConfig {
	// Long lists use the canonical scroll-panel route inside the popup.
	oa::I32 maxVisibleItems = 8;
	// Zero follows the laid-out control width.
	oa::I32 popupWidth = 0;
	oa::I32 popupGap = 4;
};

struct UiTooltipConfig {
	oa::F32 delayMs = 400.0F;
	oa::I32 maxWidth = 320;
	oa::I32 gap = 10;
	UiEdge padding = UiEdge{6.0F, 8.0F};
};


// ─── Accessibility export ───────────────────────────────────────────────

enum class UiAccessibilityRole : oa::U8 {
	Button,
	Checkbox,
	Slider,
	TextField,
	ComboBox,
	MenuItem,
	Tab,
	TreeItem,
	Splitter,
	Timeline,
};

enum class UiAccessibilityState : oa::U32 {
	None = 0U,
	Disabled = 1U << 0U,
	Focused = 1U << 1U,
	Checked = 1U << 2U,
	Selected = 1U << 3U,
	Expanded = 1U << 4U,
	Editable = 1U << 5U,
	HasPopup = 1U << 6U,
};

enum class UiAccessibilityAction : oa::U32 {
	None = 0U,
	Focus = 1U << 0U,
	Activate = 1U << 1U,
	Toggle = 1U << 2U,
	Increment = 1U << 3U,
	Decrement = 1U << 4U,
	SetValue = 1U << 5U,
	Close = 1U << 6U,
};

[[nodiscard]] constexpr UiAccessibilityState operator|(
	UiAccessibilityState inA,
	UiAccessibilityState inB) noexcept {
	return static_cast<UiAccessibilityState>(
		static_cast<oa::U32>(inA) | static_cast<oa::U32>(inB));
}

[[nodiscard]] constexpr UiAccessibilityAction operator|(
	UiAccessibilityAction inA,
	UiAccessibilityAction inB) noexcept {
	return static_cast<UiAccessibilityAction>(
		static_cast<oa::U32>(inA) | static_cast<oa::U32>(inB));
}

struct UiAccessibilityNode {
	// id is the same stable scoped identity used by focus and interaction.
	// Scope groups siblings without exposing a retained panel/document tree.
	oa::U32 id = 0U;
	oa::U32 scope = 0U;
	UiAccessibilityRole role = UiAccessibilityRole::Button;
	UiAccessibilityState state = UiAccessibilityState::None;
	UiAccessibilityAction actions = UiAccessibilityAction::None;
	PixelRect bounds;
	oa::String label;
	oa::String value;
	oa::F64 minimum = 0.0;
	oa::F64 maximum = 0.0;
	oa::F64 current = 0.0;
	bool hasNumericValue = false;
};


// ─── Widget config structs ────────────────────────────────────────────────────

struct UiPlotConfig {
	oa::Color color     = {0.388F, 0.400F, 0.945F, 1.0F};  // accent
	oa::F32    xMin      = 0.0F;
	oa::F32    xMax      = 1.0F;
	oa::F32    yMin      = 0.0F;
	oa::F32    yMax      = 1.0F;
	bool     autoScale = true;
	bool     showGrid  = true;
	bool     fill      = false;
	// Compute-line rasterization samples. The implemented set is 1/4/8;
	// unsupported values resolve to four samples rather than naming MSAA on a
	// storage-image path.
	oa::U32    antialiasSamples = 4U;
	oa::F32    lineWidth = 1.35F;
	// Higher-level plot figures paint one shared themed surface before replaying
	// multiple series; standalone widgets retain the historical default.
	bool     drawSurface = true;
};

struct UiHeatmapConfig {
	oa::I32    rows    = 0;
	oa::I32    cols    = 0;
	oa::F32    vMin    = -1.0F;
	oa::F32    vMax    =  1.0F;
	oa::U32    colormap = 0;  // 0=plasma 1=viridis 2=coolwarm 3=grays
	oa::U32    valueType = 0; // 0=Float32 1=UInt32 2=Int32
	oa::U32    offsetElements = 0;
	bool     showGrid = false;
};

struct UiTextConfig {
	FontId font = FontId::Sans;
	oa::F32 fontSize = 14.0F;
	oa::Color color = {0.961F, 0.961F, 0.961F, 1.0F};
	UiAlign horizontalAlign = UiAlign::Center;
	UiAlign verticalAlign = UiAlign::Center;
	UiTextDirection direction = UiTextDirection::LeftToRight;
};

struct UiGridConfig {
	// Absolute target-space origin and screen-pixel spacing of the 10-unit
	// tier. major and super-major lines therefore represent 100 and 1000
	// units with the defaults below.
	oa::vlm::Vec2 origin = {0.0F, 0.0F};
	oa::vlm::Vec2 minorSpacing = {10.0F, 10.0F};
	oa::U32 majorEvery = 10U;
	oa::U32 superMajorEvery = 100U;
	oa::F32 minorThickness = 0.65F;
	oa::F32 majorThickness = 0.90F;
	oa::F32 superMajorThickness = 1.0F;
	oa::F32 axisThickness = 1.25F;
	// Multiplies line alpha without affecting the optional background fill.
	oa::F32 opacity = 1.0F;
	// Optional circular guide falloff in absolute target pixels. Grid and axes
	// remain fully visible through the inner radius, then fade out at the outer
	// radius. The background fill always covers the complete rectangle.
	oa::vlm::Vec2 guideFadeCenter = {0.0F, 0.0F};
	oa::F32 guideFadeInnerRadius = 0.0F;
	oa::F32 guideFadeOuterRadius = 1.0F;
	// alpha zero selects the canonical colors derived from currentStyle().
	oa::Color backgroundTop = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::Color backgroundBottom = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::Color minorColor = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::Color majorColor = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::Color superMajorColor = {0.0F, 0.0F, 0.0F, 0.0F};
	oa::Color axisColor = {0.0F, 0.0F, 0.0F, 0.0F};
	bool fillBackground = true;
	bool drawGrid = true;
	bool drawAxes = true;
	bool radialGuideFade = false;
	// Stable spatial dithering prevents low-contrast gradients from exposing
	// the final 8-bit target's quantization bands.
	bool ditherBackground = false;
};

struct UiNodeCanvasGridConfig {
	oa::F32 minimumScreenSpacing = 8.0F;
	oa::F32 baseWorldStep = 10.0F;
	oa::U32 majorEvery = 10U;
	oa::U32 superMajorEvery = 100U;
	oa::F32 minorThickness = 0.65F;
	oa::F32 majorThickness = 0.90F;
	oa::F32 superMajorThickness = 1.0F;
	oa::F32 axisThickness = 1.25F;
	bool fillBackground = true;
	bool drawAxes = true;
};

// ─── Ui ──────────────────────────────────────────────────────────────────────

class Ui {
public:
	Ui();
	Ui(const Ui&)            = delete;
	Ui& operator=(const Ui&) = delete;
	Ui(Ui&&) noexcept;
	Ui& operator=(Ui&&) noexcept;
	~Ui();

	[[nodiscard]] oa::Status init(oa::Engine& inRt, const UiStyle& inStyle = {});
	[[nodiscard]] oa::Status setMotionSpeed(oa::UiMotionSpeed inSpeed);
	[[nodiscard]] oa::UiMotionSpeed motionSpeed() const noexcept;
	// Borrows the Viewer-owned atlas and allocates completion-tracked dynamic
	// glyph slots. Bind once before the first frame; the atlas must outlive Ui.
	[[nodiscard]] oa::Status bindTextAtlas(const TextAtlas& inAtlas);
	// Called once after init, before the first frame.
	// inComposeImageView: vkImageView (as void*) of the compose storage image (set=1, slot 0).
	[[nodiscard]] oa::Status initBlit(void* inComposeImageView);
	// Called after a compose image rebuild (resize) to refresh the image descriptor.
	void updateBlitImage(void* inComposeImageView);
	// Explicit renderer completion and release boundary. Waits every exact frame
	// completion accepted by markFrameSubmitted before releasing pipelines,
	// descriptor slots, and transient upload storage.
	[[nodiscard]] oa::Status close();

	// ── Per-frame ─────────────────────────────────────────────────────────────

	// The viewport is required only by top-layer popup/tooltip placement. Viewer
	// supplies its live compose extent. content scale converts the init style's
	// logical geometry into physical pixels; event coordinates and explicit
	// PixelRect values remain physical. Non-finite/non-positive scale is a
	// frame error.
	void beginFrame(
		oa::F32 inDeltaMs,
		PixelRect inViewport = {},
		oa::F32 inContentScale = 1.0F);
	[[nodiscard]] oa::F32 contentScale() const noexcept;
	// valid after widgets are submitted and until the next beginFrame. This is a
	// flat, caller-borrowed platform-adapter snapshot, not an OA-owned desktop or
	// document tree. bounds are ancestor-clipped physical pixels.
	[[nodiscard]] oa::Span<const UiAccessibilityNode>
	accessibilitySnapshot() const noexcept;

	// route a platform event. Returns true when an active/focused widget consumes it.
	bool routeEvent(const UiEvent& inEvent);
	// Platform bridge for enabling native committed-text/IME input only while a
	// field rendered in the current frame owns keyboard focus.
	[[nodiscard]] bool wantsTextInput() const noexcept;
	[[nodiscard]] PixelRect textInputRect() const noexcept;
	// Returns and clears a focused editor's pending copy/cut payload so the
	// platform layer can publish it without introducing SDL into Ui.
	[[nodiscard]] bool takeClipboardWrite(oa::String& outText);

	// Record all widget dispatch commands into inCmd.
	[[nodiscard]] oa::Status recordRender(
		VkCommandBuffer inCmd,
		oa::U32 inDstBindlessIdx);
	// Marks transient resources sampled by this frame. Plot buffers are recycled
	// only after the exact engine-owned graphics completion is reached.
	[[nodiscard]] oa::Status markFrameSubmitted(const oa::Event& inCompletion);

	void endFrame();

	// ── style stack (O(1), max depth 32) ─────────────────────────────────────

	void pushStyle(const UiStyle& inStyle);
	void popStyle();
	[[nodiscard]] const UiStyle& currentStyle() const noexcept;

	// ── layout containers ─────────────────────────────────────────────────────

	void beginPanel(oa::StringView inId, PixelRect inRect, const UiLayout& inLayout = {});
	void endPanel();

	// Scroll panels retain only interaction/offset state under their stable ID.
	// content coordinates are shifted and every nested draw/hit-test is clipped
	// to the viewport. inContentHeight includes any layout padding. The caller
	// owns content data; endScrollPanel must close the matching begin call.
	[[nodiscard]] UiScrollRegion beginScrollPanel(
		oa::StringView inId,
		PixelRect inViewport,
		oa::I32 inContentHeight,
		const UiLayout& inLayout = {},
		const UiScrollConfig& inConfig = {});
	void endScrollPanel();
	// Returns the visible half-open row range for the active scroll panel. rows
	// start after layout.padding.top and use inRowHeight + inRowGap stride.
	// Overscan is expressed in rows and is clamped to the item count.
	[[nodiscard]] UiVirtualRange virtualRows(
		oa::I32 inItemCount,
		oa::I32 inRowHeight,
		oa::I32 inRowGap = 0,
		oa::I32 inOverscanRows = 1) const;
	// Divides an explicit rectangle without beginning either child panel. The
	// caller owns and may persist inOutRatio; it is the first region's fraction
	// of the extent remaining after handleSize. Pointer drags retain capture
	// outside the handle. Focused row splits accept Left/Right, and focused
	// column splits accept Up/Down; shift/ctrl use the shared coarse/fine scale.
	[[nodiscard]] UiSplitRegion splitPane(
		oa::StringView inId,
		PixelRect inRect,
		oa::F32& inOutRatio,
		const UiSplitConfig& inConfig = {});
	// One row of a caller-flattened hierarchy. The caller owns open/selection
	// state and applies open when openChanged is returned. Pointer activation on
	// the disclosure toggles only; activation elsewhere selects. Focused tree
	// rows use Up/Down to select among prior rendered rows, Left/Right to close
	// or open, and Return/Space to activate. Explicit rectangles make the row
	// directly compatible with virtualRows.
	[[nodiscard]] UiTreeRowResult treeRow(
		oa::StringView inId,
		PixelRect inRect,
		oa::StringView inLabel,
		const UiTreeRowConfig& inConfig = {});
	// Draws a passive compact label/value row. Empty values leave the returned
	// Value rectangle available for a caller-owned editor control.
	[[nodiscard]] UiPropertyRegion propertyRow(
		oa::StringView inId,
		PixelRect inRect,
		oa::StringView inLabel,
		oa::StringView inValue = {},
		const UiPropertyRowConfig& inConfig = {});
	// Draws one caller-owned tab sequence. Selection and overflow position live
	// in inOutState; close and reorder are returned as requests so OA never owns
	// documents or mutates their order. Left/Right/Home/End navigate the focused
	// sequence and ctrl/Cmd+W requests closure of a closable focused tab.
	[[nodiscard]] UiTabBarResult tabBar(
		oa::StringView inId,
		PixelRect inRect,
		oa::Span<const UiTabItem> inItems,
		UiTabBarState& inOutState,
		const UiTabBarConfig& inConfig = {});

	// Explicit rows hug measured item widths and advance with layout.gap.
	// Column items otherwise stretch across the panel's padded inner width.
	void beginRow(oa::StringView inId = {});
	void endRow();

	void spacing(oa::F32 inPixels);
	// Draws a one-pixel horizontal rule in columns or a vertical rule in rows.
	void separator();

	// ── top-layer overlays ────────────────────────────────────────────────────

	// at most one interactive popup owns pointer/keyboard input. The no-anchor
	// overload anchors to the most recently submitted interactive item. The
	// explicit overload supports context menus and application-defined anchors.
	void openPopup(oa::StringView inId);
	void openPopup(oa::StringView inId, PixelRect inAnchor);
	void closePopup();
	[[nodiscard]] bool isPopupOpen(oa::StringView inId) const noexcept;
	// beginPopup/endPopup render through the same deferred UI compositor, but
	// their commands are submitted after every base-layer command. Popup calls
	// must use the same parent scope as openPopup.
	[[nodiscard]] bool beginPopup(
		oa::StringView inId,
		const UiPopupConfig& inConfig = {});
	void endPopup();
	// Menu items require an active popup and close it after activation.
	[[nodiscard]] bool menuItem(
		oa::StringView inLabel,
		bool inSelected = false,
		bool inEnabled = true);

	// ── Widgets ───────────────────────────────────────────────────────────────

	// labels identify controls within the active panel/row scope and must be
	// unique there. mouse activation fires on release-inside; Tab/shift+Tab
	// traverse focus and Return/Space activate the focused control.
	// Returns true on click or keyboard activation.
	[[nodiscard]] bool button(oa::StringView inLabel);
	// Explicit text-only control for fixed overlay geometry. inLabel is the
	// stable identity and accessibility label; inText is the rendered value.
	// The normal surface is transparent and receives the same hover, held and
	// keyboard-focus treatment as a header icon button. Padding is expressed in
	// physical pixels and keeps glyph alignment independent of the hit surface.
	[[nodiscard]] bool textButton(
		oa::StringView inLabel,
		PixelRect inRect,
		oa::StringView inText,
		const UiTextConfig& inTextConfig = {},
		bool inEnabled = true,
		const UiEdge& inTextPadding = {});
	// Explicit icon-only control. inLabel is the stable widget identity,
	// accessibility label and tooltip anchor; no text is painted in the button.
	[[nodiscard]] bool iconButton(
		oa::StringView inLabel,
		PixelRect inRect,
		UiIcon inIcon,
		bool inEnabled = true,
		UiIconButtonStyle inButtonStyle = UiIconButtonStyle::Overlay);
	// Explicit circular OSD control with a GPU-drawn previous/next chevron.
	// Compatibility convenience over iconButton.
	[[nodiscard]] bool chevronButton(
		oa::StringView inLabel,
		PixelRect inRect,
		UiChevronDirection inDirection,
		bool inEnabled = true);
	// Returns true when state changes.
	[[nodiscard]] bool checkbox(oa::StringView inLabel, bool& inOutValue);
	// Sliders clamp the referenced value into [min, max] and return true only
	// when it changes. Focused sliders accept arrow-key adjustment (1% of the
	// range, shift=10x, ctrl=0.1x); pointer drags remain captured outside bounds.
	[[nodiscard]] bool sliderF32(oa::StringView inLabel, oa::F32* inOutValue, oa::F32 inMin, oa::F32 inMax, const char* inFmt = "%.3F");
	[[nodiscard]] bool sliderI32(oa::StringView inLabel, oa::I32* inOutValue, oa::I32 inMin, oa::I32 inMax);
	// Single-line UTF-8 editor. Returns true only when inOutText changes. The
	// caller owns the value; the field retains bounded local edit history and IME
	// composition state. It supports single/double/triple-click caret/word/all
	// selection, shift and word navigation, scalar-safe deletion, clipboard,
	// ctrl/Cmd+Z redo variants, Home/End, horizontal scrolling, visible native
	// pre-edit and committed UTF-8 input. An external value replacement rebases
	// local undo history; document-level undo remains application policy.
	[[nodiscard]] bool inputText(oa::StringView inLabel, oa::String& inOutText);
	// Selection is caller-owned. Empty lists, invalid selection indices and
	// invalid popup metrics are reported through recordRender.
	[[nodiscard]] bool dropdown(
		oa::StringView inLabel,
		oa::Span<const oa::StringView> inItems,
		oa::I32& inOutSelected,
		const UiDropdownConfig& inConfig = {});
	// Attaches to the immediately preceding interactive item. tooltip geometry
	// is passive and shares the top-layer command route with popups.
	void tooltip(
		oa::StringView inText,
		const UiTooltipConfig& inConfig = {});

	// label does not soft-wrap; text wraps to the current panel's inner width.
	// Both honor explicit newlines, require an active panel and a bound text
	// atlas, and report misuse through recordRender instead of dropping text.
	void label(oa::StringView inText);
	#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 2, 3)))
	#endif
	void labelFmt(const char* inFmt, ...);
	void text(oa::StringView inText);
	// Positioned GPU text primitive for plots, overlays, and other fixed-layout
	// drawing. It does not participate in panel flow; inRect is an absolute
	// compose-space clip/alignment rectangle. Stretch alignment is invalid.
	void textAt(
		oa::StringView inText,
		PixelRect inRect,
		const UiTextConfig& inConfig = {});
	// Passive color preview participating in panel/row layout.
	void colorSwatch(oa::Color inColor, oa::vlm::Vec2 inSize = {16.0F, 16.0F});
	// Finite fractions are clamped to [0, 1]. An empty overlay renders a rounded
	// integer percentage; an explicit overlay is centered without soft wrapping.
	void progressBar(oa::F32 inFraction, oa::StringView inOverlay = {});
	// Explicit transport timeline. inOutFraction is normalized to [0, 1];
	// returns true when pointer scrubbing changes it.
	[[nodiscard]] bool timeline(
		oa::StringView inId,
		PixelRect inRect,
		oa::F32& inOutFraction,
		bool* outActive = nullptr);
	// Full-surface audio scrubber backed by a GPU [bins, 2] min/max envelope.
	[[nodiscard]] bool waveformTimeline(
		oa::StringView inId,
		PixelRect inRect,
		const oa::Matrix& inEnvelope,
		oa::F32& inOutFraction);
	// texture values are engine-checked before recording. Buffer-backed shared
	// leases are retained through the exact markFrameSubmitted completion.
	void image(
		const oa::Texture& inTexture,
		VkPipelineStageFlags2 inSourceStageMask =
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VkAccessFlags2 inSourceAccessMask =
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	// source stage/access describe the producer's last write made visible to
	// this sampled read. Defaults preserve compute storage-image producers;
	// graphics render targets must pass their color-attachment scopes.
	void imageVkRgba(
		void* inImage,
		void* inImageView,
		oa::I32 inW,
		oa::I32 inH,
		VkImageLayout inLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VkPipelineStageFlags2 inSourceStageMask =
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VkAccessFlags2 inSourceAccessMask =
			VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	// draw filled or outlined axis-aligned rectangles directly into the GPU
	// compose image. Rectangles must already be clipped to its extent.
	void rect(
		PixelRect inRect,
		oa::Color inColor,
		oa::F32 inCornerRadius = 0.0F);
	void rectOutline(
		PixelRect inRect,
		oa::Color inColor,
		oa::U32 inThickness = 1,
		oa::F32 inCornerRadius = 0.0F);
	// Replaces pixels outside a rounded rectangle with the supplied canvas
	// color. Unlike an outline, this clips already-composited GPU content.
	void maskRoundedRect(
		PixelRect inRect,
		oa::F32 inCornerRadius,
		oa::Color inOutsideColor);
	// Applies a premultiplied-alpha rounded mask to the complete compose
	// viewport. It is appended after overlay commands so client-side windows can
	// present a genuinely rounded surface rather than a square opaque buffer.
	void maskRoundedViewport(oa::F32 inCornerRadius);
	// draw an anti-aliased screen-space line in one GPU dispatch.
	void line(
		oa::vlm::Vec2 inBegin,
		oa::vlm::Vec2 inEnd,
		oa::Color inColor,
		oa::F32 inThickness = 2.0F);
	// draw one adaptive world-space canvas grid in one compute dispatch. Canvas
	// screen coordinates are local to inRect; colors follow the current style.
	void nodeCanvasGrid(
		const NodeCanvas& inCanvas,
		PixelRect inRect,
		const UiNodeCanvasGridConfig& inConfig = {});
	// One reusable viewport/chart/canvas grid. The default decimal hierarchy is
	// 10-unit minor, 100-unit major, 1000-unit super-major, plus black axes and
	// a subtle theme-derived vertical background gradient.
	void grid(PixelRect inRect, const UiGridConfig& inConfig = {});
	// draw normalized rectangle records from a bindless GPU buffer in one
	// dispatch. inDstRect maps source-image coordinates into the compose image.
	void rectOutlines(
		const oa::DetectionBuffer& inDetections,
		PixelRect inDstRect,
		PixelRect inClipRect,
		oa::Color inColor,
		oa::U32 inThickness = 1);
	// draw a source-anchored glyph batch from one persistent coverage atlas.
	void glyphs(
		const GlyphBuffer& inGlyphs,
		const TextAtlas& inAtlas,
		PixelRect inDstRect,
		PixelRect inClipRect);
	// Planar path: blitPlanar.slang handles per-channel dtype conversion + sRGB.
	// inPlanes must remain at a stable address through markFrameSubmitted; that
	// call attaches the exact consumer event used by its non-blocking RAII path.
	void imagePlanar(ImagePlanes& inPlanes, oa::I32 inDstX = 0, oa::I32 inDstY = 0);
	// draw one plane as grayscale without constructing a non-owning planes value.
	void imagePlane(ImagePlanes& inPlanes, oa::U32 inChannel,
		oa::I32 inDstX = 0, oa::I32 inDstY = 0);

	// ── Data visualization ────────────────────────────────────────────────────

	// CPU float array → line chart.
	void plotLine(oa::StringView inLabel, const oa::F32* inData, oa::I32 inCount, const UiPlotConfig& inCfg = {});
	// CPU explicit-X/Y metric arrays -> one GPU polyline dispatch. The shader
	// handles arbitrary segment direction without one dispatch per segment.
	void plotLineXY(oa::StringView inLabel, const oa::F32* inX, const oa::F32* inY,
		oa::I32 inCount, const UiPlotConfig& inCfg = {});
	// ring-buffer variant: reads inCount floats from inData[inOffset % inCount].
	void plotLineRing(oa::StringView inLabel, const oa::F32* inData, oa::I32 inCount, oa::I32 inOffset, const UiPlotConfig& inCfg = {});
	// GPU buffer → heatmap (zero-copy — reads directly from inBuffer on GPU).
	void heatmap(oa::StringView inLabel, const oavk::Buffer& inBuffer, const UiHeatmapConfig& inCfg);
	// Matrix convenience overload. rows/cols and value type are inferred when
	// omitted, and matrix byte offsets are honored for views.
	void heatmap(oa::StringView inLabel, const oa::Matrix& inMatrix, const UiHeatmapConfig& inCfg = {});
	// Host values → heatmap through the same bounded frame-safe upload ring as
	// PlotLine. Intended for compact metric tables and recorded oa::plot figures.
	void heatmap(oa::StringView inLabel, const oa::F32* inData, oa::I32 inRows,
		oa::I32 inCols, const UiHeatmapConfig& inCfg = {});

	// ── input state ───────────────────────────────────────────────────────────

	[[nodiscard]] const UiInputState& input() const noexcept { return input_; }

private:
	void abandon_() noexcept;
	void release_() noexcept;
	static oa::Status completeRetired_(void* inPayload);
	static void releaseRetired_(void* inPayload);

	struct Impl;
	oa::UniquePtr<Impl> impl_;
	UiInputState     input_;
};

}  // namespace oa
