//! oa - GPU-first architecture with explicit Vulkan
//!
//! Experimental Rust port / successor prototype for OA.
//!
//! The target API is organized as a language-like semantic surface:
//! - Semantic value types are re-exported at the root.
//! - Engine and, once admitted, Device and Event are runtime types.
//! - Stateless operations live in matrix, image, audio, video, vision, render,
//!   ml, and cryptography modules.
//!
//! Foundational contracts live under `core`; common types are also re-exported
//! from the crate root. Runtime machinery lives under `runtime`.

pub mod runtime;

#[path = "../../sdk/rs/mod.rs"]
pub mod sdk;

pub mod audio;
pub mod core;
pub mod cryptography;
pub mod data;
pub mod image;
pub mod matrix;
pub mod media;
pub mod ml;
pub mod network;
pub mod plot;
pub mod render;
pub mod ui;
pub mod video;
pub mod viewer;
pub mod vision;

pub use core::vlm;

// Re-export the common public vocabulary.
pub use audio::{Audio, AudioCapture, AudioChannelLayout, AudioEncoder, AudioPlayer};
pub use core::transform::Transform;
pub use core::{
	BufferAccess, COMPACT_BANNER, Callback, CallbackSet, CheckpointConfig, Cli, Color, DType,
	Datetime, EnvFlag, Error, ErrorKind, Filesystem, IteratorContext, LogConfig, LogMetrics,
	MappedFile, NumericMode, OpAttribute, OpAttributeKind, OpAttributeSpec, OpControlFlow,
	OpDTypeRule, OpDifferentiation, OpEffect, OpLowering, OpShapeRule, OpValueKind,
	OperationContract, Path, PerfStat, REALM_BANNER, Result, ScopedTimer, Stopwatch, Timestamp,
	VIEWPORT_TITLE, Validation, ValidationSeverity, apply_numeric_mode, brand_viewport,
};
pub use cryptography::pqc::{Keypair, PublicKey, SecretKey, Signature};
pub use cryptography::{Hash, Hasher, MerkleProof, MerkleTree, SecureBuffer, Shake128, Shake256};
pub use data::{Batch, DataLoader, DataLoaderConfig, Dataset, Sample, SplitResult, Subset};
pub use image::{Image, ImageFormat, ImageLayout};
pub use matrix::Matrix;
pub use media::MediaPlayer;
pub use network::{
	MCP_LATEST_PROTOCOL_VERSION, McpArguments, McpCacheScope, McpServer, McpServerConfig,
	McpTextResource, McpTool, McpToolResult, TcpFramed, TcpListener, TcpStream,
};
pub use render::{
	Aabb, AlphaMode, Camera, CameraProjection, FlatColorMaterial, Material, MaterialId,
	SceneMaterial, StandardSurfaceMaterial, Texture, TextureHandle, UnlitMaterial,
};
pub use video::{VideoDecoder, VideoDemuxer, VideoFrame, VideoMuxer, VideoPlayer};

pub use runtime::{
	CapturedResourceDesc, DeviceHardwareInfo, DeviceInfo, DeviceKind, DeviceSelection,
	DeviceSoftwareInfo, DeviceSummary, Engine, EngineBuilder, EngineRequirements, Event,
	ExecutionPlan, ExecutionPlanDiagnostics, LogComponent, LogLevel, LogOptions, Presenter,
	SdlWindow, SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticStorageBinding,
	SemanticValueAccess, SemanticValueDesc, SemanticValueId,
};
pub use ui::{
	FontId, GlyphInfo, NodeCanvas, NodeCanvasState, PixelRect, PositionedGlyph, TextAtlas,
	TextLayout, TextLayoutConfig, UI_MOD_ALT, UI_MOD_CTRL, UI_MOD_NONE, UI_MOD_SHIFT, UI_MOD_SUPER,
	Ui, UiAlign, UiDirection, UiEdge, UiEvent, UiEventKind, UiInputState, UiKey, UiLayout,
	UiModifiers, UiPinchPhase, UiScrollGesture, UiSizing, UiStyle, UiTabBarResult, UiTabBarState,
	UiTabItem, UiTreeRowConfig, UiTreeRowResult,
};
pub use viewer::{
	Viewer, ViewerAudioView, ViewerCanvasBackground, ViewerConfig, ViewerLiveCapabilities,
	ViewerLiveSource, ViewerMode,
};
