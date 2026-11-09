//! oa - GPU-first architecture with explicit Vulkan
//!
//! Experimental Rust port / successor prototype for OA.
//!
//! The target API is organized as a language-like semantic surface:
//! - Semantic value types are re-exported at the root.
//! - Engine and, once admitted, Device and Event are runtime types.
//! - Stateless operations live in matrix, image, audio, video, vision, render,
//!   ml, and crypto modules.
//!
//! Foundational contracts live under `core`; common types are also re-exported
//! from the crate root. Runtime machinery lives under `runtime`.

pub mod runtime;

#[path = "../../sdk/rs/mod.rs"]
pub mod sdk;

pub mod anim;
pub mod audio;
pub mod core;
pub mod crypto;
pub mod data;
pub mod image;
pub mod input;
pub mod matrix;
pub mod media;
pub mod ml;
pub mod net;
pub mod plot;
pub mod render;
pub mod ui;
pub mod video;
pub mod viewer;
pub mod vision;

pub use core::vlm;

// Re-export the common public vocabulary.
pub use anim::{AnimClip, AnimCurve, Joint, Keyframe, SkelPose};
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
pub use crypto::pqc::{Keypair, PublicKey, SecretKey, Signature};
pub use crypto::{Hash, Hasher, MerkleProof, MerkleTree, SecureBuffer, Shake128, Shake256};
pub use data::{
	Batch, DataFrame, DataFrameView, DataLoader, DataLoaderConfig, Dataset, DsOhlcv, OhlcvView,
	OhlcvWindowConfig, OhlcvWindows, Sample, Series, SeriesView, SeriesViewMut, SplitResult, Subset,
	UnixTimeUnit,
};
pub use image::{Image, ImageFormat, ImageLayout};
pub use matrix::Matrix;
pub use media::MediaPlayer;
pub use net::{
	MCP_LATEST_PROTOCOL_VERSION, McpArguments, McpCacheScope, McpServer, McpServerConfig,
	McpTextResource, McpTool, McpToolResult, TcpFramed, TcpListener, TcpStream,
};
pub use render::{
	Aabb, AlphaMode, Camera, CameraProjection, FlatColorMaterial, Material, MaterialId,
	SceneMaterial, SurfaceMaterial, Texture, TextureHandle, UnlitMaterial,
};
pub use video::{
	VideoDecoder, VideoDemuxer, VideoEncoder, VideoFrame, VideoMuxer, VideoPlayer, VideoRecorder,
};

pub use input::{
	INPUT_MOD_ALT, INPUT_MOD_CTRL, INPUT_MOD_NONE, INPUT_MOD_SHIFT, INPUT_MOD_SUPER, InputEvent,
	InputEventKind, InputModifiers, InputState, Key, MotionSpeed, PinchPhase, ScrollGesture,
};
pub use runtime::{
	CapturedResourceDesc, ClockCalibration, DeviceHardwareInfo, DeviceInfo, DeviceKind,
	DeviceSelection, DeviceSoftwareInfo, DeviceSummary, Engine, EngineBuilder, EngineRequirements,
	Event, ExecutionBindingDesc, ExecutionBindingRole, ExecutionPlan, ExecutionPlanDiagnostics,
	FusionStats, HostClockDomain, LogComponent, LogLevel, LogOptions, Presenter, SdlWindow,
	SemanticAccessMode, SemanticAliasDesc, SemanticAutogradDesc, SemanticGraph,
	SemanticLoweringAnalysis, SemanticOpDesc, SemanticOpId, SemanticStorageBinding,
	SemanticValueAccess, SemanticValueDesc, SemanticValueId, Timer, TimerDomain,
};
pub use ui::{
	FontId, GlyphInfo, NodeCanvas, NodeCanvasGrid, NodeCanvasHit, NodeCanvasHitItem, NodeCanvasState,
	PixelRect, PositionedGlyph, TextAtlas, TextLayout, TextLayoutConfig, Ui, UiAlign, UiDirection,
	UiEdge, UiLayout, UiSizing, UiStyle, UiTabBarResult, UiTabBarState, UiTabItem, UiTreeRowConfig,
	UiTreeRowResult, WorldAabb,
};
pub use viewer::{
	TrainingPreviewFrame, TrainingViewerConfig, TrainingViewerSource, Viewer, ViewerAudioView,
	ViewerCanvasBackground, ViewerConfig, ViewerLiveCapabilities, ViewerLiveSource, ViewerMode,
};
