//! Interpretation of image and video values.
//!
//! Pixel transformations and still-image codecs belong to [`crate::image`];
//! video codecs and frame transformations belong to [`crate::video`]. Vision
//! is the direct Rust operation namespace for the donor's `oa::FnDetection`
//! surface; no `FnDetection` type or forwarding facade is introduced.
//! Operations are published here only when they add interpretation semantics
//! and have an operation schema, executable lowering, and correctness oracle.

mod camera;
mod detection;
#[path = "vision/detection.gen.rs"]
mod detection_ops;
mod geometry;
mod kalman;
mod screen_capture;
#[path = "vision/type.gen.rs"]
mod types;

pub use camera::CameraCapture;
pub use detection::{Detection, DetectionBuffer};
pub use detection_ops::{
	DetectionMetricsResult, NmsConfig, NmsResult, SegmentationMetricsResult, binary_mask_counts,
	box_iou, confusion_matrix, evaluate, evaluate_segmentation, nms,
};
pub use geometry::{
	CalibrateCameraResult, EpipolarResult, EstimationMethod, HomographyResult, PnpRansacResult,
	PnpResult, RecoverPoseResult, calibrate_camera, find_essential, find_fundamental,
	find_homography, project_points, recover_pose, rodrigues, solve_pnp, solve_pnp_ransac,
	triangulate_points, undistort_points, undistort_rectify_map,
};
pub use kalman::KalmanFilter;
pub use screen_capture::{ScreenCapture, ScreenCaptureCursor, ScreenCaptureTarget};
pub use types::{CameraCaptureConfig, ScreenCaptureConfig};
