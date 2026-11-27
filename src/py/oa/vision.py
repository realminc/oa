"""Image and video interpretation operations.
"""

from ._gen.vision import *
from ._gen.vision import __all__ as _GEN_ALL
from ._native import (
	CalibrateCameraResult,
	DetectionMetricsResult,
	EpipolarResult,
	HomographyResult,
	NmsResult,
	PnpRansacResult,
	PnpResult,
	RecoverPoseResult,
	SegmentationMetricsResult,
	vision_calibrate_camera as calibrate_camera,
	vision_find_essential as find_essential,
	vision_find_fundamental as find_fundamental,
	vision_find_homography as find_homography,
	vision_project_points as project_points,
	vision_recover_pose as recover_pose,
	vision_rodrigues as rodrigues,
	vision_solve_pnp as solve_pnp,
	vision_solve_pnp_ransac as solve_pnp_ransac,
	vision_triangulate_points as triangulate_points,
	vision_undistort_points as undistort_points,
	vision_undistort_rectify_map as undistort_rectify_map,
)

__all__ = _GEN_ALL + [
	"CalibrateCameraResult",
	"DetectionMetricsResult",
	"EpipolarResult",
	"HomographyResult",
	"NmsResult",
	"PnpRansacResult",
	"PnpResult",
	"RecoverPoseResult",
	"SegmentationMetricsResult",
	"calibrate_camera",
	"find_essential",
	"find_fundamental",
	"find_homography",
	"project_points",
	"recover_pose",
	"rodrigues",
	"solve_pnp",
	"solve_pnp_ransac",
	"triangulate_points",
	"undistort_points",
	"undistort_rectify_map",
]
