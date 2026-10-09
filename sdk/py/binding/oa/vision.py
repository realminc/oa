"""Image and video interpretation operations.
"""

from ._native import (
	DetectionMetricsResult,
	NmsResult,
	SegmentationMetricsResult,
	vision_binary_mask_counts as binary_mask_counts,
	vision_box_iou as box_iou,
	vision_confusion_matrix as confusion_matrix,
	vision_evaluate as evaluate,
	vision_evaluate_segmentation as evaluate_segmentation,
	vision_nms as nms,
)

__all__ = [
	"DetectionMetricsResult",
	"NmsResult",
	"SegmentationMetricsResult",
	"binary_mask_counts",
	"box_iou",
	"confusion_matrix",
	"evaluate",
	"evaluate_segmentation",
	"nms",
]
