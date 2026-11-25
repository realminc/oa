//! Still-image codecs and stateless image transformations.
//!
//! The semantic [`Image`] value is implemented by this domain.
//! Operations belong here once they have an operation schema, an executable
//! lowering, and an independent correctness oracle.

mod augmentation;
mod codec;
#[path = "image/color.gen.rs"]
mod color;
#[path = "image/filter.gen.rs"]
mod filter;
#[path = "image/geometric.gen.rs"]
mod geometric;
#[allow(clippy::module_inception)]
mod image;
#[path = "image/image.gen.rs"]
mod methods;
#[path = "image/pixel.gen.rs"]
mod pixel;
#[path = "image/statistics.gen.rs"]
mod statistics;
mod transform;

pub use augmentation::{noise_like, random_crop, random_flip, random_rotate};
pub use image::{Image, ImageFormat, ImageLayout};

pub use codec::{
	ImageCodec, can_decode, can_encode, decode_file, decode_file_on, decode_memory, decode_memory_on,
	encode, save_file, save_rgba_file,
};
pub use color::{
	ColorConversion, ColorMap, NormalizationConfig, apply_color_map, convert_color, cvt_color,
	normalize, resize_normalize, segmentation_overlay,
};

pub use filter::{
	adaptive_threshold_gaussian, adaptive_threshold_mean, average_blur, bilateral_filter, canny,
	convolve_2d, dilate, erode, gaussian_blur, laplacian, median_blur, morphology_black_hat,
	morphology_close, morphology_gradient, morphology_open, morphology_top_hat, scharr,
	separable_convolve_2d, sharpen, sobel, unsharp_mask,
};

pub use geometric::{
	BorderMode, InterpolationMode, PolarMode, center_crop, crop, flip, pad, pyr_down, pyr_up, remap,
	resize, rotate, warp_affine, warp_perspective, warp_polar,
};
pub use pixel::{
	alpha_blend, brightness_contrast, channel_reorder, clamp, color_twist, composite, draw_circle,
	draw_line, draw_rectangle, erase, fill_polygon, flood_fill, gamma_contrast, gaussian_noise,
	grayscale, in_range, invert, posterize, salt_pepper_noise, solarize, threshold_binary,
	threshold_binary_inv, threshold_to_zero, threshold_to_zero_inv, threshold_truncate,
};
pub use statistics::{
	AutoThreshold, Connectivity, CornerScore, DistanceType, HistogramComparison, MatchTemplate,
	accumulate, accumulate_weighted, bf_match, calc_back_project, calc_hist, clahe, compare_hist,
	connected_components, corner_fast, corner_score, distance_transform, equalize_hist,
	find_contours, good_features_to_track, hough_lines, hough_lines_p, integral, match_template,
	moments, orb, psnr, ssim, threshold_auto,
};
pub use transform::{estimate_affine_2d, invert_affine, perspective_transform, rotation_matrix_2d};

#[path = "image/parity.gen.rs"]
mod parity;
