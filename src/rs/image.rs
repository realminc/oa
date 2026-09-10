//! Still-image codecs and stateless image transformations.
//!
//! The semantic [`crate::Image`] value is implemented by [`crate::core`].
//! Operations belong here once they have an operation schema, an executable
//! lowering, and an independent correctness oracle.

mod codec;
mod color;
mod filter;
mod geometric;
mod pixel;

pub use codec::{
	ImageCodec, can_decode, can_encode, decode_file, decode_memory, encode, save_file,
	save_rgba_file,
};
pub use color::{
	NormalizationParams, convert_color, normalize, resize_normalize, segmentation_overlay,
};

pub use filter::{
	adaptive_threshold_gaussian, adaptive_threshold_mean, average_blur, bilateral_filter,
	convolve_2d, dilate, erode, gaussian_blur, laplacian, median_blur, morphology_black_hat,
	morphology_close, morphology_gradient, morphology_open, morphology_top_hat, scharr,
	separable_convolve_2d, sharpen, sobel, unsharp_mask,
};

pub use geometric::{
	BorderMode, InterpolationMode, center_crop, crop, flip, pad, remap, resize, rotate,
	warp_affine, warp_perspective,
};
pub use pixel::{
	alpha_blend, brightness_contrast, channel_reorder, clamp, color_twist, composite, erase,
	gamma_contrast, gaussian_noise, grayscale, in_range, invert, posterize, salt_pepper_noise,
	solarize, threshold_binary, threshold_binary_inv, threshold_to_zero, threshold_to_zero_inv,
	threshold_truncate,
};
