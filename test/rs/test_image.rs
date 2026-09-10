//! Image operation contract tests.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "image/test_geometric.rs"]
mod geometric;

#[path = "image/test_filter.rs"]
mod filter;

#[path = "image/test_pixel.rs"]
mod pixel;

#[path = "image/test_codec.rs"]
mod codec;
