//! External contracts for the public core module.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "core/test_api.rs"]
mod api;
#[path = "core/test_buffer_access.rs"]
mod buffer_access;
#[path = "core/test_callback.rs"]
mod callback;
#[path = "core/test_cli.rs"]
mod cli;
#[path = "core/test_config.rs"]
mod config;
#[path = "core/test_constant.rs"]
mod constant;
#[path = "core/test_env_flag.rs"]
mod env_flag;
#[path = "core/test_filesystem.rs"]
mod filesystem;
#[path = "core/test_image.rs"]
mod image;
#[path = "core/test_log_metrics.rs"]
mod log_metrics;
#[path = "core/test_mapped_file.rs"]
mod mapped_file;
#[path = "core/test_memory.rs"]
mod memory;
#[path = "core/test_perf_stat.rs"]
mod perf_stat;
#[path = "core/test_time.rs"]
mod time;
#[path = "core/test_validation.rs"]
mod validation;
#[path = "core/test_vlm.rs"]
mod vlm;
