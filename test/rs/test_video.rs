//! External contracts for the public video module.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "video/test_av1.rs"]
mod av1;
#[path = "video/test_capability.rs"]
mod capability;
#[path = "video/test_decoder.rs"]
mod decoder;
#[path = "video/test_demux.rs"]
mod demux;
#[path = "video/test_frame.rs"]
mod frame;
#[path = "video/test_h264.rs"]
mod h264;
#[path = "video/test_h265.rs"]
mod h265;
#[path = "video/test_mux.rs"]
mod mux;
#[path = "video/test_nal.rs"]
mod nal;
#[path = "video/test_player.rs"]
mod player;
