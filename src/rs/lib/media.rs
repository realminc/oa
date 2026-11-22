//! Coordination of independently owned, timed audio and video tracks.
//!
//! Audio and video codecs, values, and device sessions retain their domain
//! owners. Media sessions own cross-track transport and clock policy.

mod clock;
mod player;

pub use clock::MediaTimestamp;
pub use player::MediaPlayer;
pub use types::MediaPlayerConfig;

#[path = "media/parity.gen.rs"]
mod parity;
#[path = "media/type.gen.rs"]
mod types;
