//! Coordination of independently owned, timed audio and video tracks.
//!
//! Audio and video codecs, values, and device sessions retain their domain
//! owners. Media sessions own cross-track transport and clock policy.

mod clock;
mod player;

pub use clock::MediaTimestamp;
pub use player::{MediaPlayer, MediaPlayerConfig};
