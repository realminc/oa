//! Semantic planar FP32 audio values and synchronous codec boundaries.
//!
//! The value and WAV-F32 writer are direct semantic ports of OA. Decoding uses
//! the safe Rust Symphonia boundary instead of reproducing OA's miniaudio FFI;
//! only WAV/PCM, FLAC, and MP3 support is enabled.

mod codec;
mod value;

pub use codec::{
	decode_file, decode_memory, encode_interleaved_wav_f32, encode_wav_f32, save_wav_f32,
};
pub use value::{Audio, AudioChannelLayout};
