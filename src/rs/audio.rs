//! Semantic planar FP32 audio, stateless DSP, codecs, and device sessions.
//!
//! Numerical algorithms and the WAV-F32 writer are OA ports. Symphonia owns
//! bounded WAV/PCM, FLAC, and MP3 codec work; CPAL owns device I/O.

mod capture;
mod clip;
mod codec;
mod encoder;
mod lowering;
mod player;
mod signal;
mod transform;

pub use capture::{AudioCapture, AudioCaptureChunk, AudioCaptureConfig};
pub use clip::{Audio, AudioChannelLayout};
pub use codec::{
	decode_file, decode_memory, encode_interleaved_wav_f32, encode_wav_f32, save_wav_f32,
};
pub use encoder::{AudioCodec, AudioEncodeProfile, AudioEncoder, EncodedAudioPacket};
pub use player::{AudioPlayer, AudioPlayerConfig};
pub use signal::{
	BiquadCoefficients, NormalizeAudioConfig, NormalizeAudioMode, ResampleConfig, amplitude_to_db,
	biquad, clip, fade, gain, mix, normalize, pre_emphasis, resample, reverb, saturate, sos_filter,
	to_mono, waveform_envelope,
};
pub use transform::{MelConfig, MfccConfig, StftConfig, StftWindow, mel_spectrogram, mfcc, stft};
