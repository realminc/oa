//! Semantic planar FP32 audio, stateless DSP, codecs, and device sessions.
//!
//! Numerical algorithms and PCM/WAV encoding are OA ports. Private host
//! adapters decode audio, and CPAL owns device I/O.

#[allow(clippy::module_inception)]
mod audio;
mod capture;
mod codec;
mod decoder;
mod demux;
mod device;
mod encoder;
#[path = "audio/audio.gen.rs"]
mod methods;
mod player;
#[path = "audio/signal.gen.rs"]
mod signal;
#[path = "audio/transform.gen.rs"]
mod transform;

pub(crate) use audio::validate_sample_buffer;
pub use audio::{Audio, AudioChannelLayout};
pub use capture::{AudioCapture, AudioCaptureChunk};
pub use codec::wav::{encode_interleaved_wav_f32, encode_wav_f32, save_wav_f32};
pub use decoder::{decode_file, decode_file_on, decode_memory, decode_memory_on};
pub use encoder::{AudioCodec, AudioEncodeProfile, AudioEncoder, EncodedAudioPacket};
pub use player::AudioPlayer;
pub use signal::{
	BiquadCoefficients, NormalizeAudioConfig, NormalizeAudioMode, ResampleConfig, amplitude_to_db,
	biquad, clip, fade, gain, mix, normalize, pre_emphasis, resample, reverb, saturate, sos_filter,
	to_mono, waveform_envelope,
};
pub use transform::{MelConfig, MfccConfig, StftConfig, StftWindow, mel_spectrogram, mfcc, stft};
pub use types::{AudioCaptureConfig, AudioPlayerConfig};

#[path = "audio/parity.gen.rs"]
mod parity;
#[path = "audio/type.gen.rs"]
mod types;
