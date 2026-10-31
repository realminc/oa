"""Audio values, codecs, DSP, feature extraction, capture, player, and encoder."""

from ._gen.audio import *
from ._gen.audio import __all__ as _GEN_ALL
from ._native import (
	Audio,
	AudioCapture,
	AudioCaptureChunk,
	AudioCaptureConfig,
	AudioEncodeProfile,
	AudioEncoder,
	AudioPlayer,
	AudioPlayerConfig,
	EncodedAudioPacket,
	audio_decode_file as decode_file,
	audio_decode_memory as decode_memory,
	audio_encode_interleaved_wav_f32 as encode_interleaved_wav_f32,
	audio_encode_wav_f32 as encode_wav_f32,
	audio_player_from_audio as player_from_audio,
	audio_save_wav_f32 as save_wav_f32,
)

__all__ = _GEN_ALL + [
	"Audio",
	"AudioCapture",
	"AudioCaptureChunk",
	"AudioCaptureConfig",
	"AudioEncodeProfile",
	"AudioEncoder",
	"AudioPlayer",
	"AudioPlayerConfig",
	"EncodedAudioPacket",
	"decode_file",
	"decode_memory",
	"encode_interleaved_wav_f32",
	"encode_wav_f32",
	"player_from_audio",
	"save_wav_f32",
]
