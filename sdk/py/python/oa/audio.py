"""Audio values, codecs, DSP, and feature extraction."""

from ._native import (
	Audio,
	audio_amplitude_to_db as amplitude_to_db,
	audio_clip as clip,
	audio_decode_file as decode_file,
	audio_decode_memory as decode_memory,
	audio_encode_wav_f32 as encode_wav_f32,
	audio_fade as fade,
	audio_gain as gain,
	audio_mel_spectrogram as mel_spectrogram,
	audio_mix as mix,
	audio_normalize as normalize,
	audio_pre_emphasis as pre_emphasis,
	audio_resample as resample,
	audio_reverb as reverb,
	audio_saturate as saturate,
	audio_save_wav_f32 as save_wav_f32,
	audio_stft as stft,
	audio_to_mono as to_mono,
	audio_waveform_envelope as waveform_envelope,
)

__all__ = [
	"Audio", "amplitude_to_db", "clip", "decode_file", "decode_memory",
	"encode_wav_f32", "fade", "gain", "mel_spectrogram", "mix", "normalize",
	"pre_emphasis", "resample", "reverb", "saturate", "save_wav_f32", "stft",
	"to_mono", "waveform_envelope",
]
