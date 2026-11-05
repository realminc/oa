"""SDK audio command-recognition vocabulary (mirrors ``oa::sdk::ml::audio``).

Speech-commands dataset loading, single-dispatch batched mel features, the
canonical CNN classifier builder, and the recorded OARS oracle values for the
audio command tutorial.
"""

from .._native import (
	AUDIO_BATCH_SIZE,
	AUDIO_CLIP_SAMPLE_RATE,
	AUDIO_COMMAND_LABELS,
	AUDIO_MEL_BANDS,
	AUDIO_MEL_FFT_SIZE,
	AUDIO_MEL_HOP_SIZE,
	AUDIO_MEL_CHUNK_CLIPS,
	AUDIO_RNG_SEED,
	AUDIO_SAMPLES_PER_CLIP,
	AUDIO_TRAINING_EPOCHS,
	COMMANDS_FINAL_LOSS,
	COMMANDS_TEST_ACCURACY,
	SpeechCommandsDataset,
	ml_audio_accuracy as accuracy,
	ml_audio_commands_cnn as commands_cnn,
)

__all__ = [
	"AUDIO_BATCH_SIZE",
	"AUDIO_CLIP_SAMPLE_RATE",
	"AUDIO_COMMAND_LABELS",
	"AUDIO_MEL_BANDS",
	"AUDIO_MEL_FFT_SIZE",
	"AUDIO_MEL_HOP_SIZE",
	"AUDIO_MEL_CHUNK_CLIPS",
	"AUDIO_RNG_SEED",
	"AUDIO_SAMPLES_PER_CLIP",
	"AUDIO_TRAINING_EPOCHS",
	"COMMANDS_FINAL_LOSS",
	"COMMANDS_TEST_ACCURACY",
	"SpeechCommandsDataset",
	"accuracy",
	"commands_cnn",
]
