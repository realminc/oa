#!/usr/bin/env python3
"""Publish browser-playable evidence from a successful OA audio example run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "var" / "example" / "audio" / "oaNarrationRoom.wav"
DEFAULT_OUTPUT = REPO / "sdk" / "asset" / "documentation" / "examples" / "audio"


def probe(path: Path) -> dict[str, int | float]:
	result = subprocess.run(
		[
			"ffprobe", "-v", "error", "-select_streams", "a:0",
			"-show_entries", "stream=sample_rate,channels,duration_ts,duration",
			"-of", "json", str(path),
		],
		check=True,
		capture_output=True,
		text=True,
	)
	stream = json.loads(result.stdout)["streams"][0]
	return {
		"sampleRate": int(stream["sample_rate"]),
		"channelCount": int(stream["channels"]),
		"sampleCount": int(stream["duration_ts"]),
		"durationSeconds": float(stream["duration"]),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	inputPath = args.input.resolve()
	if not inputPath.is_file():
		raise FileNotFoundError(
			f"missing example output: {inputPath}; run sdk/py/examples/audio/audio.py first"
		)
	metadata = probe(inputPath)
	expected = {
		"sampleRate": 24_000,
		"channelCount": 1,
		"sampleCount": 336_480,
		"durationSeconds": 14.02,
	}
	if metadata != expected:
		raise RuntimeError(f"audio example evidence metadata mismatch: {metadata} != {expected}")

	args.output.mkdir(parents=True, exist_ok=True)
	wavOutput = args.output / "oaNarrationRoom.wav"
	mp3Output = args.output / "oaNarrationRoom.mp3"
	shutil.copyfile(inputPath, wavOutput)
	subprocess.run(
		[
			"ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
			"-i", str(wavOutput), "-codec:a", "libmp3lame", "-b:a", "64k",
			str(mp3Output),
		],
		check=True,
	)
	print(f"published exact WAV and MP3 fallback from {inputPath}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
