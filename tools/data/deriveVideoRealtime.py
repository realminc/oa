#!/usr/bin/env python3
"""Derive and verify OA's genuine local 60 fps video fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SOURCE_NAME = "ReadySetGo_3840x2160_120fps_420_8bit_HEVC_RAW.hevc"
SOURCE_BYTES = 30_724_839
SOURCE_SHA256 = "d2bc5ffd0f9b967239d3082e6dd9d9f6e32ae979f2b284631f56def0c7d2a6c3"
EXPECTED_FRAMES = 300
MINIMUM_PSNR_DB = 40.0


@dataclass(frozen=True)
class Variant:
	name: str
	outputName: str
	outputBytes: int
	outputSha256: str
	width: int
	height: int
	codecName: str
	profile: str
	ffprobeLevel: int | None
	sourceTransform: str
	encoderArgs: tuple[str, ...]


VARIANTS = {
	"1080p60-av1": Variant(
		name="1080p60-av1",
		outputName="ready_set_go_1080p_60fps_av1_main_8bit_420.mp4",
		outputBytes=28_948_149,
		outputSha256="de3b3d89c04997d502b9e7c211529a0893f32af72d843bb552ca03267fda280e",
		width=1920,
		height=1080,
		codecName="av1",
		profile="Main",
		ffprobeLevel=9,
		sourceTransform="select='not(mod(n,2))',scale=1920:1080:flags=lanczos",
		encoderArgs=(
			"-c:v", "libsvtav1", "-preset", "8", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "0", "-g", "60",
			"-threads", "4", "-svtav1-params",
			"tune=0:keyint=60:lp=4:pred-struct=1",
		),
	),
	"1080p60-h264": Variant(
		name="1080p60-h264",
		outputName="ready_set_go_1080p_60fps_h264_high_8bit_420.mp4",
		outputBytes=13_678_088,
		outputSha256="725351e7586c03b5f371c6fb69140a7622aea1dd1ab9d5a7d7bf9212565aae22",
		width=1920,
		height=1080,
		codecName="h264",
		profile="High",
		ffprobeLevel=42,
		sourceTransform="select='not(mod(n,2))',scale=1920:1080:flags=lanczos",
		encoderArgs=(
			"-c:v", "libx264", "-preset", "fast", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.2",
			"-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
			"-bf", "0", "-refs", "4", "-threads", "4",
		),
	),
	"1080p60-h265": Variant(
		name="1080p60-h265",
		outputName="ready_set_go_1080p_60fps_h265_main_8bit_420.mp4",
		outputBytes=14_342_967,
		outputSha256="6dc3c57694fbf4e344accb4e5931111b4e9379c95070d4ca5e5f55099ddd4b75",
		width=1920,
		height=1080,
		codecName="hevc",
		profile="Main",
		ffprobeLevel=123,
		sourceTransform="select='not(mod(n,2))',scale=1920:1080:flags=lanczos",
		encoderArgs=(
			"-c:v", "libx265", "-preset", "fast", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "main", "-level:v", "4.1",
			"-g", "60", "-keyint_min", "60", "-bf", "0", "-refs", "4",
			"-threads", "4", "-x265-params",
			"scenecut=0:pools=4:frame-threads=4", "-tag:v", "hvc1",
		),
	),
	"1080p60-vp9": Variant(
		name="1080p60-vp9",
		outputName="ready_set_go_1080p_60fps_vp9_profile0_8bit_420.mp4",
		outputBytes=23_319_028,
		outputSha256="1c7d5f8affe7bbf9fa017f466437219f447f9bfbf3fa3c3918a8f4d078384dfe",
		width=1920,
		height=1080,
		codecName="vp9",
		profile="Profile 0",
		ffprobeLevel=None,
		sourceTransform="select='not(mod(n,2))',scale=1920:1080:flags=lanczos",
		encoderArgs=(
			"-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "4",
			"-crf", "18", "-b:v", "0", "-pix_fmt", "yuv420p",
			"-profile:v", "0", "-g", "60", "-keyint_min", "60",
			"-auto-alt-ref", "0", "-lag-in-frames", "0", "-row-mt", "0",
			"-threads", "4",
		),
	),
	"2160p60-av1": Variant(
		name="2160p60-av1",
		outputName="ready_set_go_2160p_60fps_av1_main_8bit_420.mp4",
		outputBytes=57_333_783,
		outputSha256="200432fd39c78cd25a76ddff2bde081de15575e02d9f5d8a65139355efd30a4b",
		width=3840,
		height=2160,
		codecName="av1",
		profile="Main",
		ffprobeLevel=13,
		sourceTransform="select='not(mod(n,2))'",
		encoderArgs=(
			"-c:v", "libsvtav1", "-preset", "8", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "0", "-g", "60",
			"-threads", "4", "-svtav1-params",
			"tune=0:keyint=60:lp=4:pred-struct=1",
		),
	),
	"2160p60-h264": Variant(
		name="2160p60-h264",
		outputName="ready_set_go_2160p_60fps_h264_high_8bit_420.mp4",
		outputBytes=35_148_235,
		outputSha256="3b9302f7eeddb04dab046c051b30b9cb3ff3d5aaff90f8cf4cdf72b1f6472682",
		width=3840,
		height=2160,
		codecName="h264",
		profile="High",
		ffprobeLevel=52,
		sourceTransform="select='not(mod(n,2))'",
		encoderArgs=(
			"-c:v", "libx264", "-preset", "fast", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "5.2",
			"-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
			"-bf", "0", "-refs", "4", "-threads", "4",
		),
	),
	"2160p60-h265": Variant(
		name="2160p60-h265",
		outputName="ready_set_go_2160p_60fps_h265_main_8bit_420.mp4",
		outputBytes=30_491_343,
		outputSha256="04c1793ddf5393efcb002b8b73658e8d662063cbd027a69468dd6da16b93a97f",
		width=3840,
		height=2160,
		codecName="hevc",
		profile="Main",
		ffprobeLevel=153,
		sourceTransform="select='not(mod(n,2))'",
		encoderArgs=(
			"-c:v", "libx265", "-preset", "fast", "-crf", "18",
			"-pix_fmt", "yuv420p", "-profile:v", "main", "-level:v", "5.1",
			"-g", "60", "-keyint_min", "60", "-bf", "0", "-refs", "4",
			"-threads", "4", "-x265-params",
			"scenecut=0:pools=4:frame-threads=4", "-tag:v", "hvc1",
		),
	),
	"2160p60-vp9": Variant(
		name="2160p60-vp9",
		outputName="ready_set_go_2160p_60fps_vp9_profile0_8bit_420.mp4",
		outputBytes=48_451_123,
		outputSha256="2c2b0e34ae208c8fb3dcd88bc3cb6299f465f8a7419ff32c85a3a49dadc08912",
		width=3840,
		height=2160,
		codecName="vp9",
		profile="Profile 0",
		ffprobeLevel=None,
		sourceTransform="select='not(mod(n,2))'",
		encoderArgs=(
			"-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "4",
			"-crf", "18", "-b:v", "0", "-pix_fmt", "yuv420p",
			"-profile:v", "0", "-g", "60", "-keyint_min", "60",
			"-auto-alt-ref", "0", "-lag-in-frames", "0", "-row-mt", "0",
			"-threads", "4",
		),
	),
}


class DerivationError(RuntimeError):
	pass


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def requireFile(path: Path, expectedBytes: int, expectedHash: str) -> None:
	if not path.is_file():
		raise DerivationError(f"missing file: {path}")
	if path.stat().st_size != expectedBytes:
		raise DerivationError(
			f"{path}: size {path.stat().st_size}, expected {expectedBytes}")
	actualHash = sha256(path)
	if actualHash != expectedHash:
		raise DerivationError(
			f"{path}: sha256 {actualHash}, expected {expectedHash}")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		command,
		check=True,
		text=True,
		stdout=subprocess.PIPE if capture else None,
		stderr=subprocess.STDOUT if capture else None,
	)


def probe(ffprobe: str, output: Path, variant: Variant) -> None:
	completed = run([
		ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
		"-show_entries",
		"stream=codec_name,profile,level,width,height,pix_fmt,r_frame_rate,"
		"avg_frame_rate,nb_frames,nb_read_frames,duration",
		"-of", "json", os.fspath(output),
	], capture=True)
	streams = json.loads(completed.stdout).get("streams", [])
	if len(streams) != 1:
		raise DerivationError(f"{output}: expected one video stream")
	stream = streams[0]
	expected = {
		"codec_name": variant.codecName,
		"profile": variant.profile,
		"width": variant.width,
		"height": variant.height,
		"pix_fmt": "yuv420p",
		"r_frame_rate": "60/1",
		"avg_frame_rate": "60/1",
		"nb_frames": str(EXPECTED_FRAMES),
		"nb_read_frames": str(EXPECTED_FRAMES),
		"duration": "5.000000",
	}
	if variant.ffprobeLevel is not None:
		expected["level"] = variant.ffprobeLevel
	for name, value in expected.items():
		if stream.get(name) != value:
			raise DerivationError(
				f"{output}: {name}={stream.get(name)!r}, expected {value!r}")


def uniqueFrames(
	ffmpeg: str,
	path: Path,
	variant: Variant,
	source: bool,
) -> int:
	command = [ffmpeg, "-v", "error"]
	if source:
		command += ["-r", "120"]
	command += ["-i", os.fspath(path)]
	if source:
		command += [
			"-vf", f"{variant.sourceTransform},setpts=N/(60*TB)",
			"-frames:v", str(EXPECTED_FRAMES),
		]
	command += ["-f", "framemd5", "-"]
	completed = run(command, capture=True)
	hashes = []
	for line in completed.stdout.splitlines():
		if line.startswith("#") or not line.strip():
			continue
		fields = [field.strip() for field in line.split(",")]
		if len(fields) >= 6:
			hashes.append(fields[5])
	if len(hashes) != EXPECTED_FRAMES:
		raise DerivationError(
			f"{path}: decoded {len(hashes)} frames, expected {EXPECTED_FRAMES}")
	return len(set(hashes))


def comparePsnr(
	ffmpeg: str,
	source: Path,
	output: Path,
	variant: Variant,
) -> float:
	completed = run([
		ffmpeg, "-hide_banner", "-r", "120", "-i", os.fspath(source),
		"-i", os.fspath(output), "-filter_complex",
		f"[0:v]{variant.sourceTransform},settb=AVTB,setpts=N/(60*TB)[ref];"
		"[1:v]settb=AVTB,setpts=N/(60*TB)[test];[ref][test]psnr",
		"-frames:v", str(EXPECTED_FRAMES), "-f", "null", "-",
	], capture=True)
	match = re.search(r"PSNR .* average:([0-9.]+)", completed.stdout)
	if match is None:
		raise DerivationError("FFmpeg did not report aggregate PSNR")
	return float(match.group(1))


def derive(
	ffmpeg: str,
	source: Path,
	output: Path,
	variant: Variant,
) -> None:
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		prefix="oa-video-realtime-", suffix=".mp4", dir=output.parent,
		delete=False) as temporary:
		temporaryPath = Path(temporary.name)
	try:
		run([
			ffmpeg, "-hide_banner", "-y", "-v", "warning", "-r", "120",
			"-i", os.fspath(source), "-vf",
			f"{variant.sourceTransform},setpts=N/(60*TB)",
			"-frames:v", str(EXPECTED_FRAMES), "-r", "60", "-fps_mode", "cfr",
			"-an", "-map_metadata", "-1", *variant.encoderArgs,
			"-movflags", "+faststart", os.fspath(temporaryPath),
		])
		requireFile(
			temporaryPath,
			variant.outputBytes,
			variant.outputSha256,
		)
		os.replace(temporaryPath, output)
	finally:
		temporaryPath.unlink(missing_ok=True)


def defaultDataRoot() -> Path:
	if value := os.getenv("OA_DATA_DIR"):
		return Path(value).expanduser().resolve()
	return Path(__file__).resolve().parents[2] / "var" / "data"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", type=Path, default=defaultDataRoot())
	parser.add_argument("--ffmpeg", default="ffmpeg")
	parser.add_argument("--ffprobe", default="ffprobe")
	parser.add_argument("--verify-only", action="store_true")
	parser.add_argument(
		"--variant",
		choices=("all", *VARIANTS),
		default="all",
		help="derive or verify one fixture variant (default: all)",
	)
	args = parser.parse_args(argv)
	root = args.data_root.expanduser().resolve() / "videoRealtime"
	source = root / SOURCE_NAME
	try:
		requireFile(source, SOURCE_BYTES, SOURCE_SHA256)
		variants = VARIANTS.values() if args.variant == "all" else (
			VARIANTS[args.variant],)
		for variant in variants:
			output = root / variant.outputName
			if not args.verify_only:
				derive(args.ffmpeg, source, output, variant)
			requireFile(
				output,
				variant.outputBytes,
				variant.outputSha256,
			)
			probe(args.ffprobe, output, variant)
			sourceUnique = uniqueFrames(
				args.ffmpeg,
				source,
				variant,
				source=True,
			)
			outputUnique = uniqueFrames(
				args.ffmpeg,
				output,
				variant,
				source=False,
			)
			if sourceUnique != EXPECTED_FRAMES or outputUnique != EXPECTED_FRAMES:
				raise DerivationError(
					f"{variant.name}: cadence failure: "
					f"reference unique={sourceUnique}, "
					f"derived unique={outputUnique}, expected {EXPECTED_FRAMES}")
			psnr = comparePsnr(args.ffmpeg, source, output, variant)
			if psnr < MINIMUM_PSNR_DB:
				raise DerivationError(
					f"{variant.name}: derived/reference PSNR {psnr:.6f} dB, "
					f"minimum {MINIMUM_PSNR_DB:.1f} dB")
			print(
				f"verified {output}: frames={EXPECTED_FRAMES} "
				f"unique={outputUnique} fps=60/1 psnr_db={psnr:.6f} "
				f"sha256={variant.outputSha256}")
		return 0
	except (DerivationError, OSError, subprocess.CalledProcessError,
			json.JSONDecodeError) as error:
		print(f"video derivation: {error}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
