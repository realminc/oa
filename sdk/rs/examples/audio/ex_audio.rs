// OA_DOC_BEGIN: audio-process
use anyhow::Context;
use oa::{Engine, Filesystem, Path, audio};

fn main() -> anyhow::Result<()> {
	let engine = Engine::new().context("create OA engine")?;

	let audio_path = Path::asset_rel("audio/oaNarration.wav");

	let output = Path::var_rel("example/audio/oaNarrationRoom.wav");
	Filesystem::create_directories(&output.parent().unwrap())?;

	let source = audio::decode_file(&engine, &audio_path)
		.with_context(|| format!("decode {}", audio_path.display()))?;
	let mono = audio::to_mono(&source)?;
	let normalized = audio::normalize(
		&mono,
		audio::NormalizeAudioConfig {
			target_db: -3.0,
			..Default::default()
		},
	)?;
	let faded = audio::fade(&normalized, 2400, 2400)?;
	let reverberated = audio::reverb(&faded, 1.5, 0.4)?;
	audio::save_wav_f32(&output, &reverberated)?;

	assert!(reverberated.channels() == 1);
	assert!(output.is_file());

	println!(
		"Saved reverberated audio (1.5 s tail): {}",
		output.display()
	);
	Ok(())
}
// OA_DOC_END: audio-process
