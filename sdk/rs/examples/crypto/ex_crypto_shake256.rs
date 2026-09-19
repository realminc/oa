// OA_DOC_BEGIN: crypto-shake256
use oa::{Engine, Matrix, cryptography};

fn main() -> oa::Result<()> {
	let engine = Engine::new()?;

	// Three 5-byte messages packed into a [3, 5] u8 matrix.
	let rows: &[&[u8]] = &[b"alpha", b"bravo", b"crypt"];
	let flat: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
	let messages = Matrix::from_slice::<u8>(&engine, [3, 5], &flat)?;

	let digests = cryptography::hash::shake256(&messages, 32)?;

	// Verify GPU digests match the independent CPU reference.
	let gpu = digests.read::<u8>()?;
	let cpu: Vec<u8> = rows
		.iter()
		.flat_map(|row| {
			let mut digest = [0u8; 32];
			cryptography::shake256(row, &mut digest);
			digest
		})
		.collect();
	assert_eq!(gpu, cpu);

	println!("3 GPU SHAKE-256 digests match the CPU oracle");
	Ok(())
}
// OA_DOC_END: crypto-shake256
