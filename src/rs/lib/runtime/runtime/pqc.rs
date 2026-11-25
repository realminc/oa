//! Checked public Matrix transport around the dedicated secret executor.
//! No secret allocation is converted to ordinary Storage or Matrix.
use super::{
	Buffer, Event, MlDsaDeviceKey, MlKemDeviceKey, MlKemSharedSecret, RuntimeHandle, SecretErasure,
	Storage,
};
use crate::{DType, Error, Matrix, Result, crypto::entropy::Entropy};

impl RuntimeHandle {
	fn pqc_public_matrix(
		&self,
		buffer: Buffer,
		ready: Event,
		shape: Vec<usize>,
		dtype: DType,
	) -> Result<Matrix> {
		let device = self.state.borrow().device.clone();
		let storage = Storage::from_submitted_buffer(&device, buffer, ready)?;
		Matrix::from_runtime_storage(self, storage, shape, dtype)
	}

	fn pqc_public_buffer(&self, value: &Matrix) -> Result<Buffer> {
		if value.dtype() != DType::U8 || !self.same_as(value.runtime()) {
			return Err(Error::invalid_argument(
				"PQC transport requires matching-owner public U8 storage",
			));
		}
		value
			.storage()
			.validate_recording_access(crate::runtime::BufferAccess::Read)?;
		if let Some(buffer) = value.storage().buffer() {
			return Ok(buffer.clone());
		}
		// An empty public message/context still needs a valid descriptor. Its
		// declared length is zero, so the shader never reads these dummy bytes.
		Buffer::host_visible_storage(&self.state.borrow().device, 4)
	}

	pub(crate) fn pqc_mldsa_keygen(
		&self,
		parameter: u32,
		entropy: Entropy<32>,
	) -> Result<(MlDsaDeviceKey, Matrix, SecretErasure)> {
		let (key, erasure) = self.generate_mldsa_key(parameter, entropy)?;
		let bytes = key.public.byte_len();
		let public = self.pqc_public_matrix(
			key.public.clone(),
			key.ready.clone(),
			vec![1, bytes],
			DType::U8,
		)?;
		Ok((key, public, erasure))
	}

	pub(crate) fn pqc_mlkem_keygen(
		&self,
		k: u32,
		entropy: Entropy<64>,
	) -> Result<(MlKemDeviceKey, Matrix, SecretErasure)> {
		let (key, erasure) = self.generate_mlkem_key(k, entropy)?;
		let bytes = key.public.byte_len();
		let public = self.pqc_public_matrix(
			key.public.clone(),
			key.ready.clone(),
			vec![1, bytes],
			DType::U8,
		)?;
		Ok((key, public, erasure))
	}

	pub(crate) fn pqc_mlkem_import(
		&self,
		k: u32,
		decapsulation_key: &[u8],
	) -> Result<(MlKemDeviceKey, Matrix, SecretErasure)> {
		let (key, erasure) = self.import_mlkem_key(k, decapsulation_key)?;
		let bytes = key.public.byte_len();
		let public = self.pqc_public_matrix(
			key.public.clone(),
			key.ready.clone(),
			vec![1, bytes],
			DType::U8,
		)?;
		Ok((key, public, erasure))
	}

	pub(crate) fn pqc_mldsa_import(
		&self,
		parameter: u32,
		secret_key: &[u8],
		public_key: &Matrix,
	) -> Result<(MlDsaDeviceKey, Matrix, SecretErasure)> {
		let public_source = self.pqc_public_buffer(public_key)?;
		let (key, erasure) = self.import_mldsa_key(parameter, secret_key, &public_source)?;
		let bytes = key.public.byte_len();
		let public = self.pqc_public_matrix(
			key.public.clone(),
			key.ready.clone(),
			vec![1, bytes],
			DType::U8,
		)?;
		Ok((key, public, erasure))
	}

	pub(crate) fn pqc_mldsa_sign(
		&self,
		key: &MlDsaDeviceKey,
		message: &Matrix,
		context: &Matrix,
		algorithm: Option<u32>,
		entropy: Entropy<32>,
	) -> Result<(Matrix, Matrix, SecretErasure)> {
		let lengths = [message.num_elements(), context.num_elements()].map(|n| {
			u32::try_from(n).map_err(|_| Error::invalid_argument("ML-DSA public input exceeds u32"))
		});
		let [Ok(message_length), Ok(context_length)] = lengths else {
			return Err(Error::invalid_argument("ML-DSA input length exceeds u32"));
		};
		let context = self.pqc_public_buffer(context)?;
		let (signature, status, erasure) = match algorithm {
			// HashML-DSA prehashes the public message with the ordinary hash
			// operation family, then reuses the qualified prehashed signer. The
			// digest stays on device through deferred submission order.
			Some(algorithm) => {
				let rows = message.reshape_view(vec![1, message.num_elements()])?;
				let digest = match algorithm {
					1 => crate::crypto::hash::sha256(&rows)?,
					2 => crate::crypto::hash::sha384(&rows)?,
					3 => crate::crypto::hash::sha512(&rows)?,
					4 => crate::crypto::hash::sha224(&rows)?,
					5 => crate::crypto::hash::sha512_224(&rows)?,
					6 => crate::crypto::hash::sha512_256(&rows)?,
					7 => crate::crypto::hash::sha3_224(&rows)?,
					8 => crate::crypto::hash::sha3_256(&rows)?,
					9 => crate::crypto::hash::sha3_384(&rows)?,
					10 => crate::crypto::hash::sha3_512(&rows)?,
					11 => crate::crypto::hash::shake128(&rows, 32)?,
					12 => crate::crypto::hash::shake256(&rows, 64)?,
					_ => {
						return Err(Error::invalid_argument(
							"ML-DSA prehash identity is unknown",
						));
					}
				};
				let digest = self.pqc_public_buffer(&digest)?;
				self.sign_mldsa_prehashed(key, &digest, &context, algorithm, context_length, entropy)?
			}
			None => {
				let message = self.pqc_public_buffer(message)?;
				self.sign_mldsa_message(
					key,
					&message,
					&context,
					[message_length, context_length],
					entropy,
				)?
			}
		};
		let (_, _, _, length) =
			super::super::dna::kernel::KernelId::mldsa_sign_layout(key.parameter)
				.ok_or_else(|| Error::internal("retained ML-DSA key has invalid parameter metadata"))?;
		let bytes = self.pqc_public_matrix(
			signature,
			erasure.event().clone(),
			vec![1, length as usize],
			DType::U8,
		)?;
		let status = self.pqc_public_matrix(status, erasure.event().clone(), vec![1], DType::U32)?;
		Ok((bytes, status, erasure))
	}

	pub(crate) fn pqc_mldsa_sign_prehashed(
		&self,
		key: &MlDsaDeviceKey,
		digest: &Matrix,
		context: &Matrix,
		algorithm: u32,
		entropy: Entropy<32>,
	) -> Result<(Matrix, Matrix, SecretErasure)> {
		let context_length = u32::try_from(context.num_elements())
			.map_err(|_| Error::invalid_argument("ML-DSA context exceeds u32"))?;
		let digest = self.pqc_public_buffer(digest)?;
		let context = self.pqc_public_buffer(context)?;
		let (signature, status, erasure) =
			self.sign_mldsa_prehashed(key, &digest, &context, algorithm, context_length, entropy)?;
		let (_, _, _, length) =
			super::super::dna::kernel::KernelId::mldsa_sign_layout(key.parameter)
				.ok_or_else(|| Error::internal("retained ML-DSA key has invalid parameter metadata"))?;
		let bytes = self.pqc_public_matrix(
			signature,
			erasure.event().clone(),
			vec![1, length as usize],
			DType::U8,
		)?;
		let status = self.pqc_public_matrix(status, erasure.event().clone(), vec![1], DType::U32)?;
		Ok((bytes, status, erasure))
	}

	pub(crate) fn pqc_mlkem_encapsulate(
		&self,
		k: u32,
		public: &Matrix,
		entropy: Entropy<32>,
	) -> Result<(MlKemSharedSecret, Matrix, Matrix, SecretErasure)> {
		let public = self.pqc_public_buffer(public)?;
		let (secret, ciphertext, status, erasure) =
			self.encapsulate_mlkem_public(k, &public, entropy)?;
		let bytes = ciphertext.byte_len();
		let ciphertext = self.pqc_public_matrix(
			ciphertext,
			erasure.event().clone(),
			vec![1, bytes],
			DType::U8,
		)?;
		let status = self.pqc_public_matrix(status, erasure.event().clone(), vec![1], DType::U32)?;
		Ok((secret, ciphertext, status, erasure))
	}

	pub(crate) fn pqc_mlkem_decapsulate(
		&self,
		key: &MlKemDeviceKey,
		ciphertext: &Matrix,
	) -> Result<(MlKemSharedSecret, SecretErasure)> {
		let ciphertext = self.pqc_public_buffer(ciphertext)?;
		self.decapsulate_mlkem(key, &ciphertext)
	}

	pub(crate) fn pqc_mlkem_derive(
		&self,
		secret: &MlKemSharedSecret,
		info: &Matrix,
		output_len: u32,
	) -> Result<(Matrix, SecretErasure)> {
		let info_len = u32::try_from(info.num_elements())
			.map_err(|_| Error::invalid_argument("ML-KEM derivation info exceeds u32"))?;
		let info = self.pqc_public_buffer(info)?;
		let (output, erasure) = self.derive_mlkem_shared_secret(secret, &info, info_len, output_len)?;
		let bytes = output.byte_len();
		let derived =
			self.pqc_public_matrix(output, erasure.event().clone(), vec![1, bytes], DType::U8)?;
		Ok((derived, erasure))
	}
}

impl MlDsaDeviceKey {
	pub(crate) fn runtime(&self) -> &RuntimeHandle {
		&self.runtime
	}
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.runtime.clone().erase_mldsa_key(self)
	}
}

impl MlKemDeviceKey {
	pub(crate) fn runtime(&self) -> &RuntimeHandle {
		&self.runtime
	}
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.runtime.clone().erase_mlkem_key(self)
	}
}

impl MlKemSharedSecret {
	pub(crate) fn runtime(&self) -> &RuntimeHandle {
		&self.runtime
	}
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.runtime.clone().erase_mlkem_shared_secret(self)
	}
}
