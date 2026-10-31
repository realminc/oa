//! Checked public Matrix transport around the dedicated secret executor.
//! No secret allocation is converted to ordinary Storage or Matrix.
use super::{
	Buffer, EngineHandle, Event, MlDsaDeviceKey, MlKemDeviceKey, MlKemSharedSecret, SecretErasure,
	Storage,
};
use crate::{DType, Error, Matrix, Result, cryptography::entropy::Entropy};

impl EngineHandle {
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
		if value.dtype() != DType::U8 || !self.same_as(value.engine_handle()) {
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
		let message = self.pqc_public_buffer(message)?;
		let context = self.pqc_public_buffer(context)?;
		let (signature, status, erasure) = match algorithm {
			Some(algorithm) => self.sign_mldsa_hash_message(
				key,
				&message,
				&context,
				[message_length, context_length],
				algorithm,
				entropy,
			)?,
			None => self.sign_mldsa_message(
				key,
				&message,
				&context,
				[message_length, context_length],
				entropy,
			)?,
		};
		let (_, _, _, length) = super::super::shader::KernelId::mldsa_sign_layout(key.parameter)
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
		let (_, _, _, length) = super::super::shader::KernelId::mldsa_sign_layout(key.parameter)
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
}

impl MlDsaDeviceKey {
	pub(crate) fn engine(&self) -> &EngineHandle {
		&self.engine
	}
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.engine.clone().erase_mldsa_key(self)
	}
}

impl MlKemDeviceKey {
	pub(crate) fn engine(&self) -> &EngineHandle {
		&self.engine
	}
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.engine.clone().erase_mlkem_key(self)
	}
}

impl MlKemSharedSecret {
	pub(crate) fn erase(self) -> Result<SecretErasure> {
		self.engine.clone().erase_mlkem_shared_secret(self)
	}
}
