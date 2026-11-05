//! Bounded little-endian length-prefix framing over one TCP connection.

use std::io::{Read, Write};

use crate::{Error, Result};

use super::TcpStream;

/// Four-byte little-endian payload length followed by exact payload bytes.
///
/// A framing error closes the stream, because a partial or rejected frame
/// leaves no trustworthy next message boundary. Protocol semantics belong to
/// the consuming session.
pub struct TcpFramed;

impl TcpFramed {
	/// Absolute donor transport ceiling (16 MiB).
	pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

	/// Write one complete bounded message.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` before writing an oversized payload. A socket
	/// failure closes the stream and returns `Io`.
	pub fn write_message(stream: &mut TcpStream, payload: &[u8]) -> Result<()> {
		if payload.len() > Self::MAX_PAYLOAD_BYTES {
			return Err(Error::invalid_argument("TCP framed payload exceeds 16 MiB"));
		}
		let length = u32::try_from(payload.len())
			.map_err(|_| Error::out_of_range("TCP framed payload length exceeds u32"))?;
		let result = (|| -> Result<()> {
			let raw = stream.raw_mut()?;
			raw
				.write_all(&length.to_le_bytes())
				.map_err(|error| Error::io("TCP framed write header", error))?;
			raw
				.write_all(payload)
				.map_err(|error| Error::io("TCP framed write body", error))
		})();
		if result.is_err() {
			stream.close();
		}
		result
	}

	/// Read one complete message, checking its length before allocation.
	///
	/// `max_payload_bytes` may narrow the transport's 16 MiB ceiling. Empty
	/// messages are valid. Any malformed or incomplete frame closes the stream.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` for an invalid caller limit, `DataLoss` for a
	/// peer length above that limit, `FailedPrecondition` after close, or `Io`
	/// for a socket failure.
	pub fn read_message(stream: &mut TcpStream, max_payload_bytes: usize) -> Result<Vec<u8>> {
		let mut payload = Vec::new();
		Self::read_message_into(stream, &mut payload, max_payload_bytes)?;
		Ok(payload)
	}

	/// Read one message into a reusable caller-owned buffer.
	///
	/// The buffer is cleared before reading and remains empty after an I/O or
	/// framing failure. Its allocation may be reused across successful reads.
	/// An invalid caller limit leaves both the buffer and stream unchanged.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::read_message`].
	pub fn read_message_into(
		stream: &mut TcpStream,
		payload: &mut Vec<u8>,
		max_payload_bytes: usize,
	) -> Result<()> {
		if max_payload_bytes > Self::MAX_PAYLOAD_BYTES {
			return Err(Error::invalid_argument(
				"TCP framed caller limit exceeds 16 MiB",
			));
		}
		payload.clear();
		let result = (|| -> Result<()> {
			let raw = stream.raw_mut()?;
			let mut header = [0_u8; 4];
			raw
				.read_exact(&mut header)
				.map_err(|error| Error::io("TCP framed read header", error))?;
			let length = u32::from_le_bytes(header) as usize;
			if length > max_payload_bytes {
				return Err(Error::data_loss(
					"TCP framed peer length exceeds the admitted limit",
				));
			}
			payload
				.try_reserve_exact(length)
				.map_err(|_| Error::resource_exhausted("TCP framed payload allocation failed"))?;
			payload.resize(length, 0);
			raw
				.read_exact(payload)
				.map_err(|error| Error::io("TCP framed read body", error))?;
			Ok(())
		})();
		if result.is_err() {
			payload.clear();
			stream.close();
		}
		result
	}
}
