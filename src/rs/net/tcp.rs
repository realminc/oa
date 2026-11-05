//! Rust-owned blocking TCP sockets with OA errors and explicit close.

use std::{
	io::{Read, Write},
	net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4},
	time::Duration,
};

use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use crate::{Error, Result};

/// One connected blocking TCP byte stream.
///
/// This type owns its socket. Dropping it releases the socket; `close` makes
/// that lifecycle transition explicit and idempotent. A clone obtained with
/// `try_clone` shares the underlying TCP connection, including shutdown state.
pub struct TcpStream {
	inner: Option<std::net::TcpStream>,
	peer: SocketAddr,
}

impl TcpStream {
	/// Connect to one resolved host and TCP port.
	///
	/// # Errors
	///
	/// Returns an OA I/O error if name resolution or connection fails.
	pub fn connect(host: &str, port: u16) -> Result<Self> {
		if host.is_empty() {
			return Err(Error::invalid_argument("TCP host must not be empty"));
		}
		let inner = std::net::TcpStream::connect((host, port))
			.map_err(|error| Error::io("TCP connect", error))?;
		inner
			.set_nodelay(true)
			.map_err(|error| Error::io("TCP disable Nagle", error))?;
		Self::from_std(inner)
	}

	fn from_std(inner: std::net::TcpStream) -> Result<Self> {
		let peer = inner
			.peer_addr()
			.map_err(|error| Error::io("TCP peer address", error))?;
		Ok(Self {
			inner: Some(inner),
			peer,
		})
	}

	/// Read at most `bytes.len()` bytes. Zero means peer EOF or an empty buffer.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for a socket failure.
	pub fn read(&mut self, bytes: &mut [u8]) -> Result<usize> {
		self
			.raw_mut()?
			.read(bytes)
			.map_err(|error| Error::io("TCP read", error))
	}

	/// Write at most `bytes.len()` bytes.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for a socket failure.
	pub fn write(&mut self, bytes: &[u8]) -> Result<usize> {
		self
			.raw_mut()?
			.write(bytes)
			.map_err(|error| Error::io("TCP write", error))
	}

	/// Write every byte or report the socket failure.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for a socket failure.
	pub fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
		self
			.raw_mut()?
			.write_all(bytes)
			.map_err(|error| Error::io("TCP write all", error))
	}

	/// Set both read and write timeouts. Zero restores blocking behavior.
	///
	/// If configuring the write timeout fails, the prior read timeout is
	/// restored before the error is returned.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for a socket failure.
	pub fn set_io_timeout(&self, timeout: Duration) -> Result<()> {
		let inner = self.raw()?;
		let previous_read = inner
			.read_timeout()
			.map_err(|error| Error::io("TCP read timeout query", error))?;
		let timeout = (!timeout.is_zero()).then_some(timeout);
		inner
			.set_read_timeout(timeout)
			.map_err(|error| Error::io("TCP read timeout", error))?;
		if let Err(error) = inner.set_write_timeout(timeout) {
			let _ = inner.set_read_timeout(previous_read);
			return Err(Error::io("TCP write timeout", error));
		}
		Ok(())
	}

	/// Return a second handle to the same connection, useful for interrupting a
	/// blocking read from another thread with `shutdown`.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for clone failure.
	pub fn try_clone(&self) -> Result<Self> {
		let inner = self
			.raw()?
			.try_clone()
			.map_err(|error| Error::io("TCP clone", error))?;
		Ok(Self {
			inner: Some(inner),
			peer: self.peer,
		})
	}

	/// Interrupt reads and writes on this connection and its clones.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for shutdown failure.
	pub fn shutdown(&self) -> Result<()> {
		self
			.raw()?
			.shutdown(Shutdown::Both)
			.map_err(|error| Error::io("TCP shutdown", error))
	}

	/// Release this socket handle. Cloned handles remain valid until closed.
	pub fn close(&mut self) {
		self.inner.take();
	}

	/// Whether this wrapper still owns a socket handle.
	#[must_use]
	pub const fn is_open(&self) -> bool {
		self.inner.is_some()
	}

	/// Return the resolved peer address captured when the stream was created.
	#[must_use]
	pub const fn peer_addr(&self) -> SocketAddr {
		self.peer
	}

	/// Return the resolved peer IP address.
	#[must_use]
	pub const fn remote_addr(&self) -> IpAddr {
		self.peer.ip()
	}

	/// Return the resolved peer port.
	#[must_use]
	pub const fn remote_port(&self) -> u16 {
		self.peer.port()
	}

	fn raw(&self) -> Result<&std::net::TcpStream> {
		self
			.inner
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("TCP stream is closed"))
	}

	pub(super) fn raw_mut(&mut self) -> Result<&mut std::net::TcpStream> {
		self
			.inner
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("TCP stream is closed"))
	}
}

/// One blocking IPv4 TCP listener.
///
/// The caller owns the listener and every accepted stream independently.
pub struct TcpListener {
	inner: Option<std::net::TcpListener>,
	local: SocketAddr,
}

impl TcpListener {
	/// Bind all IPv4 interfaces with the donor's default backlog of 128.
	///
	/// # Errors
	///
	/// Returns an OA I/O error if socket creation, binding, or listening fails.
	pub fn bind_any(port: u16) -> Result<Self> {
		Self::bind("0.0.0.0", port, 128)
	}

	/// Bind an IPv4 host and port with an explicit positive listen backlog.
	/// A zero port asks the operating system to select an available port.
	///
	/// # Errors
	///
	/// Returns `InvalidArgument` for an invalid IPv4 address or backlog, or
	/// `Io` for a socket failure.
	pub fn bind(host: &str, port: u16, backlog: i32) -> Result<Self> {
		if backlog <= 0 {
			return Err(Error::invalid_argument(
				"TCP listen backlog must be positive",
			));
		}
		let address = host
			.parse::<Ipv4Addr>()
			.map_err(|_| Error::invalid_argument("TCP bind host must be an IPv4 address"))?;
		let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
			.map_err(|error| Error::io("TCP socket", error))?;
		socket
			.set_reuse_address(true)
			.map_err(|error| Error::io("TCP reuse address", error))?;
		socket
			.bind(&SockAddr::from(SocketAddrV4::new(address, port)))
			.map_err(|error| Error::io("TCP bind", error))?;
		socket
			.listen(backlog)
			.map_err(|error| Error::io("TCP listen", error))?;
		let inner: std::net::TcpListener = socket.into();
		let local = inner
			.local_addr()
			.map_err(|error| Error::io("TCP local address", error))?;
		Ok(Self {
			inner: Some(inner),
			local,
		})
	}

	/// Accept one connected stream.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` after close, or `Io` for accept failure.
	pub fn accept(&self) -> Result<TcpStream> {
		let inner = self
			.inner
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("TCP listener is closed"))?;
		let (stream, _) = inner
			.accept()
			.map_err(|error| Error::io("TCP accept", error))?;
		stream
			.set_nodelay(true)
			.map_err(|error| Error::io("TCP disable Nagle", error))?;
		TcpStream::from_std(stream)
	}

	/// Release the listener socket without affecting accepted streams.
	pub fn close(&mut self) {
		self.inner.take();
	}

	/// Whether this wrapper still owns a listener socket.
	#[must_use]
	pub const fn is_open(&self) -> bool {
		self.inner.is_some()
	}

	/// Return the bound address captured after listen, including an OS-selected port.
	#[must_use]
	pub const fn local_addr(&self) -> SocketAddr {
		self.local
	}

	/// Return the bound port.
	#[must_use]
	pub const fn port(&self) -> u16 {
		self.local.port()
	}
}
