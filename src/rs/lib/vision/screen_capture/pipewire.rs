//! Donor screenCapture.cpp mapped-memory format negotiation and channel packing.
//! Mechanical binding adaptation: main-thread nonblocking loop, one latest
//! owned frame, immediate producer-buffer return. DMA-BUF is not advertised.

use super::{ScreenCaptureConfig, portal::Portal};
use crate::{Error, Result};
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::{
	param::video::{VideoFormat, VideoInfoRaw},
	pod::{Object, Pod, Property, Value},
};
use std::{cell::RefCell, rc::Rc, time::Instant};

pub(super) struct HostFrame {
	pub rgba: Vec<u8>,
	pub width: u32,
	pub height: u32,
	pub pts: u64,
}
struct State {
	format: VideoInfoRaw,
	pending: Option<HostFrame>,
	error: Option<Error>,
	streaming: bool,
	origin: Instant,
}

pub(super) struct Backend {
	// Listener hooks are removed before destroying the stream/core. Rc owners
	// retain their context and loop. No worker thread or Drop join is needed.
	_listener: pw::stream::StreamListener<Rc<RefCell<State>>>,
	_core_listener: pw::core::Listener,
	stream: pw::stream::StreamRc,
	loop_: pw::main_loop::MainLoopRc,
	state: Rc<RefCell<State>>,
	_portal: Box<Portal>,
}

impl Backend {
	pub(super) fn open(config: &ScreenCaptureConfig) -> Result<Self> {
		let (portal, fd, node) = Portal::open(config)?;
		pw::init();
		let loop_ = pw::main_loop::MainLoopRc::new(None).map_err(pw_error)?;
		let context = pw::context::ContextRc::new(&loop_, None).map_err(pw_error)?;
		let core = context.connect_fd_rc(fd, None).map_err(pw_error)?;
		let state = Rc::new(RefCell::new(State {
			format: VideoInfoRaw::default(),
			pending: None,
			error: None,
			streaming: false,
			origin: Instant::now(),
		}));
		let errors = state.clone();
		let core_listener = core
			.add_listener_local()
			.error(move |_, _, _, message| {
				let mut state = errors.borrow_mut();
				state.error = Some(Error::unavailable(format!("PipeWire remote: {message}")));
				state.streaming = false;
			})
			.register();
		let stream = pw::stream::StreamRc::new(core, "oa-screen-capture", properties! {
			*pw::keys::MEDIA_TYPE => "Video", *pw::keys::MEDIA_CATEGORY => "Capture", *pw::keys::MEDIA_ROLE => "Screen",
		}).map_err(pw_error)?;
		let ring = config.ring_frames.max(2);
		let listener = stream
			.add_local_listener_with_user_data(state.clone())
			.state_changed(|_, state, old, new| {
				let mut state = state.borrow_mut();
				state.streaming = matches!(new, pw::stream::StreamState::Streaming);
				if let pw::stream::StreamState::Error(message) = new {
					state.error = Some(Error::unavailable(format!(
						"PipeWire screen stream: {message}"
					)));
				} else if matches!(new, pw::stream::StreamState::Unconnected)
					&& !matches!(old, pw::stream::StreamState::Unconnected)
				{
					state.error = Some(Error::unavailable("screen sharing stopped"));
				}
			})
			.param_changed(move |stream, state, id, param| {
				if id != spa::param::ParamType::Format.as_raw() {
					return;
				}
				let Some(param) = param else {
					return;
				};
				let mut state = state.borrow_mut();
				let result = (|| -> Result<()> {
					state
						.format
						.parse(param)
						.map_err(|e| Error::data_loss(format!("invalid screen format: {e}")))?;
					let size = state.format.size();
					validate_format(state.format.format(), size.width, size.height)?;
					let stride = size
						.width
						.checked_mul(4)
						.ok_or_else(|| Error::out_of_range("screen stride overflow"))?;
					let bytes = stride
						.checked_mul(size.height)
						.and_then(|v| i32::try_from(v).ok())
						.ok_or_else(|| Error::out_of_range("screen buffer exceeds i32"))?;
					let props = vec![
						Property::new(spa::sys::SPA_PARAM_BUFFERS_buffers, Value::Int(ring as i32)),
						Property::new(spa::sys::SPA_PARAM_BUFFERS_blocks, Value::Int(1)),
						Property::new(spa::sys::SPA_PARAM_BUFFERS_size, Value::Int(bytes)),
						Property::new(
							spa::sys::SPA_PARAM_BUFFERS_stride,
							Value::Int(stride as i32),
						),
						Property::new(
							spa::sys::SPA_PARAM_BUFFERS_dataType,
							Value::Int((1 << spa::sys::SPA_DATA_MemPtr) | (1 << spa::sys::SPA_DATA_MemFd)),
						),
					];
					let values = serialize(Object {
						type_: spa::sys::SPA_TYPE_OBJECT_ParamBuffers,
						id: spa::param::ParamType::Buffers.as_raw(),
						properties: props,
					})?;
					let pod =
						Pod::from_bytes(&values).ok_or_else(|| Error::internal("invalid screen buffer pod"))?;
					stream.update_params(&mut [pod]).map_err(pw_error)
				})();
				if let Err(error) = result {
					state.error = Some(error);
				}
			})
			.process(|stream, state| {
				let Some(mut buffer) = stream.dequeue_buffer() else {
					return;
				};
				let Some(data) = buffer.datas_mut().first_mut() else {
					return;
				};
				if data.chunk().size() == 0
					|| data
						.chunk()
						.flags()
						.contains(spa::buffer::ChunkFlags::CORRUPTED)
				{
					return;
				}
				let offset = data.chunk().offset() as usize;
				let length = data.chunk().size() as usize;
				let stride = data.chunk().stride();
				let mut state = state.borrow_mut();
				let result = (|| -> Result<HostFrame> {
					let size = state.format.size();
					let source = data
						.data()
						.ok_or_else(|| Error::data_loss("PipeWire supplied unmapped screen pixels"))?;
					let end = offset
						.checked_add(length)
						.ok_or_else(|| Error::out_of_range("screen chunk offset overflow"))?;
					let source = source
						.get(offset..end)
						.ok_or_else(|| Error::data_loss("screen chunk exceeds mapped buffer"))?;
					let rgba = pack_rgba(
						source,
						stride,
						size.width,
						size.height,
						state.format.format(),
					)?;
					Ok(HostFrame {
						rgba,
						width: size.width,
						height: size.height,
						pts: u64::try_from(state.origin.elapsed().as_micros())
							.map_err(|_| Error::out_of_range("screen timestamp overflow"))?,
					})
				})();
				match result {
					Ok(frame) => state.pending = Some(frame),
					Err(error) => state.error = Some(error),
				}
				// The binding requeues the producer buffer here, after the owned copy.
			})
			.register()
			.map_err(pw_error)?;
		let formats = format_pod(config)?;
		let pod =
			Pod::from_bytes(&formats).ok_or_else(|| Error::internal("invalid screen format pod"))?;
		stream
			.connect(
				spa::utils::Direction::Input,
				Some(node),
				pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
				&mut [pod],
			)
			.map_err(pw_error)?;
		Ok(Self {
			_listener: listener,
			_core_listener: core_listener,
			stream,
			loop_,
			state,
			_portal: portal,
		})
	}
	pub(super) fn poll(&mut self) -> Result<Option<HostFrame>> {
		self._portal.dispatch()?;
		let status = self.loop_.loop_().iterate(pw::loop_::Timeout::None);
		if status < 0 {
			return Err(Error::io(
				"PipeWire screen loop",
				std::io::Error::from_raw_os_error(-status),
			));
		}
		let mut state = self.state.borrow_mut();
		if let Some(error) = state.error.take() {
			return Err(error);
		}
		Ok(state.pending.take())
	}
	pub(super) fn is_streaming(&self) -> bool {
		self.state.borrow().streaming
	}
	pub(super) fn close(&self) -> Result<()> {
		if matches!(self.stream.state(), pw::stream::StreamState::Unconnected) {
			return Ok(());
		}
		self.stream.disconnect().map_err(pw_error)
	}
}

fn pw_error(error: pw::Error) -> Error {
	Error::backend_failure("PipeWire", "screen capture", error)
}
fn serialize(object: Object) -> Result<Vec<u8>> {
	spa::pod::serialize::PodSerializer::serialize(
		std::io::Cursor::new(Vec::new()),
		&Value::Object(object),
	)
	.map(|(writer, _)| writer.into_inner())
	.map_err(|e| Error::data_loss(format!("screen format serialization: {e:?}")))
}
fn format_pod(config: &ScreenCaptureConfig) -> Result<Vec<u8>> {
	use spa::{
		param::format::{FormatProperties as F, MediaSubtype, MediaType},
		utils::{Fraction, Rectangle},
	};
	serialize(spa::pod::object!(
		spa::utils::SpaTypes::ObjectParamFormat,
		spa::param::ParamType::EnumFormat,
		spa::pod::property!(F::MediaType, Id, MediaType::Video),
		spa::pod::property!(F::MediaSubtype, Id, MediaSubtype::Raw),
		spa::pod::property!(
			F::VideoFormat,
			Choice,
			Enum,
			Id,
			VideoFormat::RGBA,
			VideoFormat::RGBA,
			VideoFormat::RGBx,
			VideoFormat::BGRA,
			VideoFormat::BGRx
		),
		spa::pod::property!(
			F::VideoSize,
			Choice,
			Range,
			Rectangle,
			Rectangle {
				width: config.preferred_width,
				height: config.preferred_height
			},
			Rectangle {
				width: 1,
				height: 1
			},
			Rectangle {
				width: 16384,
				height: 16384
			}
		),
		spa::pod::property!(
			F::VideoFramerate,
			Choice,
			Range,
			Fraction,
			Fraction {
				num: config.preferred_fps,
				denom: 1
			},
			Fraction { num: 0, denom: 1 },
			Fraction { num: 240, denom: 1 }
		),
	))
}

fn validate_format(format: VideoFormat, width: u32, height: u32) -> Result<()> {
	if !matches!(
		format,
		VideoFormat::RGBA | VideoFormat::RGBx | VideoFormat::BGRA | VideoFormat::BGRx
	) {
		return Err(Error::data_loss("unsupported screen pixel format"));
	}
	if width == 0 || height == 0 || width > 16384 || height > 16384 {
		return Err(Error::data_loss("invalid screen frame extent"));
	}
	Ok(())
}

fn pack_rgba(
	source: &[u8],
	stride: i32,
	width: u32,
	height: u32,
	format: VideoFormat,
) -> Result<Vec<u8>> {
	validate_format(format, width, height)?;
	let row_bytes = (width as usize)
		.checked_mul(4)
		.ok_or_else(|| Error::out_of_range("screen row overflow"))?;
	let stride = if stride == 0 {
		row_bytes
	} else {
		usize::try_from(stride).map_err(|_| Error::data_loss("negative screen stride"))?
	};
	if stride < row_bytes {
		return Err(Error::data_loss(
			"screen stride is smaller than a pixel row",
		));
	}
	let required = stride
		.checked_mul(height as usize - 1)
		.and_then(|v| v.checked_add(row_bytes))
		.ok_or_else(|| Error::out_of_range("screen frame overflow"))?;
	if source.len() < required {
		return Err(Error::data_loss("truncated screen frame"));
	}
	let count = row_bytes
		.checked_mul(height as usize)
		.ok_or_else(|| Error::out_of_range("screen pixel count overflow"))?;
	let mut rgba = Vec::new();
	rgba
		.try_reserve_exact(count)
		.map_err(|_| Error::resource_exhausted("screen pixel allocation failed"))?;
	rgba.resize(count, 0);
	let swap = matches!(format, VideoFormat::BGRA | VideoFormat::BGRx);
	let opaque = matches!(format, VideoFormat::RGBx | VideoFormat::BGRx);
	for y in 0..height as usize {
		let row = &source[y * stride..y * stride + row_bytes];
		let destination = &mut rgba[y * row_bytes..(y + 1) * row_bytes];
		if !swap && !opaque {
			destination.copy_from_slice(row);
			continue;
		}
		for (pixel, output) in row
			.as_chunks::<4>()
			.0
			.iter()
			.zip(destination.as_chunks_mut::<4>().0)
		{
			*output = [
				pixel[if swap { 2 } else { 0 }],
				pixel[1],
				pixel[if swap { 0 } else { 2 }],
				if opaque { 255 } else { pixel[3] },
			];
		}
	}
	Ok(rgba)
}

#[cfg(test)]
#[path = "../../../../../test/rs/vision/screen_capture_unit.rs"]
mod tests;
