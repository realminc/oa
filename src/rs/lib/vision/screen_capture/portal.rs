//! Donor libportal CreateSession → Start → OpenPipeWireRemote protocol.
//! Mechanical C ABI adaptation. Libraries remain optional at runtime; no
//! permission request occurs during Engine initialization or capability queries.
//! The process-lifetime function table keeps registered GTypes/callback code
//! loaded after sessions close; unloading a GObject library would be unsound.

use super::{ScreenCaptureConfig, ScreenCaptureCursor, ScreenCaptureTarget};
use crate::{Error, Result};
use libloading::Library;
use std::{
	ffi::{CStr, c_char, c_void},
	os::fd::{FromRawFd, OwnedFd},
	ptr,
};

type Ptr = *mut c_void;
type Callback = extern "C" fn(Ptr, Ptr, Ptr);
#[repr(C)]
struct GError {
	domain: u32,
	code: i32,
	message: *mut c_char,
}

struct Api {
	portal_new: unsafe extern "C" fn() -> Ptr,
	create: unsafe extern "C" fn(Ptr, u32, u32, u32, u32, *const c_char, Ptr, Callback, Ptr),
	create_finish: unsafe extern "C" fn(Ptr, Ptr, *mut *mut GError) -> Ptr,
	start: unsafe extern "C" fn(Ptr, Ptr, Ptr, Callback, Ptr),
	start_finish: unsafe extern "C" fn(Ptr, Ptr, *mut *mut GError) -> i32,
	streams: unsafe extern "C" fn(Ptr) -> Ptr,
	remote: unsafe extern "C" fn(Ptr) -> i32,
	close: unsafe extern "C" fn(Ptr),
	loop_new: unsafe extern "C" fn(Ptr, i32) -> Ptr,
	loop_run: unsafe extern "C" fn(Ptr),
	loop_quit: unsafe extern "C" fn(Ptr),
	loop_unref: unsafe extern "C" fn(Ptr),
	object_unref: unsafe extern "C" fn(Ptr),
	child: unsafe extern "C" fn(Ptr, usize) -> Ptr,
	n_children: unsafe extern "C" fn(Ptr) -> usize,
	uint32: unsafe extern "C" fn(Ptr) -> u32,
	variant_unref: unsafe extern "C" fn(Ptr),
	error_free: unsafe extern "C" fn(*mut GError),
	error_matches: unsafe extern "C" fn(*mut GError, u32, i32) -> i32,
	io_error_quark: unsafe extern "C" fn() -> u32,
	iterate: unsafe extern "C" fn(Ptr, i32) -> i32,
	connect_signal: unsafe extern "C" fn(
		Ptr,
		*const c_char,
		Option<unsafe extern "C" fn()>,
		Ptr,
		Option<unsafe extern "C" fn(Ptr, Ptr)>,
		u32,
	) -> usize,
	_libraries: [Library; 3],
}

impl Api {
	fn load() -> Result<&'static Self> {
		static API: std::sync::OnceLock<std::result::Result<Api, String>> = std::sync::OnceLock::new();
		API
			.get_or_init(|| Self::load_owned().map_err(|error| error.to_string()))
			.as_ref()
			.map_err(|error| Error::missing_capability(error.clone()))
	}
	fn load_owned() -> Result<Self> {
		// SAFETY: these are the platform's stable GLib/libportal C APIs. Each
		// symbol has its installed header signature and the libraries outlive it.
		unsafe {
			let portal =
				Library::new("libportal.so.1").map_err(|e| Error::backend_unavailable("libportal", e))?;
			let glib =
				Library::new("libglib-2.0.so.0").map_err(|e| Error::backend_unavailable("GLib", e))?;
			let object = Library::new("libgobject-2.0.so.0")
				.map_err(|e| Error::backend_unavailable("GObject", e))?;
			macro_rules! symbol {
				($lib:expr, $name:literal) => {
					*$lib
						.get(concat!($name, "\0").as_bytes())
						.map_err(|e| Error::backend_unavailable("desktop portal ABI", e))?
				};
			}
			Ok(Self {
				portal_new: symbol!(portal, "xdp_portal_new"),
				create: symbol!(portal, "xdp_portal_create_screencast_session"),
				create_finish: symbol!(portal, "xdp_portal_create_screencast_session_finish"),
				start: symbol!(portal, "xdp_session_start"),
				start_finish: symbol!(portal, "xdp_session_start_finish"),
				streams: symbol!(portal, "xdp_session_get_streams"),
				remote: symbol!(portal, "xdp_session_open_pipewire_remote"),
				close: symbol!(portal, "xdp_session_close"),
				loop_new: symbol!(glib, "g_main_loop_new"),
				loop_run: symbol!(glib, "g_main_loop_run"),
				loop_quit: symbol!(glib, "g_main_loop_quit"),
				loop_unref: symbol!(glib, "g_main_loop_unref"),
				object_unref: symbol!(object, "g_object_unref"),
				child: symbol!(glib, "g_variant_get_child_value"),
				n_children: symbol!(glib, "g_variant_n_children"),
				uint32: symbol!(glib, "g_variant_get_uint32"),
				variant_unref: symbol!(glib, "g_variant_unref"),
				error_free: symbol!(glib, "g_error_free"),
				error_matches: symbol!(glib, "g_error_matches"),
				io_error_quark: symbol!(portal, "g_io_error_quark"),
				iterate: symbol!(glib, "g_main_context_iteration"),
				connect_signal: symbol!(object, "g_signal_connect_data"),
				_libraries: [portal, glib, object],
			})
		}
	}
}

pub(super) struct Portal {
	api: &'static Api,
	portal: Ptr,
	session: Ptr,
	loop_: Ptr,
	error: *mut GError,
	started: bool,
	closed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Portal {
	pub(super) fn is_supported() -> bool {
		Api::load().is_ok()
	}
	pub(super) fn open(config: &ScreenCaptureConfig) -> Result<(Box<Self>, OwnedFd, u32)> {
		let api = Api::load()?;
		// SAFETY: no borrowed input is retained; returned objects are released
		// by Portal. The boxed callback state stays at a stable address until
		// the completion callback has returned and g_main_loop_run exits.
		unsafe {
			let mut this = Box::new(Self {
				portal: (api.portal_new)(),
				session: ptr::null_mut(),
				loop_: (api.loop_new)(ptr::null_mut(), 0),
				error: ptr::null_mut(),
				started: false,
				closed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
				api,
			});
			if this.portal.is_null() || this.loop_.is_null() {
				return Err(Error::unavailable("desktop portal initialization failed"));
			}
			let output = match config.target {
				ScreenCaptureTarget::MonitorOrWindow => 3,
				ScreenCaptureTarget::Monitor => 1,
				ScreenCaptureTarget::Window => 2,
			};
			let cursor = match config.cursor {
				ScreenCaptureCursor::Hidden => 1,
				ScreenCaptureCursor::Embedded => 2,
			};
			(this.api.create)(
				this.portal,
				output,
				0,
				cursor,
				0,
				ptr::null(),
				ptr::null_mut(),
				created,
				(&mut *this as *mut Self).cast(),
			);
			(this.api.loop_run)(this.loop_);
			if !this.error.is_null() {
				let message = CStr::from_ptr((*this.error).message)
					.to_string_lossy()
					.into_owned();
				if (this.api.error_matches)(this.error, (this.api.io_error_quark)(), 19) != 0 {
					return Err(Error::cancelled(format!(
						"screen capture selection: {message}"
					)));
				}
				if (this.api.error_matches)(this.error, (this.api.io_error_quark)(), 14) != 0 {
					return Err(Error::permission_denied(format!(
						"screen capture portal: {message}"
					)));
				}
				return Err(Error::backend_failure(
					"desktop portal",
					"create/start screen capture",
					std::io::Error::other(message),
				));
			}
			if !this.started {
				return Err(Error::cancelled("screen capture selection was cancelled"));
			}
			// GCallback erases its signature; the session's closed signal has
			// exactly (instance, user_data). The signal owns its own Arc until
			// GObject destroys the handler, even if an asynchronous close retains
			// the session beyond this Rust owner's lifetime.
			let callback = std::mem::transmute::<extern "C" fn(Ptr, Ptr), unsafe extern "C" fn()>(closed);
			let flag = std::sync::Arc::into_raw(this.closed.clone())
				.cast_mut()
				.cast();
			let handler = (this.api.connect_signal)(
				this.session,
				c"closed".as_ptr(),
				Some(callback),
				flag,
				Some(release_closed),
				0,
			);
			if handler == 0 {
				release_closed(flag, ptr::null_mut());
				return Err(Error::unavailable(
					"could not observe portal session closure",
				));
			}

			let streams = (this.api.streams)(this.session);
			if streams.is_null() || (this.api.n_children)(streams) != 1 {
				return Err(Error::data_loss(
					"portal must return exactly one screen stream",
				));
			}
			let stream = (this.api.child)(streams, 0);
			let node = (this.api.child)(stream, 0);
			let id = (this.api.uint32)(node);
			(this.api.variant_unref)(node);
			(this.api.variant_unref)(stream);
			let fd = (this.api.remote)(this.session);
			if fd < 0 {
				return Err(Error::unavailable(
					"portal did not provide a PipeWire remote",
				));
			}
			Ok((this, OwnedFd::from_raw_fd(fd), id))
		}
	}

	pub(super) fn dispatch(&self) -> Result<()> {
		// SAFETY: this session and its GLib context belong to the calling
		// thread. A zero wait dispatches notifications without blocking.
		unsafe {
			(self.api.iterate)(ptr::null_mut(), 0);
		}
		if self.closed.load(std::sync::atomic::Ordering::Acquire) {
			return Err(Error::unavailable("desktop screen sharing was stopped"));
		}
		Ok(())
	}
}

extern "C" fn closed(_: Ptr, data: Ptr) {
	// SAFETY: user_data is the signal-owned Arc pointer. The atomic flag is
	// independent of Portal and remains valid through asynchronous closure.
	unsafe {
		(*data.cast::<std::sync::atomic::AtomicBool>())
			.store(true, std::sync::atomic::Ordering::Release);
	}
}
unsafe extern "C" fn release_closed(data: Ptr, _: Ptr) {
	// SAFETY: GObject invokes destroy_notify once for the one Arc transferred
	// to this signal handler. It never contains Engine or GPU state.
	unsafe {
		drop(std::sync::Arc::from_raw(
			data.cast::<std::sync::atomic::AtomicBool>(),
		));
	}
}

extern "C" fn created(_: Ptr, result: Ptr, data: Ptr) {
	// SAFETY: libportal invokes this once, on the loop owning the boxed Portal;
	// result is borrowed for this callback and finish transfers the session.
	unsafe {
		let this = &mut *data.cast::<Portal>();
		this.session = (this.api.create_finish)(this.portal, result, &mut this.error);
		if this.session.is_null() {
			(this.api.loop_quit)(this.loop_);
		} else {
			(this.api.start)(
				this.session,
				ptr::null_mut(),
				ptr::null_mut(),
				started,
				data,
			);
		}
	}
}
extern "C" fn started(_: Ptr, result: Ptr, data: Ptr) {
	// SAFETY: same pinned callback ownership as created; finish consumes only
	// the asynchronous result. No Rust reference escapes the callback.
	unsafe {
		let this = &mut *data.cast::<Portal>();
		this.started = (this.api.start_finish)(this.session, result, &mut this.error) != 0;
		(this.api.loop_quit)(this.loop_);
	}
}
impl Drop for Portal {
	fn drop(&mut self) {
		// SAFETY: these objects are owned once, callbacks finished before open
		// returns, and libraries remain loaded through this destructor. Closing
		// the portal releases host protocol state; no GPU work or waits occur.
		unsafe {
			if !self.session.is_null() {
				(self.api.close)(self.session);
				(self.api.object_unref)(self.session);
			}
			if !self.portal.is_null() {
				(self.api.object_unref)(self.portal);
			}
			if !self.loop_.is_null() {
				(self.api.loop_unref)(self.loop_);
			}
			if !self.error.is_null() {
				(self.api.error_free)(self.error);
			}
		}
	}
}
