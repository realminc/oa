//! Private external-weight container adapters.
//!
//! Model-specific translators consume these format-neutral views and emit the
//! one native `.oam` artifact. Container types do not enter the public API.

mod safe_tensors;

pub(crate) use safe_tensors::SafeTensorsSource;
