//! Public-data fixed-width unsigned device arithmetic.
//!
//! U32 Matrix rows encode little-endian limbs. Operations preserve their Engine
//! ownership and return device values without host observation. These ordinary
//! Matrix APIs are not the secret-device execution boundary.

#[path = "bigint/arithmetic.gen.rs"]
mod arithmetic;
#[path = "bigint/parity.gen.rs"]
mod parity;

include!("bigint/exports.gen.rs");
