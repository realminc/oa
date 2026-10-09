//! Blocking CPU network transport primitives.
//!
//! TCP carries ordered bytes. [`TcpFramed`] adds bounded message boundaries;
//! neither layer authenticates peers or interprets application payloads.
//! [`McpServer`] provides a transport-independent JSON-RPC 2.0 Model Context
//! Protocol control plane.
//!
//! Donor: `oa/network/tcp.h`, `tcpFramed.h`, `mcp.h`, and their implementations.

mod framing;
pub mod mcp;
mod tcp;

pub use framing::TcpFramed;
pub use mcp::{
	MCP_LATEST_PROTOCOL_VERSION, McpArguments, McpCacheScope, McpServer, McpServerConfig,
	McpTextResource, McpTool, McpToolResult,
};
pub use tcp::{TcpListener, TcpStream};
