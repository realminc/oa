// OaCluster — Multi-node distributed compute (stubs)
//
// Coordinates ranks across machines while borrowing this process's local mesh.
// It does not wrap/own OaComputeEngine or create an OaContext. Real networking
// lives in oa/network/ (TCP) or a future RDMA transport.
// Header-only stubs — no .cpp until network transport is implemented.

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Allocator.h>

class OaDeviceMesh;

// Remote machine info.
class OaClusterNode {
public:
	OaU32 Id = 0;
	OaString Address;
	OaU16 Port = 0;
	OaU32 DeviceCount = 0;     // GPUs on that machine
	OaU64 TotalVramBytes = 0;  // Aggregate VRAM
	OaBool IsLocal = false;
};

class OaClusterConfig {
public:
	OaVec<OaString> NodeAddresses;  // "host:port" strings
	OaU16 ListenPort = 9100;
	OaU32 RankId = 0;               // This machine's rank
};

// Multi-machine coordination companion.
// Stub: all methods return NotImplemented. Real transport in Phase 3.
class OaCluster {
public:
	OaVec<OaClusterNode> Nodes;
	OaDeviceMesh* LocalMesh = nullptr;
	OaU32 Rank = 0;

	[[nodiscard]] static OaResult<OaCluster> Create(
		const OaClusterConfig& InConfig,
		OaDeviceMesh& InLocalMesh) {
		(void)InConfig;
		(void)InLocalMesh;
		return OaStatus::Unimplemented("OaCluster: multi-node not yet implemented");
	}

	void Destroy() {}

	[[nodiscard]] OaU32 NodeCount() const { return static_cast<OaU32>(Nodes.Size()); }
	[[nodiscard]] OaU32 WorldSize() const { return NodeCount(); }

	// Send buffer to remote node (stub).
	[[nodiscard]] OaStatus Send(OaU32 InDstRank, const OaVkBuffer& InBuf) {
		(void)InDstRank;
		(void)InBuf;
		return OaStatus::Unimplemented("OaCluster::Send");
	}

	// Receive buffer from remote node (stub).
	[[nodiscard]] OaStatus Recv(OaU32 InSrcRank, OaVkBuffer& OutBuf) {
		(void)InSrcRank;
		(void)OutBuf;
		return OaStatus::Unimplemented("OaCluster::Recv");
	}
};
