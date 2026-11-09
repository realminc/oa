use super::*;
use crate::runtime::dna::kernel::KernelId;

#[test]
fn secret_kernel_cannot_enter_ordinary_graph_recording() {
	for kernel in [
		KernelId::CryptographyMlKemKeygenU8,
		KernelId::CryptographyMlDsaKeygenU8,
		KernelId::CryptographyMlDsaSignU8,
		KernelId::CryptographyMlDsaSignMessageU8,
		KernelId::CryptographyMlDsaSignPrehashedU8,
		KernelId::CryptographyMlDsaSignHashMessageU8,
		KernelId::CryptographyMlKemEncapsU8,
		KernelId::CryptographyMlKemDecapsU8,
	] {
		let node = ComputeNode {
			operation: "test rejected ordinary PQC route",
			kernel,
			buffers: Vec::new(),
			push_constants: Vec::new(),
			workgroups: [1, 1, 1],
			indirect: None,
			semantic_ops: Vec::new(),
			op_contract_hash: 0,
			jit: None,
			fusion_signature: None,
		};
		assert!(matches!(reject_secret_node(&node), Err(error)
			if error.kind() == crate::ErrorKind::FailedPrecondition));
	}
}
