use super::*;
use crate::runtime::shader::KernelId;

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
			semantic_ops: Vec::new(),
			op_contract_hash: 0,
		};
		for profile in [ExecutionProfile::Strict, ExecutionProfile::Compatibility] {
			assert!(matches!(prepare_dispatch(&[], &node, profile), Err(error)
			if error.kind() == crate::ErrorKind::FailedPrecondition));
		}
	}
}
