#include <Oa/Runtime/Scheduler.h>
#include <Oa/Runtime/Topology.h>

OaScheduler OaScheduler::Create(OaDeviceMesh& InMesh) {
	OaScheduler sched;
	sched.Mesh = &InMesh;
	return sched;
}

OaDeviceNode* OaScheduler::Route(const OaDispatchHint& InHint) {
	if (!Mesh || Mesh->NodeCount() == 0) {
		return nullptr;
	}
	// Explicit node override
	if (InHint.PreferNode != OA_NODE_AUTO) {
		auto* node = Mesh->GetNode(InHint.PreferNode);
		if (node) {
			return node;
		}
	}
	return Route(InHint.Class);
}

OaDeviceNode* OaScheduler::Route(OaComputeClass InClass) {
	if (!Mesh || Mesh->NodeCount() == 0) return nullptr;
	if (Mesh->NodeCount() == 1) return Mesh->Primary;

	if (InClass == OaComputeClass::Light) {
		auto* aux = Mesh->GetByRole(OaDeviceRole::Auxiliary);
		return aux ? aux : Mesh->Primary;
	}
	return Mesh->Primary;
}
