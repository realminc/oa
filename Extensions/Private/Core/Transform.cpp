#include <Core/Transform.h>

#include <cmath>

VlmQuat VlmQuatMul(const VlmQuat& A, const VlmQuat& B) noexcept {
	return {
		A.W * B.X + A.X * B.W + A.Y * B.Z - A.Z * B.Y,
		A.W * B.Y - A.X * B.Z + A.Y * B.W + A.Z * B.X,
		A.W * B.Z + A.X * B.Y - A.Y * B.X + A.Z * B.W,
		A.W * B.W - A.X * B.X - A.Y * B.Y - A.Z * B.Z,
	};
}

VlmQuat VlmQuatNormalize(const VlmQuat& Q) noexcept {
	const OaF32 n = std::sqrt(Q.X * Q.X + Q.Y * Q.Y + Q.Z * Q.Z + Q.W * Q.W);
	if (n < 1e-12f) {
		return { 0.0f, 0.0f, 0.0f, 1.0f };
	}
	const OaF32 inv = 1.0f / n;
	return { Q.X * inv, Q.Y * inv, Q.Z * inv, Q.W * inv };
}

VlmQuat OaEulerXyzDegToQuat(const VlmVec3& InDegXyz) noexcept {
	const OaF32 deg2rad = Vlm::PI / 180.0f;
	const OaF32 hx = InDegXyz.X * deg2rad * 0.5f;
	const OaF32 hy = InDegXyz.Y * deg2rad * 0.5f;
	const OaF32 hz = InDegXyz.Z * deg2rad * 0.5f;
	const VlmQuat qx = { std::sin(hx), 0.0f, 0.0f, std::cos(hx) };
	const VlmQuat qy = { 0.0f, std::sin(hy), 0.0f, std::cos(hy) };
	const VlmQuat qz = { 0.0f, 0.0f, std::sin(hz), std::cos(hz) };
	// Apply X first, then Y, then Z  ⇒  qz ⊗ qy ⊗ qx.
	return VlmQuatMul(qz, VlmQuatMul(qy, qx));
}

VlmMat4 OaTrsMatrix(const VlmVec3& InScale, const VlmQuat& InRot, const VlmVec3& InTrans) noexcept {
	// QuaternionToMatrix gives the column-vector rotation R (v_rot = R · v).
	// For the row-vector convention used here (v' = v · M) the 3×3 block is Rᵀ,
	// with the per-axis scale folded into the rows.
	const VlmMat4 r = Vlm::QuaternionToMatrix(InRot);
	const OaF32 s[3] = { InScale.X, InScale.Y, InScale.Z };
	VlmMat4 m = VlmMat4::Identity();
	for (OaI32 i = 0; i < 3; ++i) {
		for (OaI32 k = 0; k < 3; ++k) {
			m.M[i][k] = s[i] * r.M[k][i];   // transpose: row i, col k ← R[k][i]
		}
	}
	m.M[3][0] = InTrans.X;
	m.M[3][1] = InTrans.Y;
	m.M[3][2] = InTrans.Z;
	return m;
}

VlmVec3 OaTransformPoint(const VlmMat4& InM, const VlmVec3& InP) noexcept {
	return {
		InP.X * InM.M[0][0] + InP.Y * InM.M[1][0] + InP.Z * InM.M[2][0] + InM.M[3][0],
		InP.X * InM.M[0][1] + InP.Y * InM.M[1][1] + InP.Z * InM.M[2][1] + InM.M[3][1],
		InP.X * InM.M[0][2] + InP.Y * InM.M[1][2] + InP.Z * InM.M[2][2] + InM.M[3][2],
	};
}
