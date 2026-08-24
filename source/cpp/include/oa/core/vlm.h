#pragma once

// VLM — OA's Vulkan-native linear math contract.
//
// Spatial coordinates are right-handed, +X right, +Y up, and camera forward
// -Z. Matrices use row-major storage and row-vector multiplication so CPU
// values match OA's explicit `row_major` Slang ABI without transposition.
// Projection helpers emit Vulkan NDC depth [0, 1]. Raster Y orientation is
// selected once by the Vulkan viewport; it is never hidden in world matrices.
// External formats convert units, axes, handedness, and quaternion order once
// at their import/export boundary.
//
// Operators are ergonomic wrappers over the named `oa::vlm` primitives. The
// named primitive owns each formula, so operator and function syntax cannot
// drift. Homogeneous transforms use `Vec4 * Mat4` or `transform(value, matrix)`;
// `Vec3 * Mat4` is intentionally absent because it cannot infer point versus
// direction semantics.

#include <oa/core/vlm/vector.h>
#include <oa/core/vlm/quaternion.h>
#include <oa/core/vlm/matrix.h>
#include <oa/core/vlm/math.h>
