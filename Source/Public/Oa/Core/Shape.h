#pragma once

// Renamed to MatrixShape.h (type OaShape -> OaMatrixShape). This shim keeps any
// stragglers compiling; delete it once nothing includes <Oa/Core/Shape.h>:
//   git rm Source/Public/Oa/Core/Shape.h
#include <Oa/Core/MatrixShape.h>
