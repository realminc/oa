#pragma once

// OA scalar math — camelCase wrappers over <cmath>.
//
// HONEST NOTE: sqrt/sin/exp/… are libm routines mapped to hardware. There is no
// clean-room replacement; these only provide the OA-consistent naming (matching
// oa::min/oa::sort in algo.h. Float and
// double overloads so the right precision is picked without casts.

#include <cmath>

namespace oa {

// ── Roots / powers ──────────────────────────────────────────────────────────
[[nodiscard]] inline float  cbrt(float inX)          { return std::cbrt(inX); }
[[nodiscard]] inline double cbrt(double inX)         { return std::cbrt(inX); }
[[nodiscard]] inline float  pow(float inB, float inE)    { return std::pow(inB, inE); }
[[nodiscard]] inline double pow(double inB, double inE)  { return std::pow(inB, inE); }

// ── Exp / log ───────────────────────────────────────────────────────────────
[[nodiscard]] inline float  exp(float inX)          { return std::exp(inX); }
[[nodiscard]] inline double exp(double inX)         { return std::exp(inX); }
[[nodiscard]] inline float  exp2(float inX)         { return std::exp2(inX); }
[[nodiscard]] inline double exp2(double inX)        { return std::exp2(inX); }
[[nodiscard]] inline float  log(float inX)          { return std::log(inX); }
[[nodiscard]] inline double log(double inX)         { return std::log(inX); }
[[nodiscard]] inline float  log2(float inX)         { return std::log2(inX); }
[[nodiscard]] inline double log2(double inX)        { return std::log2(inX); }
[[nodiscard]] inline float  log10(float inX)        { return std::log10(inX); }
[[nodiscard]] inline double log10(double inX)       { return std::log10(inX); }

// ── Trig ────────────────────────────────────────────────────────────────────
[[nodiscard]] inline float  sin(float inX)          { return std::sin(inX); }
[[nodiscard]] inline double sin(double inX)         { return std::sin(inX); }
[[nodiscard]] inline float  cos(float inX)          { return std::cos(inX); }
[[nodiscard]] inline double cos(double inX)         { return std::cos(inX); }
[[nodiscard]] inline float  tan(float inX)          { return std::tan(inX); }
[[nodiscard]] inline double tan(double inX)         { return std::tan(inX); }
[[nodiscard]] inline float  asin(float inX)         { return std::asin(inX); }
[[nodiscard]] inline double asin(double inX)        { return std::asin(inX); }
[[nodiscard]] inline float  acos(float inX)         { return std::acos(inX); }
[[nodiscard]] inline double acos(double inX)        { return std::acos(inX); }
[[nodiscard]] inline float  atan(float inX)         { return std::atan(inX); }
[[nodiscard]] inline double atan(double inX)        { return std::atan(inX); }
[[nodiscard]] inline float  atan2(float inY, float inX)   { return std::atan2(inY, inX); }
[[nodiscard]] inline double atan2(double inY, double inX) { return std::atan2(inY, inX); }
[[nodiscard]] inline float  tanh(float inX)         { return std::tanh(inX); }
[[nodiscard]] inline double tanh(double inX)        { return std::tanh(inX); }

// ── Rounding / sign / abs ───────────────────────────────────────────────────
[[nodiscard]] inline float  floor(float inX)        { return std::floor(inX); }
[[nodiscard]] inline double floor(double inX)       { return std::floor(inX); }
[[nodiscard]] inline float  ceil(float inX)         { return std::ceil(inX); }
[[nodiscard]] inline double ceil(double inX)        { return std::ceil(inX); }
[[nodiscard]] inline float  round(float inX)        { return std::round(inX); }
[[nodiscard]] inline double round(double inX)       { return std::round(inX); }
[[nodiscard]] inline float  trunc(float inX)        { return std::trunc(inX); }
[[nodiscard]] inline double trunc(double inX)       { return std::trunc(inX); }
[[nodiscard]] inline float  fmod(float inA, float inB)   { return std::fmod(inA, inB); }
[[nodiscard]] inline double fmod(double inA, double inB) { return std::fmod(inA, inB); }

[[nodiscard]] inline float     abs(float inX)     { return std::fabs(inX); }
[[nodiscard]] inline double    abs(double inX)    { return std::fabs(inX); }
[[nodiscard]] inline int       abs(int inX)       { return inX < 0 ? -inX : inX; }
[[nodiscard]] inline long long abs(long long inX) { return inX < 0 ? -inX : inX; }

// ── Classification ──────────────────────────────────────────────────────────
[[nodiscard]] inline bool isNan(float inX)     { return std::isnan(inX); }
[[nodiscard]] inline bool isNan(double inX)    { return std::isnan(inX); }
[[nodiscard]] inline bool isInf(float inX)     { return std::isinf(inX); }
[[nodiscard]] inline bool isInf(double inX)    { return std::isinf(inX); }
[[nodiscard]] inline bool isFinite(float inX)  { return std::isfinite(inX); }
[[nodiscard]] inline bool isFinite(double inX) { return std::isfinite(inX); }

} // namespace oa
