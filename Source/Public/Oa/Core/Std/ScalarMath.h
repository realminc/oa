#pragma once

// OaStd scalar math — PascalCase wrappers over <cmath>.
//
// HONEST NOTE: sqrt/sin/exp/… are libm routines mapped to hardware. There is no
// clean-room replacement; these only provide the OA-consistent naming (matching
// OaStdMin/OaStdSort in Algo.h — canonical OaStd*, no short alias). float and
// double overloads so the right precision is picked without casts.

#include <cmath>

// ── Roots / powers ──────────────────────────────────────────────────────────
[[nodiscard]] inline float  OaStdSqrt(float InX)          { return std::sqrt(InX); }
[[nodiscard]] inline double OaStdSqrt(double InX)         { return std::sqrt(InX); }
[[nodiscard]] inline float  OaStdCbrt(float InX)          { return std::cbrt(InX); }
[[nodiscard]] inline double OaStdCbrt(double InX)         { return std::cbrt(InX); }
[[nodiscard]] inline float  OaStdPow(float InB, float InE)    { return std::pow(InB, InE); }
[[nodiscard]] inline double OaStdPow(double InB, double InE)  { return std::pow(InB, InE); }

// ── Exp / log ───────────────────────────────────────────────────────────────
[[nodiscard]] inline float  OaStdExp(float InX)          { return std::exp(InX); }
[[nodiscard]] inline double OaStdExp(double InX)         { return std::exp(InX); }
[[nodiscard]] inline float  OaStdExp2(float InX)         { return std::exp2(InX); }
[[nodiscard]] inline double OaStdExp2(double InX)        { return std::exp2(InX); }
[[nodiscard]] inline float  OaStdLog(float InX)          { return std::log(InX); }
[[nodiscard]] inline double OaStdLog(double InX)         { return std::log(InX); }
[[nodiscard]] inline float  OaStdLog2(float InX)         { return std::log2(InX); }
[[nodiscard]] inline double OaStdLog2(double InX)        { return std::log2(InX); }
[[nodiscard]] inline float  OaStdLog10(float InX)        { return std::log10(InX); }
[[nodiscard]] inline double OaStdLog10(double InX)       { return std::log10(InX); }

// ── Trig ────────────────────────────────────────────────────────────────────
[[nodiscard]] inline float  OaStdSin(float InX)          { return std::sin(InX); }
[[nodiscard]] inline double OaStdSin(double InX)         { return std::sin(InX); }
[[nodiscard]] inline float  OaStdCos(float InX)          { return std::cos(InX); }
[[nodiscard]] inline double OaStdCos(double InX)         { return std::cos(InX); }
[[nodiscard]] inline float  OaStdTan(float InX)          { return std::tan(InX); }
[[nodiscard]] inline double OaStdTan(double InX)         { return std::tan(InX); }
[[nodiscard]] inline float  OaStdAsin(float InX)         { return std::asin(InX); }
[[nodiscard]] inline double OaStdAsin(double InX)        { return std::asin(InX); }
[[nodiscard]] inline float  OaStdAcos(float InX)         { return std::acos(InX); }
[[nodiscard]] inline double OaStdAcos(double InX)        { return std::acos(InX); }
[[nodiscard]] inline float  OaStdAtan(float InX)         { return std::atan(InX); }
[[nodiscard]] inline double OaStdAtan(double InX)        { return std::atan(InX); }
[[nodiscard]] inline float  OaStdAtan2(float InY, float InX)   { return std::atan2(InY, InX); }
[[nodiscard]] inline double OaStdAtan2(double InY, double InX) { return std::atan2(InY, InX); }
[[nodiscard]] inline float  OaStdTanh(float InX)         { return std::tanh(InX); }
[[nodiscard]] inline double OaStdTanh(double InX)        { return std::tanh(InX); }

// ── Rounding / sign / abs ───────────────────────────────────────────────────
[[nodiscard]] inline float  OaStdFloor(float InX)        { return std::floor(InX); }
[[nodiscard]] inline double OaStdFloor(double InX)       { return std::floor(InX); }
[[nodiscard]] inline float  OaStdCeil(float InX)         { return std::ceil(InX); }
[[nodiscard]] inline double OaStdCeil(double InX)        { return std::ceil(InX); }
[[nodiscard]] inline float  OaStdRound(float InX)        { return std::round(InX); }
[[nodiscard]] inline double OaStdRound(double InX)       { return std::round(InX); }
[[nodiscard]] inline float  OaStdTrunc(float InX)        { return std::trunc(InX); }
[[nodiscard]] inline double OaStdTrunc(double InX)       { return std::trunc(InX); }
[[nodiscard]] inline float  OaStdFmod(float InA, float InB)   { return std::fmod(InA, InB); }
[[nodiscard]] inline double OaStdFmod(double InA, double InB) { return std::fmod(InA, InB); }

[[nodiscard]] inline float     OaStdAbs(float InX)     { return std::fabs(InX); }
[[nodiscard]] inline double    OaStdAbs(double InX)    { return std::fabs(InX); }
[[nodiscard]] inline int       OaStdAbs(int InX)       { return InX < 0 ? -InX : InX; }
[[nodiscard]] inline long long OaStdAbs(long long InX) { return InX < 0 ? -InX : InX; }

// ── Classification ──────────────────────────────────────────────────────────
[[nodiscard]] inline bool OaStdIsNan(float InX)     { return std::isnan(InX); }
[[nodiscard]] inline bool OaStdIsNan(double InX)    { return std::isnan(InX); }
[[nodiscard]] inline bool OaStdIsInf(float InX)     { return std::isinf(InX); }
[[nodiscard]] inline bool OaStdIsInf(double InX)    { return std::isinf(InX); }
[[nodiscard]] inline bool OaStdIsFinite(float InX)  { return std::isfinite(InX); }
[[nodiscard]] inline bool OaStdIsFinite(double InX) { return std::isfinite(InX); }
