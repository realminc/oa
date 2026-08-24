// OA ML - Neural Network layers
//
// Standard building blocks, all inheriting from oa::Module.
// PyTorch torch.nn equivalent with OA naming.
//
// layers dispatch to vulkan compute. No CPU fallbacks.

#pragma once

#include <oa/ml/fnMatrix.h>
#include <oa/ml/module.h>
#include <oa/ml/nn/attention.h>

// GENERATED LAYERS
// Regenerate via: python3 Tools/NnAutogen/oannautogen.py --live
#include <oa/ml/nn/layers.h>

// MANUAL LAYERS

// DROPOUT
#include <oa/ml/nn/dropout/dropout.h>

// ROTARY POSITION EMBEDDING (RoPE)
#include <oa/ml/nn/rope/rope.h>

// MOE (Mixture of Experts)
#include <oa/ml/nn/moe/moe.h>

// FFN (Feed-forward Network with SwiGLU)
#include <oa/ml/nn/ffn/ffn.h>

// GRU (gated recurrent unit) cell + stacked sequence module
#include <oa/ml/nn/gru/gru.h>

// RNN (vanilla Elman) cell + stacked sequence module
#include <oa/ml/nn/rnn/rnn.h>

// MAMBA-3 (Selective State Space Model)
#include <oa/ml/nn/mamba3/mamba3.h>

// TRANSFORMER — pre-norm transformer block with causal or bidirectional self-attention
#include <oa/ml/nn/transformer/transformer.h>

// FLOW — time embedding plus bidirectional Transformer denoiser family
#include <oa/ml/nn/flow/flowTimeEmbedding.h>
#include <oa/ml/nn/flow/flowTransformer.h>
#include <oa/ml/nn/flow/flowDenoiser.h>

// EMPYREALM CORE — high-utilization sequential backbone
// (byte + mixer + flat residual pattern, designed for subclassing/specialization
//  across text, motion sequences, audio, etc.)
#include <oa/ml/nn/empyrealm/empyrealmCore.h>

// VECTOR QUANTIZATION (VQ-VAE / RVQ discrete bottleneck) — modality-agnostic
// tokenizer front-end for discrete-token generation (MoMask / T2M-GPT). EMA
// codebook, checkpointed via buffers().
#include <oa/ml/nn/vq/vq.h>

// oa::Alm (the skeletal-motion pipeline: Conv1d VQ-VAE tokenizer + AR token LM)
// and its PoseClip codec live in sdk/cpp/include/ml/nn/alm and are surfaced
// through the repository-local SDK support target, not from core, since they
// are a concrete model line.
