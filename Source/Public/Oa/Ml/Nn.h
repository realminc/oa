// OA ML - Neural Network Layers
//
// Standard building blocks, all inheriting from OaModule.
// PyTorch torch.nn equivalent with OA naming.
//
// Layers dispatch to Vulkan compute. No CPU fallbacks.

#pragma once

#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/NnType.h>

// GENERATED LAYERS
// Regenerate via: python3 Tools/NnAutogen/oannautogen.py --live
#include <Oa/Ml/Nn/Layers.h>

// MANUAL LAYERS

// DROPOUT
#include <Oa/Ml/Nn/Dropout/Dropout.h>

// ROTARY POSITION EMBEDDING (RoPE)
#include <Oa/Ml/Nn/Rope/Rope.h>

// MOE (Mixture of Experts)
#include <Oa/Ml/Nn/Moe/Moe.h>

// FFN (Feed-Forward Network with SwiGLU)
#include <Oa/Ml/Nn/Ffn/Ffn.h>

// GRU (gated recurrent unit) cell + stacked sequence module
#include <Oa/Ml/Nn/Gru/Gru.h>

// RNN (vanilla Elman) cell + stacked sequence module
#include <Oa/Ml/Nn/Rnn/Rnn.h>

// MAMBA-3 (Selective State Space Model)
#include <Oa/Ml/Nn/Mamba3/Mamba3.h>

// TRANSFORMER — pre-norm transformer block with causal or bidirectional self-attention
#include <Oa/Ml/Nn/Transformer/Transformer.h>

// FLOW — time embedding plus bidirectional Transformer denoiser family
#include <Oa/Ml/Nn/Flow/FlowTimeEmbedding.h>
#include <Oa/Ml/Nn/Flow/FlowTransformer.h>
#include <Oa/Ml/Nn/Flow/FlowDenoiser.h>

// EMPYREALM CORE — high-utilization sequential backbone
// (byte + mixer + flat residual pattern, designed for subclassing/specialization
//  across text, motion sequences, audio, etc.)
#include <Oa/Ml/Nn/Empyrealm/EmpyrealmCore.h>

// VECTOR QUANTIZATION (VQ-VAE / RVQ discrete bottleneck) — modality-agnostic
// tokenizer front-end for discrete-token generation (MoMask / T2M-GPT). EMA
// codebook, checkpointed via Buffers().
#include <Oa/Ml/Nn/Vq/Vq.h>

// Legacy sequence/conv VQ tokenizer experiments are surfaced via <Ml/Nn.h>.

// OaAlm (the skeletal-motion pipeline: Conv1d VQ-VAE tokenizer + AR token LM)
// and its PoseClip codec live in Extensions/Public/Ml/Nn/Alm/ and are surfaced
// via <Ml/Nn.h> — not from core, since they're a concrete model line.
