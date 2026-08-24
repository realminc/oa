#pragma once

// ML neural network modules — extends oa/ml with model architectures.
#include <oa/ml/module.h>
#include <oa/ml/nn.h>

#include <ml/nn/gptOss/gptOss.h>
#include <ml/nn/yoloV11/yoloV11.h>

// PoseClip .3danim codec — skeletal-motion clip IO.
#include <anim/poseClip.h>

// oa::AlmAg — complete autograd motion pipeline: Conv1d VQ-VAE tokenizer plus a
// caption-conditioned causal Transformer with pluggable dense/MoE FFNs.
#include <ml/nn/alm/almConfig.h>
#include <ml/nn/alm/almAg.h>
#include <ml/nn/alm/clipTextAg.h>
#include <ml/nn/alm/clipTextModelTranslator.h>
#include <ml/nn/alm/clipTokenizer.h>
#include <ml/nn/alm/almTokenizerAg.h>
#include <ml/nn/alm/almPriorAg.h>
