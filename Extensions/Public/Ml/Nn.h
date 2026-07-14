#pragma once

// ML neural network modules — extends oa/ml with model architectures.
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Nn.h>

#include <Ml/Nn/GptOss/GptOss.h>
#include <Ml/Nn/YoloV11/YoloV11.h>

// PoseClip .3danim codec — skeletal-motion clip IO.
#include <Anim/PoseClip.h>

// OaAlmAg — complete autograd motion pipeline: Conv1d VQ-VAE tokenizer plus a
// caption-conditioned causal Transformer with pluggable dense/MoE FFNs.
#include <Ml/Nn/Alm/AlmConfig.h>
#include <Ml/Nn/Alm/AlmAg.h>
#include <Ml/Nn/Alm/ClipTextAg.h>
#include <Ml/Nn/Alm/ClipTextWeightAdapter.h>
#include <Ml/Nn/Alm/ClipTokenizer.h>
#include <Ml/Nn/Alm/AlmTokenizerAg.h>
#include <Ml/Nn/Alm/AlmPriorAg.h>
