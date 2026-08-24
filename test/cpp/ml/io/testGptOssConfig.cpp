#include <ml/nn/gptOss/gptOss.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

TEST(GptOssConfig, PublishedPresets) {
	const auto m20 = oa::GptOssConfig::preset20B();
	EXPECT_TRUE(m20.validate().isOk());
	EXPECT_TRUE(m20.isPublished20B());
	EXPECT_EQ(m20.expectedLogicalWeightCount(), 459);
	EXPECT_TRUE(m20.layerUsesSlidingAttention(0));
	EXPECT_FALSE(m20.layerUsesSlidingAttention(1));

	const auto m120 = oa::GptOssConfig::preset120B();
	EXPECT_TRUE(m120.validate().isOk());
	EXPECT_TRUE(m120.isPublished120B());
	EXPECT_EQ(m120.expectedLogicalWeightCount(), 687);
}

TEST(GptOssConfig, ParsesExactPublishedContract) {
	const oa::String path = "/tmp/oa_gpt_oss_config_test.json";
	std::ofstream out(path.cStr());
	out << R"({
  "model_type":"gpt_oss", "attention_bias":true,
  "tie_word_embeddings":false, "vocab_size":201088,
  "num_hidden_layers":2, "hidden_size":2880, "intermediate_size":2880,
  "num_attention_heads":64, "num_key_value_heads":8, "head_dim":64,
  "num_local_experts":32, "num_experts_per_tok":4,
  "sliding_window":128, "initial_context_length":4096,
  "max_position_embeddings":131072, "rope_theta":150000,
  "rms_norm_eps":0.00001, "swiglu_limit":7,
  "pad_token_id":199999, "eos_token_id":200002,
  "layer_types":["sliding_attention","full_attention"],
  "rope_scaling":{"factor":32,"beta_slow":1,"beta_fast":32},
  "quantization_config":{"quant_method":"mxfp4"}
})";
	out.close();
	auto parsed = oa::GptOssConfig::fromJson(path);
	ASSERT_TRUE(parsed.isOk()) << parsed.getStatus().toString().cStr();
	EXPECT_EQ(parsed->numLayers, 2);
	EXPECT_EQ(parsed->numExperts, 32);
	EXPECT_EQ(parsed->queryWidth(), 4096);
	EXPECT_EQ(parsed->kvWidth(), 512);
	std::remove(path.cStr());
}

TEST(GptOssConfig, RejectsWrongLayerSchedule) {
	const oa::String path = "/tmp/oa_gpt_oss_bad_config_test.json";
	std::ofstream out(path.cStr());
	out << R"({
  "model_type":"gpt_oss", "attention_bias":true,
  "tie_word_embeddings":false, "vocab_size":201088,
  "num_hidden_layers":2, "hidden_size":2880, "intermediate_size":2880,
  "num_attention_heads":64, "num_key_value_heads":8, "head_dim":64,
  "num_local_experts":32, "num_experts_per_tok":4,
  "sliding_window":128, "initial_context_length":4096,
  "max_position_embeddings":131072, "rope_theta":150000,
  "rms_norm_eps":0.00001, "swiglu_limit":7,
  "pad_token_id":199999, "eos_token_id":200002,
  "layer_types":["full_attention","full_attention"],
  "rope_scaling":{"factor":32,"beta_slow":1,"beta_fast":32},
  "quantization_config":{"quant_method":"mxfp4"}
})";
	out.close();
	auto parsed = oa::GptOssConfig::fromJson(path);
	EXPECT_TRUE(parsed.isError());
	std::remove(path.cStr());
}
