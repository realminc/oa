#include <Ml/Nn/GptOss/GptOss.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

TEST(GptOssConfig, PublishedPresets) {
	const auto m20 = OaGptOssConfig::Preset20B();
	EXPECT_TRUE(m20.Validate().IsOk());
	EXPECT_TRUE(m20.IsPublished20B());
	EXPECT_EQ(m20.ExpectedLogicalWeightCount(), 459);
	EXPECT_TRUE(m20.LayerUsesSlidingAttention(0));
	EXPECT_FALSE(m20.LayerUsesSlidingAttention(1));

	const auto m120 = OaGptOssConfig::Preset120B();
	EXPECT_TRUE(m120.Validate().IsOk());
	EXPECT_TRUE(m120.IsPublished120B());
	EXPECT_EQ(m120.ExpectedLogicalWeightCount(), 687);
}

TEST(GptOssConfig, ParsesExactPublishedContract) {
	const OaString path = "/tmp/oa_gpt_oss_config_test.json";
	std::ofstream out(path.CStr());
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
	auto parsed = OaGptOssConfig::FromJson(path);
	ASSERT_TRUE(parsed.IsOk()) << parsed.GetStatus().ToString().CStr();
	EXPECT_EQ(parsed->NumLayers, 2);
	EXPECT_EQ(parsed->NumExperts, 32);
	EXPECT_EQ(parsed->QueryWidth(), 4096);
	EXPECT_EQ(parsed->KvWidth(), 512);
	std::remove(path.CStr());
}

TEST(GptOssConfig, RejectsWrongLayerSchedule) {
	const OaString path = "/tmp/oa_gpt_oss_bad_config_test.json";
	std::ofstream out(path.CStr());
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
	auto parsed = OaGptOssConfig::FromJson(path);
	EXPECT_TRUE(parsed.IsError());
	std::remove(path.CStr());
}
