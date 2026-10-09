"""NN schema, ownership and complete-body generation proofs (no GPU required)."""

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("oa_nn_generate", ROOT / "tool/gen/nn/generate.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class NnGenerationTests(unittest.TestCase):
	def schema(self, family):
		return json.loads((ROOT / f"tool/gen/nn/schema/ml/ml_nn_{family}.json").read_text())

	def test_live_inventory_and_public_owners(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		self.assertEqual(len(outputs), 26)
		self.assertEqual(GEN.check(outputs, ROOT), [])
		text = "\n".join(outputs.values())
		for name in ("Relu", "Gelu", "Silu", "Softmax", "LogSoftmax", "Identity", "Flatten", "Dropout", "Linear", "Embedding", "LayerNorm", "RmsNorm", "Conv1d", "ConvTranspose1d", "ConvTranspose2d", "Conv2d", "Swiglu", "AvgPool2d", "MaxPool2d", "AdaptiveAvgPool2d", "Upsample", "Rope", "BatchNorm2d", "Sequential", "MultiHeadAttention", "Ffn", "Transformer", "TransformerBlock", "Moe", "Rnn", "RnnCell", "Gru", "GruCell", "ByteEmbedding", "ByteHead", "VectorQuantizer", "ResidualVectorQuantizer", "Mamba3", "FlowTimeEmbedding", "FlowTransformer", "FlowDenoiser", "Empyrealm"):
			self.assertEqual(text.count(f"pub struct {name} {{"), 1)
			self.assertEqual(text.count(f"impl Module for {name} {{"), 1)
		self.assertNotIn("macro_rules!", text)
		for family in GEN.FAMILIES:
			self.assertFalse((ROOT / f"src/rs/ml/nn/{family}.rs").exists())
			self.assertIn(f'nn/{family}.gen.rs', (ROOT / 'src/rs/ml/nn.rs').read_text())




	def test_empyrealm_provider_and_checkpoint_schema_validation(self):
		original = self.schema("empyrealm")
		for key in GEN.PROVIDER_ROLES["empyrealm"]:
			schema = copy.deepcopy(original)
			schema["layers"][0]["values"][key] = "matrix::sum"
			with self.subTest(provider=key), self.assertRaises(GEN.SchemaError):
				GEN.validate(schema)
		for name in ("Matrix", "Mamba3", "Embedding", "ModuleRegistry"):
			schema = copy.deepcopy(original)
			schema["layers"][0]["name"] = name
			with self.subTest(name=name), self.assertRaises(GEN.SchemaError):
				GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"][0]["values"]["mixer_name"] = "embed"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"].append(copy.deepcopy(schema["layers"][0]))
		schema["layers"][1]["name"] = "OtherEmpyrealm"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)

	def test_empyrealm_schema_owns_type_and_registered_children(self):
		layer = self.schema("empyrealm")["layers"][0]
		template = (ROOT / "tool/gen/nn/template/empyrealm.rs.in").read_text()
		text = GEN.render(template, {"name": "CustomEmpyrealm", "doc": layer["doc"], **layer["values"], "embedding_name": "tokens", "mixer_name": "scan"})
		self.assertIn("impl Module for CustomEmpyrealm", text)
		self.assertIn('register_module("tokens", embedding.clone())', text)
		self.assertIn('register_module("scan", mixer.clone())', text)
		self.assertNotIn("EmpyrealmCore", text)
		self.assertIn("seed.wrapping_add(1)", text)
		self.assertIn("self.mixer.step(embedded, state)?", text)
		self.assertIn("crate::matrix::add(&mixed, embedded)", text)
		self.assertNotIn(".wait()", text)
		self.assertNotIn(".checkpoint(", text)

	def test_empyrealm_rust_and_python_public_name_match(self):
		self.assertIn("pub use empyrealm::Empyrealm;", (ROOT / "src/rs/ml/nn.rs").read_text())
		binding = (ROOT / "src/py/ml/nn/empyrealm.rs").read_text()
		self.assertIn('pyclass(name = "Empyrealm", unsendable)', binding)
		self.assertIn("oa::ml::nn::Empyrealm", binding)
		self.assertIn("class Empyrealm:", (ROOT / "sdk/py/binding/oa/_native.pyi").read_text())
		for name in ("sdk/py/binding/oa/ml/__init__.py", "sdk/py/binding/oa/ml/nn/__init__.py"):
			text = (ROOT / name).read_text()
			self.assertIn('"Empyrealm"', text)
			self.assertNotIn("EmpyrealmCore", text)


	def test_flow_schema_rejects_providers_duplicate_family_and_type_collisions(self):
		original = self.schema("flow")
		for key in GEN.PROVIDER_ROLES["flow"]:
			with self.subTest(provider=key):
				schema = copy.deepcopy(original)
				schema["layers"][0]["values"][key] = "matrix::sum"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(schema)
		for key in ("name", "config_type", "time_type", "transformer_type", "transformer_config_type"):
			for collision in ("Matrix", "Cell", "Linear", "TransformerBlock", "FlowDenoiserConfig"):
				if key == "config_type" and collision == "FlowDenoiserConfig":
					continue
				with self.subTest(type=key, collision=collision):
					schema = copy.deepcopy(original)
					owner = schema["layers"][0] if key == "name" else schema["layers"][0]["values"]
					owner[key] = collision
					with self.assertRaises(GEN.SchemaError):
						GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"][0]["values"]["backbone_name"] = "output_projection"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"].append(copy.deepcopy(schema["layers"][0]))
		schema["layers"][1]["name"] = "SecondDenoiser"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)

	def test_flow_schema_owns_all_types_and_checkpoint_references(self):
		layer = self.schema("flow")["layers"][0]
		template = (ROOT / "tool/gen/nn/template/flow.rs.in").read_text()
		values = {"name": "CustomDenoiser", "doc": layer["doc"], **layer["values"]}
		values.update(config_type="CustomDenoiserConfig", time_type="CustomTime", transformer_type="CustomBackbone", transformer_config_type="CustomBackboneConfig")
		text = GEN.render(template, values)
		for name in ("CustomDenoiser", "CustomDenoiserConfig", "CustomTime", "CustomBackbone", "CustomBackboneConfig"):
			self.assertIn(f"pub struct {name} {{", text)
		for name in ("CustomDenoiser", "CustomTime", "CustomBackbone"):
			self.assertIn(f"impl Module for {name}", text)
		for key, original in layer["values"].items():
			if GEN.CONTRACTS["flow"][key] != "registration_name":
				continue
			with self.subTest(registration=key):
				text = GEN.render(template, dict(values, **{key: "custom_"}))
				if key == "block_prefix":
					self.assertIn('format!("custom_{index}")', text)
					self.assertNotIn('"block_{index}"', text)
				else:
					self.assertIn('"custom_"', text)
					self.assertNotIn(f'"{original}"', text)
		self.assertIn("Rc<CustomTime>", GEN.render(template, values))
		self.assertIn("backbone: CustomBackboneConfig", GEN.render(template, values))

	def test_flow_complete_family_keeps_mask_guidance_and_registered_ownership(self):
		layer = self.schema("flow")["layers"][0]
		text = GEN.render((ROOT / "tool/gen/nn/template/flow.rs.in").read_text(), {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
		for evidence in ('register_buffer("frequencies", frequencies.clone(), false)', "AttentionMode::Bidirectional", 'format!("block_{index}")', "self.is_training() && self.config.condition_dropout_probability > 0.0", "1.0 - self.config.condition_dropout_probability", "autograd::record_parameter_leaf(&self.position)", "matrix::sub_scalar(&key_mask, 1.0)", "matrix::repeat_interleave(&key_mask, repeats, 1)", "self.config.set(config)"):
			self.assertIn(evidence, text)
		guided = text[text.index("pub fn forward_guided("):text.index("fn validate_sample(")]
		self.assertLess(guided.index("let _eval = self.scoped_eval()"), guided.index("self.forward_conditioned("))
		self.assertIn("matrix::sub(&conditional, &unconditional)", guided)
		for bad in (".wait()", ".checkpoint(", ".read_f32(", "ComputeDispatch"):
			self.assertNotIn(bad, text)
		self.assertFalse((ROOT / "src/rs/ml/nn/flow").exists())
		self.assertFalse((ROOT / "src/rs/ml/nn/flow.rs").exists())

	def test_mamba3_schema_rejects_wrong_providers_and_type_name_collisions(self):
		original = self.schema("mamba3")
		for key in GEN.PROVIDER_ROLES["mamba3"]:
			with self.subTest(role=key):
				schema = copy.deepcopy(original)
				schema["layers"][0]["values"][key] = "matrix::sum"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(schema)
		for key in ("name", "config_type", "state_type"):
			for collision in ("Matrix", "Parameter", "ModuleRegistry", "Mamba3"):
				if key == "name" and collision == "Mamba3":
					continue
				with self.subTest(type=key, collision=collision):
					schema = copy.deepcopy(original)
					owner = schema["layers"][0] if key == "name" else schema["layers"][0]["values"]
					owner[key] = collision
					with self.assertRaises(GEN.SchemaError):
						GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"][0]["values"]["state_type"] = "Mamba3Config"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)
		for value in ("in_proj", "a.b", "", 'x"', "a/b"):
			schema = copy.deepcopy(original)
			schema["layers"][0]["values"]["skip_name"] = value
			with self.assertRaises(GEN.SchemaError):
				GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"].append(copy.deepcopy(schema["layers"][0]))
		schema["layers"][1]["name"] = "SecondMamba"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)

	def test_mamba3_schema_owns_types_and_both_parameter_name_references(self):
		layer = self.schema("mamba3")["layers"][0]
		template = (ROOT / "tool/gen/nn/template/mamba3.rs.in").read_text()
		values = {"name": "CustomMamba", "doc": layer["doc"], **layer["values"]}
		values.update(config_type="CustomConfig", state_type="CustomState")
		text = GEN.render(template, values)
		for name in ("CustomMamba", "CustomConfig", "CustomState"):
			self.assertIn(f"pub struct {name} {{", text)
		self.assertIn("impl Module for CustomMamba", text)
		self.assertIn("state: &mut CustomState", text)
		self.assertNotIn("Mamba3State", text)
		self.assertNotIn("Mamba3Config", text)
		for key, original in layer["values"].items():
			if GEN.CONTRACTS["mamba3"][key] != "parameter_name":
				continue
			with self.subTest(parameter=key):
				mutated = dict(values, **{key: "CustomParameter"})
				text = GEN.render(template, mutated)
				self.assertEqual(text.count('"CustomParameter"'), 2)
				self.assertNotIn(f'"{original}"', text)

	def test_mamba3_complete_generated_body_preserves_explicit_inference_state(self):
		layer = self.schema("mamba3")["layers"][0]
		text = GEN.render((ROOT / "tool/gen/nn/template/mamba3.rs.in").read_text(), {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
		step = text[text.index("pub fn step("):text.index("fn parameter_linear(")]
		self.assertLess(step.index("autograd::recording_active()"), step.index("matrix::linear("))
		self.assertLess(step.index("state.owner_id != owner.value_id()"), step.index("matrix::linear("))
		for buffer in ("ssm", "angle", "key", "value"):
			self.assertEqual(step.count(f"&mut state.{buffer},"), 2)
		for provider in ("matrix::mamba3_siso", "matrix::mamba3_mimo", "matrix::mamba3_siso_step", "matrix::mamba3_mimo_step"):
			self.assertIn(provider + "(", text)
		for evidence in ("weight.snapshot()", "autograd::record_linear(", "autograd::record_parameter_leaf(", "Matrix::allocate(owner.engine_handle()", "dt + (-(-dt).exp_m1()).ln()", "config.state_size > 128", "!(1..=8).contains(&config.mimo_rank)"):
			self.assertIn(evidence, text)
		self.assertNotIn(".wait()", text)
		self.assertNotIn(".checkpoint(", text)
		self.assertNotIn(".read_f32(", text)

	def test_byte_schema_rejects_wrong_providers_and_import_collisions(self):
		for index in range(2):
			original = self.schema("byte")
			for key in original["layers"][index]["values"]:
				with self.subTest(layer=index, role=key):
					schema = copy.deepcopy(original)
					schema["layers"][index]["values"][key] = "matrix::sum"
					with self.assertRaises(GEN.SchemaError):
						GEN.validate(schema)
			for name in ("Embedding", "Linear"):
				with self.subTest(layer=index, name=name):
					schema = copy.deepcopy(original)
					schema["layers"][index]["name"] = name
					with self.assertRaises(GEN.SchemaError):
						GEN.validate(schema)

	def test_byte_composition_keeps_child_registry_and_parameter_ownership(self):
		for layer in self.schema("byte")["layers"]:
			with self.subTest(layer=layer["name"]):
				values = {"name": "CustomByteLayer", "doc": layer["doc"], **layer["values"]}
				template = (ROOT / f"tool/gen/nn/template/{layer['template']}.rs.in").read_text()
				text = GEN.render(template, values)
				self.assertIn("pub struct CustomByteLayer {", text)
				self.assertIn("impl Module for CustomByteLayer {", text)
				self.assertIn("CustomByteLayer::forward(self, input)", text)
				self.assertNotIn(layer["name"], text)
				self.assertIn("self.inner.registry()", text)
				self.assertIn("self.inner.weight()", text)
				self.assertIn("self.inner.forward(", text)
				self.assertNotIn("register_module(", text)
				self.assertNotIn("record_", text)
				self.assertIn("byte::VOCAB_SIZE", text)
				if layer["template"] == "byte_embedding":
					self.assertIn("weight.shape().first() != Some(&byte::VOCAB_SIZE)", text)
					self.assertIn("Embedding::from_matrix(weight)?", text)
				else:
					self.assertIn('expect("byte head always constructs a trainable bias")', text)


	def test_vq_schema_rejects_provider_state_and_support_collisions(self):
		original = self.schema("vq")
		for key in GEN.PROVIDER_ROLES["vq"]:
			with self.subTest(role=key):
				schema = copy.deepcopy(original)
				schema["layers"][0]["values"][key] = "matrix::linear"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(schema)
		for key in ("config_type", "result_type", "residual_type", "residual_result_type"):
			with self.subTest(type=key):
				schema = copy.deepcopy(original)
				schema["layers"][0]["values"][key] = "VectorQuantizer"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(schema)
		for key in ("name", "config_type", "result_type", "residual_type", "residual_result_type"):
			for imported in ("Matrix", "NamedBuffer", "NamedStateU32", "ModuleRegistry"):
				with self.subTest(type=key, imported=imported):
					schema = copy.deepcopy(original)
					if key == "name":
						schema["layers"][0][key] = imported
					else:
						schema["layers"][0]["values"][key] = imported
					with self.assertRaises(GEN.SchemaError):
						GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"][0]["values"]["result_type"] = "ResidualVqResult"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"][0]["values"]["ema_step_name"] = "codebook"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)
		schema = copy.deepcopy(original)
		schema["layers"].append(copy.deepcopy(schema["layers"][0]))
		schema["layers"][1]["name"] = "AnotherQuantizer"
		with self.assertRaises(GEN.SchemaError):
			GEN.validate(schema)

	def test_vq_schema_owns_all_public_types_and_checkpoint_references(self):
		layer = self.schema("vq")["layers"][0]
		template = (ROOT / "tool/gen/nn/template/vq.rs.in").read_text()
		values = {"name": layer["name"], "doc": layer["doc"], **layer["values"]}
		for key in ("name", "config_type", "result_type", "residual_type", "residual_result_type"):
			with self.subTest(type=key):
				changed = dict(values, **{key: "CustomVqType"})
				text = GEN.render(template, changed)
				self.assertIn("pub struct CustomVqType {", text)
				self.assertNotRegex(text, rf"\b{values[key]}\b")
		for key in ("codebook_name", "embed_sum_name", "cluster_size_name", "ema_step_name"):
			with self.subTest(state=key):
				text = GEN.render(template, dict(values, **{key: "custom_state"}))
				self.assertNotIn(f'"{values[key]}"', text)
				self.assertIn('"custom_state"', text)
				if key != "ema_step_name":
					self.assertIn('buffer_handle("custom_state")', text)
					self.assertIn('register_buffer("custom_state",', text)
				else:
					self.assertIn('register_state_u32("custom_state", 0)', text)
		text = GEN.render(template, dict(values, level_prefix="custom_level"))
		self.assertIn('format!("custom_level{level}")', text)
		self.assertNotIn('format!("level{level}")', text)

	def test_vq_generated_body_keeps_ema_ownership_and_explicit_seed_completion(self):
		layer = self.schema("vq")["layers"][0]
		template = (ROOT / "tool/gen/nn/template/vq.rs.in").read_text()
		text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
		self.assertEqual(text.count(".checkpoint(false)?.wait()"), 1)
		self.assertNotIn("register_parameter(", text)
		self.assertIn("self.ema_step.replace_value(step.wrapping_add(1))", text)
		self.assertLess(text.index("self.codebook.replace_data(state.codebook)?"), text.index("self.ema_step.replace_value(step.wrapping_add(1))"))
		self.assertIn("matrix::detach(&core_matrix::sub(&assignment.quantized, latent)?)?", text)
		self.assertIn("residuals.push(residual.clone())", text)
		self.assertIn("self.config.num_codes > 512", text)
		self.assertIn("!config.ema_epsilon.is_finite()", text)
		self.assertIn("!(0.0..=1.0).contains(&config.ema_decay)", text)
		self.assertIn("Rc::new(VectorQuantizer::with_seed(", text)
		self.assertIn("codebook.engine_handle().same_as(embed_sum.engine_handle())", text)

	def test_malformed_metadata_is_rejected(self):
		original = self.schema("activation")
		mutations = [
			lambda s: s.update(family=[]),
			lambda s: s.update(version=True),
			lambda s: s.update(helper="../../foreign"),
			lambda s: s.update(imports=["use foo; unsafe {}"]),
			lambda s: s["layers"].append(copy.deepcopy(s["layers"][0])),
			lambda s: s["layers"][0].update(name="../../Escape"),
			lambda s: s["layers"][0].update(template="dropout"),
			lambda s: s["layers"][0].update(doc="injected\ncode"),
			lambda s: s["layers"][0].update(donor="../foreign"),
			lambda s: s["layers"][0]["values"].update(operation="matrix::nonexistent"),
			lambda s: s["layers"][0]["values"].update(unexpected=1),
		]
		for mutate in mutations:
			with self.subTest(mutation=mutate):
				schema = copy.deepcopy(original)
				mutate(schema)
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(schema)

	def test_defaults_are_typed_and_template_owned(self):
		for family, field, invalid in [("softmax", "default_dim", 2**31), ("utility", "start_dim", True), ("dropout", "default_seed", -1)]:
			schema = self.schema(family)
			layer = next(layer for layer in schema["layers"] if field in layer["values"])
			layer["values"][field] = invalid
			with self.assertRaises(GEN.SchemaError):
				GEN.validate(schema)
		schema = self.schema("softmax")
		layer = schema["layers"][0]
		layer["values"]["default_dim"] = 2
		GEN.validate(schema)
		template = (ROOT / "tool/gen/nn/template/softmax.rs.in").read_text()
		text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **{k: str(v) for k, v in layer["values"].items()}})
		self.assertIn("Self::new(2)", text)
		with self.assertRaises(GEN.SchemaError):
			GEN.render("{{unknown}}", {"name": "Layer"})

	def test_identity_flatten_and_dropout_keep_behavior(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		utility = outputs[ROOT / "src/rs/ml/nn/utility.gen.rs"]
		dropout = outputs[ROOT / "src/rs/ml/nn/dropout.gen.rs"]
		self.assertIn("Ok(input.clone())", utility)
		self.assertIn("input.reshape(shape)", utility)
		self.assertIn(".checked_mul(*dimension)", utility)
		self.assertIn("Self::new(1, -1)", utility)
		self.assertIn("!self.is_training() || self.probability == 0.0", dropout)
		self.assertIn("matrix::dropout(input, self.probability, self.seed)", dropout)
		for text in outputs.values():
			self.assertNotIn("BufferBinding", text)
			self.assertNotIn(".submit(", text)
			self.assertNotIn(".read(", text)

	def test_trainable_registration_versions_and_adjoint_owners(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		for family, parameters in [("linear", ["weight", "bias"]), ("embedding", ["weight"]), ("layer_norm", ["weight", "bias"]), ("rms_norm", ["weight"])]:
			with self.subTest(family=family):
				text = outputs[ROOT / f"src/rs/ml/nn/{family}.gen.rs"]
				for parameter in parameters:
					self.assertIn(f'Parameter::new("{parameter}",', text)
					self.assertIn(f'registry.register_parameter("{parameter}",', text)
				self.assertIn(".snapshot()", text)
				self.assertIn(GEN.PROVIDER_ROLES[family]["autograd_operation"] + "(", text)
				self.assertNotIn("requires_grad(true)", text)
		linear = outputs[ROOT / "src/rs/ml/nn/linear.gen.rs"]
		self.assertIn("enum LinearBias", linear)
		self.assertIn("Self::Zero(value) => (value.clone(), None, false)", linear)
		self.assertIn("if let LinearBias::Parameter(parameter) = &bias", linear)
		self.assertIn("Matrix::allocate(", linear)
		self.assertNotIn("Matrix::allocate_output(", linear)
		self.assertIn("random::symmetric_uniform(weight_count, limit, seed)", linear)
		layer_norm = outputs[ROOT / "src/rs/ml/nn/layer_norm.gen.rs"]
		self.assertIn("result.normalized", layer_norm)
		self.assertIn("result.inverse_stddev", layer_norm)
		self.assertIn("autograd::record_channel_norm(", layer_norm)

	def test_parameter_names_and_provider_roles_are_schema_owned(self):
		for family in ("linear", "embedding", "layer_norm", "rms_norm"):
			schema = self.schema(family)
			layer = schema["layers"][0]
			layer["values"]["weight_name"] = "kernel_weight"
			GEN.validate(schema)
			template = (ROOT / f"tool/gen/nn/template/{family}.rs.in").read_text()
			text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
			self.assertIn('Parameter::new("kernel_weight",', text)
			self.assertIn('registry.register_parameter("kernel_weight",', text)
			for key, invalid in [("weight_name", "bad.path"), ("operation", "matrix::sum"), ("autograd_operation", "autograd::record_embedding_bad")]:
				broken = copy.deepcopy(schema)
				broken["layers"][0]["values"][key] = invalid
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(broken)
			if "bias_name" in layer["values"]:
				layer["values"]["bias_name"] = "kernel_weight"
				with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
					GEN.validate(schema)

	def test_convolution_preserves_geometry_layouts_and_parameterized_adjoints(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		text = outputs[ROOT / "src/rs/ml/nn/conv.gen.rs"]
		for provider in ("conv_1d", "conv_transpose_1d", "conv_transpose_2d", "conv_2d"):
			self.assertEqual(text.count(f"matrix::{provider}_parameterized("), 1)
		self.assertEqual(text.count("(self.weight.clone(), weight, weight_version)"), 4)
		self.assertEqual(text.count("(self.bias.clone(), bias, bias_version)"), 3)
		self.assertIn("!input_channels.is_multiple_of(groups)", text)
		self.assertIn("!output_channels.is_multiple_of(groups)", text)
		self.assertIn(".checked_mul(groups)", text)
		self.assertIn("[output_channels, input_channels, kernel_size]", text)
		self.assertIn("[input_channels, output_channels, kernel_size]", text)
		self.assertIn("[input_channels, output_channels, kernel_size, kernel_size]", text)
		start = text.index("pub struct ConvTranspose1d")
		end = text.index("pub struct ConvTranspose2d")
		transpose_1d = text[start:end]
		self.assertNotIn('Parameter::new("bias"', transpose_1d)
		self.assertIn("pub fn parameters(&self) -> [Parameter; 1]", transpose_1d)
		self.assertIn("self.padding,\n\t\t\t1,", transpose_1d)

	def test_swiglu_preserves_seed_offsets_projection_versions_and_zero_bias(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		text = outputs[ROOT / "src/rs/ml/nn/swiglu.gen.rs"]
		self.assertIn("seed.wrapping_add(1)", text)
		self.assertIn("seed.wrapping_add(2)", text)
		self.assertIn('["gate_weight", "up_weight", "down_weight"]', text)
		self.assertIn('["gate_bias", "up_bias", "down_bias"]', text)
		self.assertIn("if let ProjectionBias::Parameter(parameter) = projection_bias", text)
		self.assertIn("Self::Zero(value) => (value.clone(), None, false)", text)
		self.assertIn("weight.snapshot()", text)
		self.assertIn("bias.snapshot()", text)
		self.assertIn("if weight_requires_grad || bias_requires_grad", text)
		self.assertEqual(text.count("autograd::record_linear("), 1)
		self.assertIn("matrix::swiglu(&gate, &up)", text)
		self.assertIn("output.reshape(output_shape)", text)

	def test_convolution_and_swiglu_schema_registration_roles(self):
		for family in ("conv", "swiglu"):
			schema = self.schema(family)
			for index, layer in enumerate(schema["layers"]):
				for key in layer["values"]:
					if not key.endswith("_name"):
						broken = copy.deepcopy(schema)
						broken["layers"][index]["values"][key] = "matrix::sum"
						with self.assertRaises(GEN.SchemaError):
							GEN.validate(broken)
				changed = copy.deepcopy(schema)
				item = changed["layers"][index]
				key = next(key for key in item["values"] if key.endswith("_name"))
				item["values"][key] = "projection_weight"
				GEN.validate(changed)
				template = (ROOT / f"tool/gen/nn/template/{item['template']}.rs.in").read_text()
				text = GEN.render(template, {"name": item["name"], "doc": item["doc"], **item["values"]})
				self.assertEqual(text.count('"projection_weight"'), 2)
				self.assertNotIn('"' + layer['values'][key] + '"', text)
			broken = copy.deepcopy(schema)
			layer = broken["layers"][-1]
			names = [key for key in layer["values"] if key.endswith("_name")]
			layer["values"][names[1]] = layer["values"][names[0]]
			with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
				GEN.validate(broken)
		schema = self.schema("swiglu")
		extra = copy.deepcopy(schema["layers"][0])
		extra["name"] = "OtherSwiglu"
		schema["layers"].append(extra)
		with self.assertRaisesRegex(GEN.SchemaError, "shared projection helper"):
			GEN.validate(schema)

	def test_spatial_provider_roles_and_defaults_are_schema_owned(self):
		for family in ("pool", "upsample", "rope"):
			schema = self.schema(family)
			GEN.validate(schema)
			for index, layer in enumerate(schema["layers"]):
				broken = copy.deepcopy(schema)
				broken["layers"][index]["values"]["operation"] = "matrix::sum"
				with self.assertRaisesRegex(GEN.SchemaError, "unsupported operation provider"):
					GEN.validate(broken)
		schema = self.schema("upsample")
		layer = schema["layers"][0]
		layer["values"]["default_mode"] = "UpsampleMode::Nearest"
		GEN.validate(schema)
		template = (ROOT / "tool/gen/nn/template/upsample.rs.in").read_text()
		text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
		self.assertIn("Self::with_mode(scale_factor, UpsampleMode::Nearest)", text)
		for invalid in ("UpsampleMode::Cubic", "false", True):
			layer["values"]["default_mode"] = invalid
			with self.assertRaisesRegex(GEN.SchemaError, "unsupported interpolation default"):
				GEN.validate(schema)

	def test_spatial_layers_keep_operation_owned_adjoints_and_checked_geometry(self):
		outputs = GEN.expected_outputs(ROOT, ROOT)
		pool = outputs[ROOT / "src/rs/ml/nn/pool.gen.rs"]
		self.assertEqual(pool.count("Self::with_options(kernel_size, kernel_size, 0)"), 2)
		self.assertEqual(pool.count("kernel_size == 0 || stride == 0"), 2)
		self.assertIn("matrix::max_pool_2d(input, self.kernel_size, self.stride, self.padding)?.output", pool)
		self.assertIn("Self::with_output_size(output_size, output_size)", pool)
		self.assertIn("output_height == 0 || output_width == 0", pool)
		self.assertIn("matrix::adaptive_avg_pool_2d(input, self.output_height, self.output_width)", pool)
		upsample = outputs[ROOT / "src/rs/ml/nn/upsample.gen.rs"]
		self.assertIn("Self::with_mode(scale_factor, UpsampleMode::Bilinear)", upsample)
		self.assertIn("scale_factor == 0", upsample)
		rope = outputs[ROOT / "src/rs/ml/nn/rope.gen.rs"]
		for check in ("num_heads == 0", "head_dim == 0", "!head_dim.is_multiple_of(2)", "!theta_base.is_finite()", "theta_base <= 0.0", "num_heads.checked_mul(head_dim).is_none()"):
			self.assertIn(check, rope)
		self.assertIn("matrix::rope(input, self.num_heads, self.head_dim, self.theta_base, 0)", rope)
		for text in (pool, upsample, rope):
			self.assertNotIn("autograd::record_", text)
			self.assertNotIn("ComputeDispatch", text)
			self.assertNotIn("Parameter::new", text)

	def test_upsample_mode_alias_has_one_existing_owner(self):
		schema = self.schema("upsample")
		alias = "pub use super::super::matrix::UpsampleMode;"
		self.assertEqual(schema["imports"].count(alias), 1)
		outputs = GEN.expected_outputs(ROOT, ROOT)
		text = outputs[ROOT / "src/rs/ml/nn/upsample.gen.rs"]
		self.assertIn(alias, text)
		self.assertNotIn("pub enum UpsampleMode", text)
		for imports in (["pub use foreign::UpsampleMode;"], ["pub use super::super::matrix::*;"], [alias + " unsafe {}"]):
			broken = copy.deepcopy(schema)
			broken["imports"] = imports
			with self.assertRaisesRegex(GEN.SchemaError, "invalid imports"):
				GEN.validate(broken)
		broken = self.schema("rope")
		broken["imports"].append(alias)
		with self.assertRaisesRegex(GEN.SchemaError, "invalid imports"):
			GEN.validate(broken)

	def test_batch_norm_schema_owns_state_names_and_exact_provider_roles(self):
		schema = self.schema("batch_norm")
		layer = schema["layers"][0]
		template = (ROOT / "tool/gen/nn/template/batch_norm.rs.in").read_text()
		for key, original in layer["values"].items():
			with self.subTest(field=key):
				changed = copy.deepcopy(schema)
				values = changed["layers"][0]["values"]
				if key.endswith("_name"):
					values[key] = "custom_state"
					GEN.validate(changed)
					text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **values})
					self.assertEqual(text.count('"custom_state"'), 2)
					self.assertNotIn('"' + original + '"', text)
					values[key] = "bad.path"
				else:
					values[key] = "matrix::sum"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(changed)
		# Buffers and parameters share one registry namespace.
		layer["values"]["running_mean_name"] = layer["values"]["weight_name"]
		with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
			GEN.validate(schema)

	def test_batch_norm_retains_persistent_handles_and_mode_sensitive_adjoints(self):
		text = GEN.expected_outputs(ROOT, ROOT)[ROOT / "src/rs/ml/nn/batch_norm.gen.rs"]
		self.assertIn('registry.register_buffer("running_mean", running_mean, true)', text)
		self.assertIn('registry.register_buffer("running_variance", running_variance, true)', text)
		self.assertIn('.buffer_handle("running_mean")', text)
		self.assertIn('.buffer_handle("running_variance")', text)
		self.assertIn("self.weight.snapshot()", text)
		self.assertIn("self.bias.snapshot()", text)
		self.assertIn("let training = self.is_training();", text)
		training = text[text.index("let (output, mean, variance) = if training"):text.index("} else {")]
		inference = text[text.index("} else {"):text.index("autograd::record_batch_norm_2d(")]
		self.assertIn("matrix::batch_norm_2d_forward(", training)
		self.assertIn("matrix::batch_norm_2d_running_update(", training)
		self.assertEqual(training.count(".replace_data("), 2)
		self.assertIn("matrix::batch_norm_2d_with_stats_forward(", inference)
		self.assertNotIn(".replace_data(", inference)
		self.assertEqual(text.count("autograd::record_batch_norm_2d("), 1)
		self.assertIn("weight_requires_grad.then(|| (self.weight.clone(), weight_version))", text)
		self.assertIn("bias_requires_grad.then(|| (self.bias.clone(), bias_version))", text)
		self.assertNotIn("allocate_output", text)

	def test_sequential_schema_keeps_registry_owned_children_and_prefix(self):
		schema = self.schema("sequential")
		layer = schema["layers"][0]
		layer["values"]["child_prefix"] = "stage"
		GEN.validate(schema)
		template = (ROOT / "tool/gen/nn/template/sequential.rs.in").read_text()
		text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **layer["values"]})
		self.assertIn('format!("stage_{}", self.len())', text)
		self.assertIn("`stage_N`", text)
		for invalid in ("bad.path", "", "stage;", 1):
			layer["values"]["child_prefix"] = invalid
			with self.assertRaises(GEN.SchemaError):
				GEN.validate(schema)
		text = GEN.expected_outputs(ROOT, ROOT)[ROOT / "src/rs/ml/nn/sequential.gen.rs"]
		self.assertIn('format!("layer_{}", self.len())', text)
		self.assertIn("module: Rc<dyn Module>", text)
		self.assertIn("self.registry.register_module(name, module)?;", text)
		self.assertIn("let mut output = input.clone();", text)
		self.assertIn("for child in self.registry.child_modules()", text)
		self.assertIn("output = child.forward(&output)?;", text)
		self.assertNotIn("children: Vec", text)
		self.assertNotIn("fn train(", text)

	def test_attention_schema_owns_policy_types_registration_and_providers(self):
		schema = self.schema("attention")
		layer = schema["layers"][0]
		template = (ROOT / "tool/gen/nn/template/attention.rs.in").read_text()
		for key, original in layer["values"].items():
			with self.subTest(field=key):
				changed = copy.deepcopy(schema)
				values = changed["layers"][0]["values"]
				if key.endswith("_type"):
					values[key] = "CustomPolicy"
					GEN.validate(changed)
					text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **values})
					self.assertIn("pub enum CustomPolicy", text)
					self.assertNotIn(original, text)
					values[key] = "bad.path"
				elif key.endswith("_name"):
					values[key] = "custom_projection"
					GEN.validate(changed)
					text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **values})
					self.assertEqual(text.count('"custom_projection"'), 1)
					self.assertNotIn('"' + original + '"', text)
					values[key] = "bad.path"
				else:
					values[key] = "matrix::sum"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(changed)
		broken = copy.deepcopy(schema)
		broken["layers"][0]["values"]["key_name"] = "q_proj"
		with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
			GEN.validate(broken)
		broken = copy.deepcopy(schema)
		extra = copy.deepcopy(layer)
		extra["name"] = "OtherAttention"
		broken["layers"].append(extra)
		with self.assertRaisesRegex(GEN.SchemaError, "one policy and mask helper"):
			GEN.validate(broken)

	def test_attention_supporting_types_share_public_collision_checks(self):
		schema = self.schema("attention")
		for key, value in [("mode_type", "AttentionBackend"), ("backend_type", "MultiHeadAttention")]:
			broken = copy.deepcopy(schema)
			broken["layers"][0]["values"][key] = value
			with self.assertRaisesRegex(GEN.SchemaError, "duplicate public NN type within layer"):
				GEN.validate(broken)
		for key in ("backend_type", "mode_type"):
			broken = copy.deepcopy(schema)
			broken["layers"][0]["values"][key] = "AttentionMask"
			with self.assertRaisesRegex(GEN.SchemaError, "private attention support"):
				GEN.validate(broken)
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			shutil.copytree(ROOT / "tool/gen/nn", root / "tool/gen/nn")
			shutil.copyfile(ROOT / "rustfmt.toml", root / "rustfmt.toml")
			schema["layers"][0]["values"]["mode_type"] = "Relu"
			(root / "tool/gen/nn/schema/ml/ml_nn_attention.json").write_text(json.dumps(schema))
			with self.assertRaisesRegex(GEN.SchemaError, "duplicate public NN type across schemas"):
				GEN.expected_outputs(root, root)
		text = "\n".join(GEN.expected_outputs(ROOT, ROOT).values())
		for name in ("AttentionBackend", "AttentionMode"):
			self.assertEqual(text.count(f"pub enum {name} {{"), 1)

	def test_attention_retains_explicit_routes_replay_dropout_and_mask_cache(self):
		text = GEN.expected_outputs(ROOT, ROOT)[ROOT / "src/rs/ml/nn/attention.gen.rs"]
		for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
			self.assertEqual(text.count(f'registry.register_module("{name}"'), 1)
		self.assertEqual(text.count("Linear::with_seed_and_bias("), 4)
		for offset in (1, 2, 3, 4):
			self.assertIn(f"seed.wrapping_add({offset})", text)
		self.assertIn("AttentionBackend::Auto | AttentionBackend::Standard", text)
		self.assertIn("self.forward_standard(input, None, mode == AttentionMode::Causal)", text)
		for guard in ("mode != AttentionMode::Causal", "input.dtype() != DType::F32", "self.dropout_probability != 0.0", "self.sequence_length.get() > 1024"):
			self.assertIn(guard, text)
		self.assertIn("self.forward_standard(input, Some(additive_mask), false)", text)
		self.assertIn("if self.is_training() && self.dropout_probability > 0.0", text)
		self.assertIn("core_matrix::dropout(&probability, self.dropout_probability, self.dropout_seed)", text)
		self.assertIn("mask.batch == batch && mask.sequence_length == sequence_length && mask.causal == causal", text)
		self.assertIn("self.mask_cache.borrow_mut().clear()", text)
		self.assertIn("flash_attention_causal(&query, &key, &value, scale)", text)
		self.assertNotIn("autograd::record_", text)

	def test_ffn_schema_owns_child_names_and_exact_composition_roles(self):
		schema = self.schema("ffn")
		layer = schema["layers"][0]
		template = (ROOT / "tool/gen/nn/template/ffn.rs.in").read_text()
		for key, original in layer["values"].items():
			with self.subTest(field=key):
				changed = copy.deepcopy(schema)
				values = changed["layers"][0]["values"]
				if key.endswith("_name"):
					values[key] = "custom_child"
					GEN.validate(changed)
					text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **values})
					self.assertEqual(text.count('"custom_child"'), 1)
					self.assertNotIn('"' + original + '"', text)
					values[key] = "bad.path"
				else:
					values[key] = "core_matrix::sum"
				with self.assertRaises(GEN.SchemaError):
					GEN.validate(changed)
		layer["values"]["up_name"] = layer["values"]["gate_name"]
		with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
			GEN.validate(schema)

	def test_ffn_keeps_seeded_pre_norm_swiglu_residual_and_child_adjoints(self):
		text = GEN.expected_outputs(ROOT, ROOT)[ROOT / "src/rs/ml/nn/ffn.gen.rs"]
		positions = [text.index(f'registry.register_module("{name}"') for name in ("norm", "gate", "up", "down")]
		self.assertEqual(positions, sorted(positions))
		self.assertIn("RmsNorm::new(engine, model_width, epsilon)", text)
		self.assertEqual(text.count("Linear::with_seed("), 3)
		self.assertIn("seed.wrapping_add(1)", text)
		self.assertIn("seed.wrapping_add(2)", text)
		forward = text[text.index("pub fn forward(&self, input: &Matrix)"):text.index("/// Return the input")]
		steps = [
			"self.norm.forward(input)?",
			"self.gate.forward(&normalized)?",
			"self.up.forward(&normalized)?",
			"swiglu(&gate, &up)?",
			"self.down.forward(&activated)?",
			"core_matrix::add(input, &projected)",
		]
		positions = [forward.index(step) for step in steps]
		self.assertEqual(positions, sorted(positions))
		self.assertNotIn("autograd::record_", text)
		self.assertNotIn("Parameter::new", text)
		self.assertNotIn("ComputeDispatch", text)

	def test_transformer_contract_rejects_duplicate_owners_and_wrong_providers(self):
		schema = self.schema("transformer")
		GEN.validate(schema)
		for field in GEN.PROVIDER_ROLES["transformer"]:
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"][field] = "matrix::sum"
			with self.subTest(field=field), self.assertRaises(GEN.SchemaError):
				GEN.validate(changed)
		for collision in ("Transformer", "FeedForward", "FeedForwardConfig"):
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"]["block_type"] = collision
			with self.subTest(collision=collision), self.assertRaises(GEN.SchemaError):
				GEN.validate(changed)
		changed = copy.deepcopy(schema)
		changed["layers"].append(copy.deepcopy(changed["layers"][0]))
		with self.assertRaisesRegex(GEN.SchemaError, "one shared recipe"):
			GEN.validate(changed)
		changed = copy.deepcopy(schema)
		changed["layers"][0]["values"]["head_name"] = "final_norm"
		with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
			GEN.validate(changed)

	def test_transformer_schema_owns_both_types_and_checkpoint_names(self):
		schema = self.schema("transformer")
		layer = schema["layers"][0]
		template = (ROOT / "tool/gen/nn/template/transformer.rs.in").read_text()
		for key in [key for key in layer["values"] if key.endswith("_name")] + ["block_prefix"]:
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"][key] = "custom_child"
			GEN.validate(changed)
			text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **changed["layers"][0]["values"]})
			with self.subTest(field=key):
				self.assertIn("custom_child", text)
				self.assertNotIn('register_module("' + layer["values"][key] + '",', text)
		changed = copy.deepcopy(schema)
		changed["layers"][0]["values"]["block_type"] = "CustomBlock"
		GEN.validate(changed)
		text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **changed["layers"][0]["values"]})
		self.assertIn("pub struct CustomBlock", text)
		self.assertNotIn("TransformerBlock", text)

	def test_recurrent_contract_rejects_wrong_providers_and_duplicate_owners(self):
		for family in ("rnn", "gru"):
			schema = self.schema(family)
			GEN.validate(schema)
			for field in GEN.PROVIDER_ROLES[family]:
				changed = copy.deepcopy(schema)
				changed["layers"][0]["values"][field] = "matrix::sum"
				with self.subTest(family=family, field=field), self.assertRaises(GEN.SchemaError):
					GEN.validate(changed)
			for collision in (schema["layers"][0]["name"], "RnnBias", "RnnLayer", "GruBias", "GruLayer"):
				changed = copy.deepcopy(schema)
				changed["layers"][0]["values"]["cell_type"] = collision
				with self.subTest(family=family, collision=collision), self.assertRaises(GEN.SchemaError):
					GEN.validate(changed)
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"]["hidden_weight_name"] = "weight_ih"
			with self.assertRaisesRegex(GEN.SchemaError, "duplicate registration"):
				GEN.validate(changed)
			changed = copy.deepcopy(schema)
			changed["layers"].append(copy.deepcopy(changed["layers"][0]))
			with self.assertRaisesRegex(GEN.SchemaError, "one cell and layer helper set"):
				GEN.validate(changed)

	def test_recurrent_schema_owns_cell_types_and_all_checkpoint_references(self):
		for family in ("rnn", "gru"):
			schema = self.schema(family)
			layer = schema["layers"][0]
			template = (ROOT / f"tool/gen/nn/template/{family}.rs.in").read_text()
			for key in [key for key in layer["values"] if key.endswith("_name")] + ["layer_prefix"]:
				changed = copy.deepcopy(schema)
				changed["layers"][0]["values"][key] = "custom_slot"
				GEN.validate(changed)
				text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **changed["layers"][0]["values"]})
				with self.subTest(family=family, field=key):
					self.assertIn("custom_slot", text)
					if key == "layer_prefix":
						self.assertIn('format!("custom_slot{index}")', text)
						self.assertIn('register_module("custom_slot0",', text)
						self.assertNotIn('"layer0"', text)
						self.assertNotIn('"layer{index}"', text)
					else:
						self.assertNotIn('"' + layer["values"][key] + '"', text)
			changed = copy.deepcopy(schema)
			changed["layers"][0].update(name="CustomRecurrent")
			changed["layers"][0]["values"]["cell_type"] = "CustomCell"
			GEN.validate(changed)
			text = GEN.render(template, {"name": "CustomRecurrent", "doc": layer["doc"], **changed["layers"][0]["values"]})
			self.assertIn("pub struct CustomRecurrent {", text)
			self.assertIn("impl Module for CustomRecurrent {", text)
			self.assertIn("pub struct CustomCell {", text)
			self.assertIn("impl Module for CustomCell {", text)
			self.assertNotIn(layer["values"]["cell_type"], text)

	def test_moe_contract_rejects_wrong_providers_and_duplicate_owners(self):
		schema = self.schema("moe")
		GEN.validate(schema)
		for field in GEN.PROVIDER_ROLES["moe"]:
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"][field] = "matrix::sum"
			with self.subTest(field=field), self.assertRaises(GEN.SchemaError):
				GEN.validate(changed)
		for field, value in (("stats_type", "Moe"), ("routing_bias_name", "router")):
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"][field] = value
			with self.subTest(field=field), self.assertRaises(GEN.SchemaError):
				GEN.validate(changed)
		changed = copy.deepcopy(schema)
		changed["layers"].append(copy.deepcopy(changed["layers"][0]))
		with self.assertRaisesRegex(GEN.SchemaError, "one routing and parameter tree"):
			GEN.validate(changed)

	def test_moe_schema_owns_telemetry_type_and_every_registration_reference(self):
		schema = self.schema("moe")
		layer = schema["layers"][0]
		template = (ROOT / "tool/gen/nn/template/moe.rs.in").read_text()
		for key in [key for key in layer["values"] if key.endswith("_name")] + ["shared_prefix"]:
			changed = copy.deepcopy(schema)
			changed["layers"][0]["values"][key] = "custom_slot"
			GEN.validate(changed)
			text = GEN.render(template, {"name": layer["name"], "doc": layer["doc"], **changed["layers"][0]["values"]})
			with self.subTest(field=key):
				self.assertIn("custom_slot", text)
				self.assertNotIn('"' + layer["values"][key] + '"', text)
				if key == "shared_prefix":
					self.assertIn('format!("custom_slot_{index}")', text)
					self.assertNotIn("shared_expert_{index}", text)
		changed = copy.deepcopy(schema)
		changed["layers"][0]["values"]["stats_type"] = "CustomStats"
		GEN.validate(changed)
		text = GEN.render(template, {"name": "CustomMoe", "doc": layer["doc"], **changed["layers"][0]["values"]})
		self.assertIn("pub struct CustomStats", text)
		self.assertIn("Result<CustomStats>", text)
		self.assertIn("pub struct CustomMoe", text)
		self.assertIn("impl Module for CustomMoe", text)
		self.assertNotIn("MoeRouteStats", text)
		self.assertNotIn("impl Moe", text)

	def test_publication_idempotence_and_orphan_authority(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = GEN.expected_outputs(ROOT, root)
			GEN.publish(outputs, root)
			before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in outputs}
			GEN.publish(outputs, root)
			self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in outputs})
			orphan = root / "src/rs/ml/nn/old.gen.rs"
			orphan.write_text(GEN.BANNER + "\n")
			manual = orphan.with_name("manual.gen.rs")
			manual.write_text("// handwritten\n")
			foreign = orphan.with_name("foreign.gen.rs")
			foreign.write_text("// @generated by tool/gen/fn/generate.py; DO NOT EDIT.\n")
			self.assertTrue(any("orphaned" in e for e in GEN.check(outputs, root)))
			self.assertTrue(orphan.exists())
			GEN.publish(outputs, root)
			self.assertFalse(orphan.exists())
			self.assertTrue(manual.exists())
			self.assertTrue(foreign.exists())

	def test_cross_schema_collisions_and_missing_templates(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			shutil.copytree(ROOT / "tool/gen/nn", root / "tool/gen/nn")
			shutil.copyfile(ROOT / "rustfmt.toml", root / "rustfmt.toml")
			path = root / "tool/gen/nn/schema/ml/ml_nn_softmax.json"
			schema = json.loads(path.read_text())
			schema["layers"][0]["name"] = "Relu"
			path.write_text(json.dumps(schema))
			with self.assertRaisesRegex(GEN.SchemaError, "duplicate public NN type"):
				GEN.expected_outputs(root, root)
			schema["layers"][0]["name"] = "Softmax"
			path.write_text(json.dumps(schema))
			(root / "tool/gen/nn/template/softmax.rs.in").write_text("{{unknown}}")
			with self.assertRaisesRegex(GEN.SchemaError, "metadata does not match tokens"):
				GEN.expected_outputs(root, root)

	def test_schema_key_order_preserves_output(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			shutil.copytree(ROOT / "tool/gen/nn", root / "tool/gen/nn")
			shutil.copyfile(ROOT / "rustfmt.toml", root / "rustfmt.toml")
			before = GEN.expected_outputs(root, root)
			for path in (root / "tool/gen/nn/schema/ml").glob("*.json"):
				schema = json.loads(path.read_text())
				path.write_text(json.dumps(schema, sort_keys=True))
			self.assertEqual(before, GEN.expected_outputs(root, root))

	def test_symlink_and_foreign_destination_fail_before_mutation(self):
		with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
			root = Path(directory)
			(root / "src/rs/ml").mkdir(parents=True)
			(root / "src/rs/ml/nn").symlink_to(outside, target_is_directory=True)
			outputs = GEN.expected_outputs(ROOT, root)
			with self.assertRaises(GEN.SchemaError):
				GEN.publish(outputs, root)
			self.assertEqual(list(Path(outside).iterdir()), [])
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			outputs = GEN.expected_outputs(ROOT, root)
			first = next(iter(outputs))
			first.parent.mkdir(parents=True)
			first.write_text("// user work\n")
			with self.assertRaises(GEN.SchemaError):
				GEN.publish(outputs, root)
			self.assertEqual(first.read_text(), "// user work\n")
			self.assertEqual(len(list(first.parent.iterdir())), 1)


if __name__ == "__main__":
	unittest.main()
