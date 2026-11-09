// Tiled Linear data adjoint with a fused producer-side prologue on the
// upstream gradient, adapted from linear_backward.slang. Computes
//   dy[m,n] = prologue(upstream, saved)[m,n]
//   dx[m,k] = sum_n dy[m,n] * weight[n,k]
// and also materializes dy so downstream consumers (parameter backward,
// autograd saves and observers) see exactly the buffers the unfused pair
// would have written. dy is stored only by the first column tile so every
// element keeps one writer.
// prologue_kind: 0-8 are the activation adjoints of linear_act.slang
//   (0=relu 1=gelu 2=silu 3=tanh 4=sigmoid 5=mish 6=elu 7=leaky_relu
//   8=softplus), 9=silu_mul (split gate/up over a packed [gate|up]
//   projection of output_features = 2*hidden elements), and 10-13 are
//   pointwise combines folded from a producing Binary node
//   (10=add 11=sub 12=mul 13=scalar splat mul).
// `upstream` and `saved` are the two read operands of the producer: for
// activation adjoints upstream is the adjoint's gradient input and saved is
// whatever the source kernel read (the activation output for
// relu/tanh/sigmoid/elu/softplus and the preactivation input otherwise; for
// silu_mul the packed gate|up projection); for the combines they are the
// left/right operand pair, with a scalar `saved` splat for kind 13.
// alpha feeds elu/leaky_relu only.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint weight_index;
	uint upstream_gradient_index;
	uint saved_index;
	uint dy_index;
	uint input_gradient_index;
	uint batch;
	uint input_features;
	uint output_features;
	uint prologue_kind;
	float alpha;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = 32u;
static const uint BK = 32u;
static const uint BN = 16u;
static const uint TM = 4u;
static const uint TK = 4u;
static const uint BN_PAD = BN + 1u;
static const uint PROLOGUE_SILU_MUL = 9u;
static const uint PROLOGUE_SCALAR_MUL = 13u;
groupshared float shared_output_gradient[BM * BN_PAD];
groupshared float shared_weight[BK * BN_PAD];

float apply_prologue(float saved, float upstream) {
	switch (push.prologue_kind) {
	case 0u:
		return saved > 0.0f ? upstream : 0.0f;
	case 1u: {
		float x2 = saved * saved;
		float inner = 0.7978845608028654f * (saved + 0.044715f * x2 * saved);
		float t = tanh(inner);
		return upstream * (0.5f * (1.0f + t)
			+ 0.5f * saved * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 0.134145f * x2));
	}
	case 2u:
		return upstream * silu_grad(saved);
	case 3u:
		return upstream * (1.0f - saved * saved);
	case 4u:
		return upstream * saved * (1.0f - saved);
	case 5u: {
		float softplus = saved > 20.0f ? saved : log(1.0f + exp(saved));
		float tanh_softplus = tanh(softplus);
		return upstream * (tanh_softplus
			+ saved * sigmoid(saved) * (1.0f - tanh_softplus * tanh_softplus));
	}
	case 6u:
		return upstream * (saved > 0.0f ? 1.0f : saved + push.alpha);
	case 7u:
		return saved > 0.0f ? upstream : upstream * push.alpha;
	case 8u:
		return upstream * (1.0f - exp(-saved));
	case 10u:
		return upstream + saved;
	case 11u:
		return upstream - saved;
	default:
		return upstream * saved;
	}
}

[kernel_name("linear_backward_prologue")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BK;
	if (row_base >= push.batch || column_base >= push.input_features) {
		return;
	}

	uint thread = group_thread_id.x;
	uint thread_row = thread / 8u;
	uint thread_column = thread % 8u;
	float accumulators[TM * TK];
	[[unroll]] for (uint index = 0u; index < TM * TK; ++index) {
		accumulators[index] = 0.0f;
	}
	// Kind 13 splats a one-element scalar operand instead of reading `saved`
	// per tile element; the scalar buffer owns exactly one f32.
	float scalar = push.prologue_kind == PROLOGUE_SCALAR_MUL
		? load_f32(storage_buffers[push.saved_index], 0)
		: 0.0f;

	for (uint output_base = 0u; output_base < push.output_features; output_base += BN) {
		[[unroll]] for (uint load = 0u; load < 8u; ++load) {
			uint index = thread + load * 64u;
			uint output_lane = index / BN;
			uint output_feature = output_base + index % BN;
			uint row = row_base + output_lane;
			bool valid = row < push.batch && output_feature < push.output_features;
			uint element = row * push.output_features + output_feature;
			float dy = 0.0f;
			if (valid && push.prologue_kind == PROLOGUE_SILU_MUL) {
				// Packed gate|up: column selects the half, saved and upstream
				// index the reduced silu_mul output tile of width
				// output_features/2.
				uint hidden = push.output_features / 2u;
				uint column = output_feature % hidden;
				uint reduced = row * hidden + column;
				float gate = load_f32(
					storage_buffers[push.saved_index],
					row * push.output_features + column
				);
				float upstream = load_f32(
					storage_buffers[push.upstream_gradient_index],
					reduced
				);
				if (output_feature < hidden) {
					float up = load_f32(
						storage_buffers[push.saved_index],
						row * push.output_features + hidden + column
					);
					dy = upstream * up * silu_grad(gate);
				} else {
					dy = upstream * silu(gate);
				}
			} else {
				float upstream = valid
					? load_f32(storage_buffers[push.upstream_gradient_index], element)
					: 0.0f;
				float saved = valid && push.prologue_kind != PROLOGUE_SCALAR_MUL
					? load_f32(storage_buffers[push.saved_index], element)
					: scalar;
				dy = valid ? apply_prologue(saved, upstream) : 0.0f;
			}
			shared_output_gradient[output_lane * BN_PAD + index % BN] = dy;
			if (valid && group_id.y == 0u) {
				store_f32(storage_buffers[push.dy_index], element, dy);
			}
			uint input_feature = column_base + output_lane;
			shared_weight[output_lane * BN_PAD + index % BN] =
				input_feature < push.input_features && output_feature < push.output_features
				? load_f32(storage_buffers[push.weight_index], output_feature * push.input_features + input_feature)
				: 0.0f;
		}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint output_feature = 0u; output_feature < BN; ++output_feature) {
			float output_values[TM];
			float weight_values[TK];
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				output_values[row] = shared_output_gradient[(thread_row * TM + row) * BN_PAD + output_feature];
			}
			[[unroll]] for (uint column = 0u; column < TK; ++column) {
				weight_values[column] = shared_weight[(thread_column * TK + column) * BN_PAD + output_feature];
			}
			[[unroll]] for (uint row = 0u; row < TM; ++row) {
				[[unroll]] for (uint column = 0u; column < TK; ++column) {
					accumulators[row * TK + column] += output_values[row] * weight_values[column];
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint row = 0u; row < TM; ++row) {
		[[unroll]] for (uint column = 0u; column < TK; ++column) {
			uint output_row = row_base + thread_row * TM + row;
			uint input_feature = column_base + thread_column * TK + column;
			if (output_row < push.batch && input_feature < push.input_features) {
				store_f32(
					storage_buffers[push.input_gradient_index],
					output_row * push.input_features + input_feature,
					accumulators[row * TK + column]
				);
			}
		}
	}
}
