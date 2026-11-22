import attributes;

[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

#include "to_matrix_semiplanar.slang"

[kernel_name("to_matrix_p010")]
[domain("video")]
[variant("semiplanar_420_10bit")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	to_matrix_semiplanar(thread_id, 10u);
}
