//! Runtime API, execution-plan, and Vulkan integration contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "runtime/test_api.rs"]
mod api;
#[path = "runtime/test_execution_plan.rs"]
mod execution_plan;
#[path = "runtime/test_semantic_graph.rs"]
mod semantic_graph;
#[path = "runtime/test_vk.rs"]
mod vk;
