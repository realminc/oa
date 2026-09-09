use oa::{
	DType, ErrorKind, OpAttribute, OpAttributeKind, OpAttributeSpec, OpDifferentiation, OpEffect,
	OpLowering, OpShapeRule, OpValueKind, OperationContract, SemanticAccessMode, SemanticGraph,
	SemanticValueDesc,
};

const MATRIX: &[OpValueKind] = &[OpValueKind::Matrix];
const TWO_MATRICES: &[OpValueKind] = &[OpValueKind::Matrix, OpValueKind::Matrix];
const SCALE_ATTRIBUTES: &[OpAttributeSpec] =
	&[OpAttributeSpec::new("scalar", OpAttributeKind::Float)];

const ADD: OperationContract = OperationContract::new(
	"oa::FnMatrix::add",
	0x0a2a_8622_0085_9b54,
	TWO_MATRICES,
	MATRIX,
)
.with_differentiation(OpDifferentiation::Reverse)
.effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS));

const SCALE: OperationContract =
	OperationContract::new("oa::FnMatrix::scale", 0x3fa3_800c_ff89_b8ea, MATRIX, MATRIX)
		.attributes(SCALE_ATTRIBUTES)
		.with_differentiation(OpDifferentiation::Reverse)
		.effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS));

const ADD_IN_PLACE: OperationContract = OperationContract::new(
	"oa::FnMatrix::addInPlace",
	0x5720_66f6_95b2_e963,
	TWO_MATRICES,
	MATRIX,
)
.effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS))
.mutated_inputs(1)
.alias(0, 0);

const BACKWARD: OperationContract =
	OperationContract::new("oa::GradAdd", 0xda77_9010_8b5a_3511, MATRIX, MATRIX)
		.effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS));

fn matrix(name: &str, shape: &[usize], external: bool) -> SemanticValueDesc {
	SemanticValueDesc::new(name, OpValueKind::Matrix, shape, DType::F32)
		.expect("valid test shape")
		.external(external)
}

#[test]
fn ports_oa_semantic_identity_access_and_attribute_contracts() -> oa::Result<()> {
	let mut graph = SemanticGraph::new();
	let left = graph.add_value(matrix("left", &[2, 3], true))?;
	let right = graph.add_value(matrix("right", &[2, 3], true))?;
	let sum = graph.add_value(matrix("sum", &[2, 3], false))?;
	let add = graph.add_operation(ADD, &[Some(left), Some(right)], &[sum], &[], &[])?;
	let scaled = graph.add_value(matrix("scaled", &[2, 3], false))?;
	let scale = graph.add_operation(
		SCALE,
		&[Some(sum)],
		&[scaled],
		&[add],
		&[OpAttribute::Float {
			name: "scalar".into(),
			value: 1.25,
		}],
	)?;

	graph.validate()?;
	assert_eq!(graph.values().len(), 4);
	assert_eq!(graph.operations().len(), 2);
	assert_eq!(
		graph.find_value(sum).and_then(|value| value.producer()),
		Some(add)
	);
	assert_eq!(graph.operations()[0].name(), "oa::FnMatrix::add");
	assert_eq!(graph.operations()[0].contract_hash(), ADD.hash());
	assert_eq!(graph.operations()[0].inputs(), &[Some(left), Some(right)]);
	assert_eq!(graph.operations()[0].outputs(), &[sum]);
	assert_eq!(graph.operations()[0].accesses().len(), 3);
	assert_eq!(
		graph.operations()[0].accesses()[0].mode(),
		SemanticAccessMode::Read
	);
	assert_eq!(
		graph.operations()[0].accesses()[2].mode(),
		SemanticAccessMode::Write
	);
	assert_eq!(graph.operations()[1].control_dependencies(), &[add]);
	assert_eq!(graph.operations()[1].id(), scale);
	assert_eq!(graph.operations()[1].attributes().len(), 1);
	let report = graph.debug_report_json("semantic\n\"probe");
	let report: serde_json::Value =
		serde_json::from_str(&report).expect("semantic report must be valid JSON");
	assert_eq!(report["schema"], "oa.semantic_graph.v2");
	assert_eq!(report["name"], "semantic\n\"probe");
	assert_eq!(report["values"][0]["name"], "left");
	assert_eq!(report["values"][0]["dtype"], "float32");
	assert_eq!(
		report["operations"][0]["contract_hash"],
		"0x0a2a862200859b54"
	);
	assert_eq!(report["operations"][1]["attributes"][0]["value"], 1.25);
	assert_eq!(report["operations"][1]["control_dependencies"][0], 0);
	Ok(())
}

#[test]
fn ports_metadata_views_without_fake_operations() -> oa::Result<()> {
	let mut graph = SemanticGraph::new();
	let source = graph.add_value(matrix("source", &[2, 3], true).byte_offset(16))?;
	let view = graph.add_value(matrix("view", &[3, 2], false).strides([2, 1])?)?;
	graph.add_view(source, view, 8)?;

	graph.validate()?;
	assert_eq!(graph.operations().len(), 0);
	assert_eq!(graph.view_count(), 1);
	assert_eq!(
		graph.values()[view.index() as usize].view_source(),
		Some(source)
	);
	assert_eq!(
		graph.values()[view.index() as usize].byte_offset_value(),
		24
	);
	assert_eq!(graph.values()[view.index() as usize].strides_ref(), [2, 1]);
	Ok(())
}

#[test]
fn ports_mutation_alias_and_autograd_provenance() -> oa::Result<()> {
	let mut graph = SemanticGraph::new();
	let left = graph.add_value(matrix("left", &[4], true))?;
	let right = graph.add_value(matrix("right", &[4], true))?;
	let sum = graph.add_value(matrix("sum", &[4], false))?;
	let forward = graph.add_operation(ADD, &[Some(left), Some(right)], &[sum], &[], &[])?;
	graph.attach_autograd(forward, 0, 17)?;

	let gradient = graph.add_value(matrix("gradient", &[4], true))?;
	let left_gradient = graph.add_value(matrix("left_gradient", &[4], false))?;
	let backward = graph.add_operation(
		BACKWARD,
		&[Some(gradient)],
		&[left_gradient],
		&[forward],
		&[],
	)?;
	graph.complete_autograd(forward, 17, backward, 1)?;

	let updated = graph.add_value(matrix("updated", &[4], false))?;
	let mutation = graph.add_operation(
		ADD_IN_PLACE,
		&[Some(left), Some(right)],
		&[updated],
		&[backward],
		&[],
	)?;

	graph.validate()?;
	assert_eq!(
		graph.operations()[backward.index() as usize].backward_of(),
		Some(forward)
	);
	assert_eq!(
		graph.operations()[backward.index() as usize].backward_sequence(),
		17
	);
	assert_eq!(
		graph.operations()[mutation.index() as usize].mutated_inputs(),
		&[left]
	);
	assert_eq!(
		graph.operations()[mutation.index() as usize]
			.aliases()
			.len(),
		1
	);
	assert_eq!(
		graph.operations()[mutation.index() as usize].aliases()[0].input(),
		left
	);
	assert_eq!(
		graph.operations()[mutation.index() as usize].aliases()[0].output(),
		updated
	);
	assert!(graph.autograd()[0].is_backward_expanded());
	assert_eq!(graph.autograd()[0].backward_first_op(), Some(backward));
	let report: serde_json::Value = serde_json::from_str(&graph.debug_report_json("autograd"))
		.expect("semantic report must be valid JSON");
	assert_eq!(report["operations"][1]["backward_of"], 0);
	assert_eq!(report["operations"][2]["mutated_inputs"][0], 0);
	assert_eq!(report["operations"][2]["aliases"][0]["output"], 5);
	assert_eq!(report["autograd"][0]["backward_expanded"], true);
	assert_eq!(report["autograd"][0]["backward_first_operation"], 1);
	Ok(())
}

#[test]
fn rejects_invalid_contract_edges_transactionally() -> oa::Result<()> {
	let mut graph = SemanticGraph::new();
	let input = graph.add_value(matrix("input", &[4], true))?;
	let output = graph.add_value(matrix("output", &[4], false))?;

	let error = graph
		.add_operation(ADD, &[Some(input)], &[output], &[], &[])
		.unwrap_err();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert!(graph.operations().is_empty());
	assert_eq!(
		graph.find_value(output).and_then(|value| value.producer()),
		None
	);

	let error = graph
		.add_operation(
			SCALE,
			&[Some(input)],
			&[output],
			&[],
			&[OpAttribute::Float {
				name: "wrong".into(),
				value: 1.0,
			}],
		)
		.unwrap_err();
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert!(graph.operations().is_empty());
	Ok(())
}

#[test]
fn preserves_lowering_and_generation_contracts() -> oa::Result<()> {
	const MAT_MUL_NT: OperationContract = OperationContract::new(
		"oa::FnMatrix::matMulNt",
		0x4309_b836_dbcb_8f84,
		TWO_MATRICES,
		MATRIX,
	)
	.with_differentiation(OpDifferentiation::Reverse)
	.with_lowering(OpLowering::Gemm)
	.with_shape_rule(OpShapeRule::MatMulNt)
	.effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS));

	let mut source = SemanticGraph::new();
	let left = source.add_value(matrix("left", &[2, 3], true))?;
	let right = source.add_value(matrix("right", &[4, 3], true))?;
	let output = source.add_value(matrix("output", &[2, 4], false))?;
	source.add_operation(MAT_MUL_NT, &[Some(left), Some(right)], &[output], &[], &[])?;
	assert_eq!(source.operations()[0].lowering(), OpLowering::Gemm);

	let mut copy = SemanticGraph::new();
	copy.copy_from(&source)?;
	copy.validate()?;
	assert_eq!(copy.operations()[0].contract_hash(), MAT_MUL_NT.hash());
	copy.reset();
	assert_eq!(copy.generation(), 2);
	assert!(copy.values().is_empty());
	Ok(())
}
