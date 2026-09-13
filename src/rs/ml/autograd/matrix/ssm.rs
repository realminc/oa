use crate::ml::matrix::{Mamba3PreprocessConfig, Mamba3PreprocessResult, SsmConfig};
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_mamba3_preprocess(
	projected: &Matrix,
	dt_bias: &Matrix,
	outputs: &Mamba3PreprocessResult,
	config: Mamba3PreprocessConfig,
) -> Result<()> {
	let [x, z, bh, ch, dt, adt, trap, angle] = outputs.matrices();
	record_node(Node::Mamba3Preprocess(Box::new(
		super::super::node::Mamba3PreprocessNode {
			projected: projected.clone(),
			dt_bias: dt_bias.clone(),
			outputs: [
				x.clone(),
				z.clone(),
				bh.clone(),
				ch.clone(),
				dt.clone(),
				adt.clone(),
				trap.clone(),
				angle.clone(),
			],
			config,
		},
	)))
}

pub(in crate::ml) fn record_mamba3_siso(
	inputs: [&Matrix; 11],
	output: &Matrix,
	config: SsmConfig,
) -> Result<()> {
	let [c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d] = inputs;
	record_node(Node::Mamba3Siso(Box::new(
		super::super::node::Mamba3SisoNode {
			c: c.clone(),
			b: b.clone(),
			x: x.clone(),
			z: z.clone(),
			adt: adt.clone(),
			dt: dt.clone(),
			trap: trap.clone(),
			angle: angle.clone(),
			c_bias: c_bias.clone(),
			b_bias: b_bias.clone(),
			d: d.clone(),
			config,
			output_id: output.value_id(),
		},
	)))
}

pub(in crate::ml) fn record_mamba3_mimo(
	inputs: [&Matrix; 15],
	output: &Matrix,
	config: SsmConfig,
) -> Result<()> {
	record_node(Node::Mamba3Mimo(Box::new(
		super::super::node::Mamba3MimoNode {
			inputs: inputs.map(Clone::clone),
			config,
			output_id: output.value_id(),
		},
	)))
}
