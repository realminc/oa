use crate::ml::matrix::SsmConfig;
use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

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
