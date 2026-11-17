import unittest

import oa
from oa import _native


class ViewerBindingScopeTest(unittest.TestCase):
	def test_python_presentation_surface_is_viewer_and_plot(self) -> None:
		self.assertIs(oa.ui.Viewer, oa.Viewer)
		self.assertIs(oa.plot.Figure, _native.Figure)
		for name in ("Ui", "Renderer", "RenderFrame", "Texture"):
			self.assertFalse(hasattr(_native, name), name)
			self.assertFalse(hasattr(oa, name), name)
		self.assertFalse(hasattr(oa, "render"))

	def test_workspace_docks_figure_views_and_serializes_layout(self) -> None:
		self.assertIs(oa.viewer.Workspace, _native.Workspace)
		self.assertIn("Workspace", dir(oa.viewer.ViewerMode))
		ws = oa.viewer.Workspace()
		fig = oa.plot.Figure()
		fig.ax(0, 0).plot([0.0, 1.0, 0.5])
		view = oa.plot.FigureView(fig)
		ws.add_view("signal", view)
		doc = ws.to_json()
		self.assertIn("signal", doc)
		ws.set_layout(doc)
		self.assertIn("signal", ws.to_json())


if __name__ == "__main__":
	unittest.main()
