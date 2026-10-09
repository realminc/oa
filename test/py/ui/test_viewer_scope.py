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


if __name__ == "__main__":
	unittest.main()
