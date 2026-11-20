import numpy as np
from PIL import Image

from gui_recover.portable_tail import inspect_selection


def test_thin_bleed_from_the_previous_glyph_is_not_a_selected_glyph():
    original = np.zeros((60, 100, 3), dtype=np.uint8)
    original[20:26, 39:42] = 255
    highlighted = original.copy()
    highlighted[12:42, 40:49] = (80, 40, 100)
    interior = (20, 8, 75, 49)
    result = inspect_selection(Image.fromarray(original), Image.fromarray(highlighted), interior)
    assert result["status"] == "blank_tail" and result["left_edge_bleed"]
    original[22:26, 45:47] = 255
    result = inspect_selection(Image.fromarray(original), Image.fromarray(highlighted), interior)
    assert result["status"] == "visible_tail" and not result["left_edge_bleed"]
