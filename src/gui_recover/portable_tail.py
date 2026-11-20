"""A color-independent tail-selection detector for a known editable region."""

import numpy as np


def inspect_selection(restored, selected, field_interior):
    """Classify one selected character from screenshots and a field interior rectangle."""
    if restored.size != selected.size:
        return {"status": "unknown", "reason": "different_screenshot_sizes"}
    left, top, right, bottom = field_interior
    if not (0 <= left < right <= restored.width and 0 <= top < bottom <= restored.height):
        return {"status": "unknown", "reason": "invalid_field_interior"}

    original = np.asarray(restored.convert("RGB"), dtype=np.int16)[top:bottom, left:right]
    highlighted = np.asarray(selected.convert("RGB"), dtype=np.int16)[top:bottom, left:right]
    changed = np.max(np.abs(highlighted - original), axis=2) >= 20
    ys, xs = np.nonzero(changed)
    if not len(xs):
        return {"status": "unknown", "reason": "selection_missing"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    if (height < 8 or width < 2 or width > height or len(xs) / (width * height) < .45
            or x0 == 0 or x1 == right - left):
        return {"status": "unknown", "reason": "selection_outside_supported_geometry"}

    pixels = original[y0:y1, x0:x1].reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    majority = int(counts.argmax())
    if counts[majority] * 2 <= len(pixels):
        return {"status": "unknown", "reason": "no_uniform_background"}
    background = colors[majority]
    ink_mask = np.max(np.abs(pixels - background), axis=1).reshape(height, width) >= 45
    ink = int(ink_mask.sum())
    # Italic or antialiased glyphs can overhang into an adjacent selected
    # space. Treat a narrow strip as bleed only when it connects to ink on
    # the unselected side of that edge.
    outside_ink = np.max(np.abs(original[y0:y1, x0-1] - background), axis=1) >= 45
    gutter = min(2, width // 3)
    edge_bleed = (gutter > 0 and not ink_mask[:, gutter:].any()
                  and np.any(ink_mask[:, 0] & outside_ink))
    status = "blank_tail" if ink <= 1 or edge_bleed else "visible_tail" if ink >= 4 else "unknown"
    return {"status": status, "ink_pixels": ink, "background_rgb": colors[majority].tolist(),
            "left_edge_bleed": bool(edge_bleed), "selection_region": [left+x0, top+y0, left+x1, top+y1]}
