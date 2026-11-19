"""Visual appearance guards for initially empty, outlined input fields.

It checks appearance of ink, not text correctness. No DOM, OCR, reward or
injection labels enter this API. Unsupported layouts return unknown.
"""

from collections import deque

import numpy as np


def _component(mask, start):
    height, width = mask.shape
    queue = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= xx < width and 0 <= yy < height and mask[yy, xx] and (xx, yy) not in visited:
                visited.add((xx, yy))
                queue.append((xx, yy))
    return visited


def _blank_region(image, position):
    pixels = np.asarray(image.convert("RGB"))
    white = pixels.min(axis=2) >= 245
    height, width = white.shape
    x = min(width - 1, int(position[0] * width))
    y = min(height - 1, int(position[1] * height))
    candidates = [(xx, yy) for yy in range(max(0, y - 3), min(height, y + 4))
                  for xx in range(max(0, x - 3), min(width, x + 4)) if white[yy, xx]]
    if not candidates:
        return None
    examined = set()
    for start in sorted(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2):
        if start in examined:
            continue
        component = _component(white, start)
        examined.update(component)
        x0 = min(p[0] for p in component)
        x1 = max(p[0] for p in component) + 1
        y0 = min(p[1] for p in component)
        y1 = max(p[1] for p in component) + 1
        w, h = x1 - x0, y1 - y0
        if (x0 == 0 or y0 == 0 or x1 == width or y1 == height or w < 20 or h < 8
                or not 2 <= w / h <= 20 or w * h > width * height * 0.3
                or len(component) / (w * h) < 0.97):
            continue
        return (x0, y0, x1, y1)
    return None


def _ink(image, bounds, ignore_speckles=False):
    x0, y0, x1, y1 = bounds
    # Two interior pixels exclude Chromium's thicker focused outline at the
    # fixed MiniWoB screenshot scale, as well as border antialiasing.
    crop = np.asarray(image.convert("RGB"))[y0 + 2:y1 - 2, x0 + 2:x1 - 2]
    dark = crop.mean(axis=2) < 180
    remaining = set(zip(*np.nonzero(dark)[::-1]))
    ink = 0
    ignored_caret_pixels = 0
    ignored_speckle_pixels = 0
    while remaining:
        component = _component(dark, next(iter(remaining)))
        remaining.difference_update(component)
        xs, ys = zip(*component)
        w = max(xs) - min(xs) + 1
        h = max(ys) - min(ys) + 1
        if ignore_speckles and len(component) == 1:
            ignored_speckle_pixels += 1
        elif w <= 2 and h >= crop.shape[0] * 0.5:
            ignored_caret_pixels += len(component)
        else:
            ink += len(component)
    return ink, ignored_caret_pixels, ignored_speckle_pixels


def inspect_input(before_image, after_image, position, version="v1"):
    if version == "v3":
        return _inspect_uniform_input(before_image, after_image, position)
    if version not in ("v1", "v2"):
        raise ValueError("Unknown input guard version")
    if before_image.size != after_image.size:
        return {"status": "unknown", "reason": "screenshot_size_changed"}
    bounds = _blank_region(before_image, position)
    if bounds is None:
        return {"status": "unknown", "reason": "no_enclosed_white_input_region"}
    before_ink, _, _ = _ink(before_image, bounds, ignore_speckles=version == "v2")
    if before_ink:
        return {"status": "unknown", "reason": "region_was_not_empty", "region": bounds}
    x0, y0, x1, y1 = bounds
    white = np.asarray(after_image.convert("RGB")).min(axis=2) >= 245
    borders = [white[y0 - 1, x0:x1], white[y1, x0:x1],
               white[y0:y1, x0 - 1], white[y0:y1, x1]]
    if any(float((~border).mean()) < 0.65 for border in borders):
        return {"status": "unknown", "reason": "input_outline_moved_or_changed", "region": bounds}
    after_ink, caret, speckles = _ink(after_image, bounds, ignore_speckles=version == "v2")
    interior_area = (x1 - x0 - 4) * (y1 - y0 - 4)
    if after_ink > interior_area * 0.4:
        return {"status": "unknown", "reason": "region_background_changed", "region": bounds}
    status = "text_visible" if after_ink >= 8 else ("empty" if after_ink == 0 else "unknown")
    result = {"status": status, "region": bounds, "ink_before": before_ink,
            "ink_after": after_ink, "ignored_caret_pixels": caret,
            "reason": "appearance_only_not_text_correctness"}
    if version == "v2":
        result["ignored_speckle_pixels"] = speckles
    return result


def _uniform_region(pixels, position):
    """Find a nearly solid enclosed rectangle, using its own RGB as background."""
    height, width, _ = pixels.shape
    x = min(width - 1, int(position[0] * width))
    y = min(height - 1, int(position[1] * height))
    candidates = [(xx, yy) for yy in range(max(0, y - 3), min(height, y + 4))
                  for xx in range(max(0, x - 3), min(width, x + 4))]
    examined = set()
    for start in sorted(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2):
        if start in examined:
            continue
        background = pixels[start[1], start[0]]
        similar = np.abs(pixels - background).max(axis=2) <= 12
        component = _component(similar, start)
        examined.update(component)
        xs, ys = zip(*component)
        bounds = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        x0, y0, x1, y1 = bounds
        w, h = x1 - x0, y1 - y0
        if (x0 == 0 or y0 == 0 or x1 == width or y1 == height or w < 20 or h < 8
                or not 2 <= w / h <= 20 or w * h > width * height * 0.3
                or len(component) / (w * h) < 0.97):
            continue
        background = np.median(pixels[y0:y1, x0:x1], axis=(0, 1))
        if _has_outline(pixels, bounds, background):
            return bounds, background
    return None


def _has_outline(pixels, bounds, background):
    x0, y0, x1, y1 = bounds
    contrast = np.abs(pixels - background).max(axis=2) >= 24
    borders = [contrast[y0 - 1, x0:x1], contrast[y1, x0:x1],
               contrast[y0:y1, x0 - 1], contrast[y0:y1, x1]]
    return all(float(border.mean()) >= 0.65 for border in borders)


def _contrast_ink(pixels, bounds, background):
    x0, y0, x1, y1 = bounds
    crop = pixels[y0 + 2:y1 - 2, x0 + 2:x1 - 2]
    foreground = np.abs(crop - background).max(axis=2) >= 45
    remaining = set(zip(*np.nonzero(foreground)[::-1]))
    ink = caret = speckles = 0
    while remaining:
        component = _component(foreground, next(iter(remaining)))
        remaining.difference_update(component)
        xs, ys = zip(*component)
        w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if len(component) == 1:
            speckles += 1
        elif w <= 2 and h >= crop.shape[0] * 0.5:
            caret += len(component)
        else:
            ink += len(component)
    return ink, caret, speckles, crop


def _inspect_uniform_input(before_image, after_image, position):
    if before_image.size != after_image.size:
        return {"status": "unknown", "reason": "screenshot_size_changed"}
    before = np.asarray(before_image.convert("RGB"), dtype=np.int16)
    after = np.asarray(after_image.convert("RGB"), dtype=np.int16)
    found = _uniform_region(before, position)
    if found is None:
        return {"status": "unknown", "reason": "no_enclosed_uniform_input_region"}
    bounds, background = found
    before_ink, _, _, _ = _contrast_ink(before, bounds, background)
    if before_ink:
        return {"status": "unknown", "reason": "region_was_not_empty", "region": bounds}
    if not _has_outline(after, bounds, background):
        return {"status": "unknown", "reason": "input_outline_moved_or_changed", "region": bounds}
    ink, caret, speckles, crop = _contrast_ink(after, bounds, background)
    # A changed fill must not be mistaken for either text or a blank field.
    if (np.abs(np.median(crop, axis=(0, 1)) - background).max() > 12
            or ink > crop.shape[0] * crop.shape[1] * 0.4):
        return {"status": "unknown", "reason": "region_background_changed", "region": bounds}
    return {"status": "text_visible" if ink >= 8 else ("empty" if ink == 0 else "unknown"),
            "region": bounds, "background_rgb": background.astype(int).tolist(),
            "ink_before": before_ink, "ink_after": ink,
            "ignored_caret_pixels": caret, "ignored_speckle_pixels": speckles,
            "reason": "appearance_only_not_text_correctness"}
