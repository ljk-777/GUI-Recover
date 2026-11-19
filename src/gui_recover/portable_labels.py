"""Explicit rasterization and two-scale agreement for the checkbox reader."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .target_state import CheckboxStateVerifier, _similarity, _trim, _unknown


_SCALES = (1, 4)
_BLUR = ImageFilter.GaussianBlur(.5)


def _smoothed(ink):
    raster = Image.fromarray((ink*255).astype(np.uint8))
    return np.asarray(raster.filter(_BLUR), dtype=float)/255


class PortableCheckboxStateVerifier(CheckboxStateVerifier):
    """Opt-in label matcher with the same state and recovery checks."""

    def __init__(self, font_path, *, minimum_score=.87, minimum_margin=.18, minimum_agreement_margin=.12):
        if not (0 < minimum_score <= 1 and
                0 < minimum_agreement_margin <= minimum_margin <= 1):
            raise ValueError("Invalid portable label thresholds")
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.minimum_agreement_margin = minimum_agreement_margin
        self.fonts = tuple(ImageFont.truetype(str(font_path), 15*scale,
                                             layout_engine=ImageFont.Layout.BASIC)
                           for scale in _SCALES)

    @lru_cache(maxsize=128)
    def templates(self, label):
        results = []
        for font, scale in zip(self.fonts, _SCALES):
            raster = Image.new("L", (150*scale, 30*scale))
            ImageDraw.Draw(raster).text((scale, 0), label, fill=255, font=font)
            if scale != 1:
                raster = raster.resize((150, 30), Image.Resampling.LANCZOS)
            ink = _trim(np.asarray(raster, dtype=float)/255)
            if ink is None:
                return ()
            results.append(_smoothed(ink))
        return tuple(results)

    def match_label(self, label, rows):
        templates = self.templates(label)
        if not templates:
            return None, _unknown("target_font_unreadable")
        inks = [_smoothed(row.ink) for row in rows]
        scale_results = []
        for scale, template in zip(_SCALES, templates):
            scores = [_similarity(template, ink) for ink in inks]
            first, second = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
            scale_results.append({"scale": scale, "row": first, "score": scores[first],
                                  "margin": scores[first]-scores[second]})
        evidence = {"label_profile": "basic_two_scale", "label_scales": scale_results}
        if len({r["row"] for r in scale_results}) != 1:
            return None, _unknown("disagreeing_label_scales", **evidence)
        if (max(r["score"] for r in scale_results) < self.minimum_score
                or max(r["margin"] for r in scale_results) < self.minimum_margin
                or min(r["margin"] for r in scale_results) < self.minimum_agreement_margin):
            return None, _unknown("ambiguous_target_label", **evidence)
        return scale_results[0]["row"], evidence
