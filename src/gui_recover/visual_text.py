"""Visual comparison of one rendered input line against its requested text."""

import argparse
from base64 import b64decode, b64encode
import json
from pathlib import Path
import re

import numpy as np

from .input_guard import inspect_input
from .recovery_env import Decision, FRAME_SIZE, Phase, transition_view
from .runner import write_json
from .verification_capture import capture_after_repaint
from .verification_env import task_request


_SUPPORTED_TEXT = re.compile(r"[A-Za-z]{1,16}")
_CAPTURE = {"width":332,"height":214}
_RENDER_SCALE = 1.5
_MATCH = .995
_MISMATCH = .965
_PIXELS = 160


def _crop_ink(picture, bounds, margin):
    left, top, right, bottom = bounds
    area = np.asarray(picture.convert("L"))[top+margin:bottom-margin,left+margin:right-margin]
    ink = area < _PIXELS
    if not ink.size or ink[:, -2:].any():
        return None
    y, x = np.nonzero(ink)
    if not len(y):
        return None
    return ink[y.min():y.max()+1, x.min():x.max()+1]


def _iou(reference, observed):
    # Match small differences in glyph alignment without stretching letter
    # widths or deleting characters.
    h = max(reference.shape[0], observed.shape[0])+2
    w = max(reference.shape[1], observed.shape[1])+2
    a = np.zeros((h,w),dtype=bool)
    rh,rw = reference.shape
    oh,ow = observed.shape
    a[1:1+rh,1:1+rw] = reference
    best = 0.
    for dy in (0,1,2):
        for dx in (0,1,2):
            b = np.zeros_like(a)
            b[dy:dy+oh,dx:dx+ow] = observed
            best = max(best,np.count_nonzero(a&b)/np.count_nonzero(a|b))
    return best


def _pack(ink):
    return {"shape":list(ink.shape),"bits":b64encode(np.packbits(ink).tobytes()).decode("ascii")}


def _unpack(item):
    height,width = item["shape"]
    if type(height) is not int or type(width) is not int or not (1<=height<=32 and 1<=width<=185):
        raise ValueError("Reference shape outside text window")
    buf = b64decode(item["bits"],validate=True)
    if len(buf)!=(height*width+7)//8:
        raise ValueError("Invalid packed text reference")
    result = np.unpackbits(np.frombuffer(buf,dtype=np.uint8),count=height*width).reshape(height,width).astype(bool)
    if not result.any():
        raise ValueError("Empty reference ink")
    return result


def render_text_references(values):
    """Render requested strings on an isolated local page with no task state."""
    from playwright.sync_api import sync_playwright

    values = sorted(set(values))
    if not values or not all(_SUPPORTED_TEXT.fullmatch(v) for v in values):
        raise ValueError("Expected supported ASCII letter strings")
    css = ("body{font-family:Arial;margin:0;background:white}"
           "input{position:absolute;width:120px;caret-color:transparent}")
    # These positions cover both native pixel phases at 1.5x, in each axis.
    controls = "".join(f'<input type="text" style="left:{x}px;top:{y}px">'
                       for x in (20,171) for y in (12,43))
    templates = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport=_CAPTURE)
            page._bgym_scale_factor = _RENDER_SCALE
            page.set_content(f"<style>{css}</style>{controls}")
            bounds = page.locator("input").evaluate_all(
                "els => els.map(e=>{const r=e.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]})")
            bounds = [tuple(int(v*_RENDER_SCALE) for v in row) for row in bounds]
            for value in values:
                page.evaluate("(value)=>document.querySelectorAll('input').forEach(el=>{el.value=value})",value)
                frame = capture_after_repaint(page)
                refs = [_crop_ink(frame, region,4) for region in bounds]
                if any(ink is None for ink in refs):
                    raise ValueError("Reference text falls outside the template window")
                templates[value] = [_pack(ink) for ink in refs]
            return {"schema_version":1,"browser_version":browser.version,"pixel_scale":_RENDER_SCALE,
                    "thresholds":{"match":_MATCH,"mismatch":_MISMATCH,"ink":_PIXELS},
                    "templates":templates}
        finally:
            browser.close()


class VisualTextVerifier:
    """Check glyph evidence in an initially blank native Chromium input field."""

    def __init__(self, bank):
        if (bank["schema_version"]!=1 or bank["pixel_scale"]!=_RENDER_SCALE or
            bank["thresholds"]!={"match":_MATCH,"mismatch":_MISMATCH,"ink":_PIXELS}):
            raise ValueError("Unsupported text-reference bank")
        self.templates = {}
        for value, items in bank["templates"].items():
            if not _SUPPORTED_TEXT.fullmatch(value) or len(items)!=4:
                raise ValueError("Invalid text reference")
            self.templates[value] = tuple(_unpack(item) for item in items)

    def verify(self, view):
        if (view.proposed.action!="INPUT" or view.target!="text field" or view.after.size!=FRAME_SIZE
                or view.goal!=f'Enter "{view.proposed.value}" into the text field and press Submit.'):
            return {"status":"unknown","reason":"unsupported_input_profile"}
        templates = self.templates.get(view.proposed.value)
        if not templates:
            return {"status":"unknown","reason":"missing_text_reference"}
        guard = inspect_input(view.before,view.after,view.proposed.position,version="v2")
        if guard["status"]=="unknown":
            return {"status":"unknown","reason":guard["reason"]}
        if guard["status"]=="empty":
            return {"status":"empty","reason":"no_visible_ink","region":guard["region"]}
        region = guard["region"]
        observed = _crop_ink(view.after,region,2)
        if observed is None:
            return {"status":"unknown","reason":"clipped_or_unreadable_ink","region":region}
        score = max(_iou(template,observed) for template in templates)
        status = "matched" if score>=_MATCH else "different" if score<=_MISMATCH else "unknown"
        return {"status":status,"reason":"rendered_text_comparison","region":region,
                "score":score,"ink_shape":list(observed.shape)}


class ContentRecoveryPolicy:
    def __init__(self, verifier):
        self.verifier = verifier
        self.evidence = []

    def __call__(self, observation):
        if observation["phase"] != Phase.REVIEW:
            return Decision.CONTINUE
        result = self.verifier.verify(transition_view(observation))
        status = result["status"]
        if observation["action_mask"][Decision.WAIT] and status in ("empty","unknown"):
            action = Decision.WAIT
        elif status in ("empty","different"):
            action = Decision.RETRY_FIXED_ZOOM
        else:
            action = Decision.CONTINUE
        self.evidence.append({**result,"waited":bool(observation["waited"]),"decision":action.name})
        return action


def main():
    parser = argparse.ArgumentParser(description="Render references from the preflight's visible goals")
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError("Output bank already exists")
    plan=json.loads(args.plan.read_text())
    values=[task_request(r["task"],r["goal"])[-1] for r in plan["selected"]]
    write_json(args.out,render_text_references(values))


if __name__ == "__main__":
    main()
