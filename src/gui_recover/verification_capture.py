"""Native CDP screenshots with a repaint barrier for the verification fixture."""

import base64
import io

from PIL import Image


def capture_after_repaint(page):
    viewport=page.viewport_size
    if viewport is None:
        viewport=page.evaluate("() => ({width:innerWidth,height:innerHeight})")
    dimensions={"width":viewport["width"],"height":viewport["height"],"mobile":False}
    scale=getattr(page,"_bgym_scale_factor",1.0)
    cdp=page.context.new_cdp_session(page)
    try:
        cdp.send("Emulation.setDeviceMetricsOverride",{**dimensions,"deviceScaleFactor":scale})
        # BrowserGym 0.14.3 captures immediately after the metric override.
        # Wait for paint at the requested scale without modifying image pixels.
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        result=cdp.send("Page.captureScreenshot",{"format":"png"})
    finally:
        cdp.send("Emulation.setDeviceMetricsOverride",{**dimensions,"deviceScaleFactor":1.0})
        cdp.detach()
    return Image.open(io.BytesIO(base64.b64decode(result["data"]))).convert("RGB")


def capture_selected_tail(page):
    """Capture the final character selection, then collapse it to the right."""
    page.keyboard.press("Shift+ArrowLeft")
    try:
        return capture_after_repaint(page)
    finally:
        page.keyboard.press("ArrowRight")


def capture_tail_selection():
    """Temporarily select the last character and save a native screenshot.

    Examples:
        capture_tail_selection()
    """
    from gui_recover.verification_capture import capture_selected_tail

    # BrowserGym supplies its current Playwright page to custom actions.
    page._gui_recover_tail_selection = capture_selected_tail(page)  # noqa: F821
