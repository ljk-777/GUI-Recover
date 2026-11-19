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
