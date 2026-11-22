import json
import re
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_portable_tail_demo_distinguishes_controls_and_preserves_clean_fields(tmp_path):
    out = tmp_path / "run"
    # The standalone command owns its browser loop. BrowserGym tests in the
    # parent pytest process use a separate, shared Playwright session.
    subprocess.run([sys.executable, "-m", "gui_recover", "demo-tail-probe", "--out", str(out)], check=True)
    report = json.loads((out / "summary.json").read_text())
    assert report["layout_pairs"] == report["identical_unselected_pairs"] == 12
    assert report["clean_successes"] == report["trailing_space_successes"] == 12
    assert report["clean_rewrites"] == 0
    assert report["selection_preserved_input"] == 24
    assert report["negative_control_abstentions"] == report["negative_controls"] == 8
    html = (out / "report.html").read_text()
    images = re.findall(r'<img src="([^"]+)"', html)
    assert "Trailing-space recovery" in html and len(images) == 24
    assert all((out / image).is_file() for image in images)
