from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from gui_recover.tail_demo import run_tail_demo


@pytest.mark.integration
def test_portable_tail_demo_distinguishes_controls_and_preserves_clean_fields(tmp_path):
    with sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).is_file():
            pytest.skip("Install the pinned Chromium browser")
    report = run_tail_demo(tmp_path / "run")
    assert report["layout_pairs"] == report["identical_unselected_pairs"] == 12
    assert report["clean_successes"] == report["trailing_space_successes"] == 12
    assert report["clean_rewrites"] == 0
    assert report["selection_preserved_input"] == 24
    assert report["negative_control_abstentions"] == report["negative_controls"] == 8
