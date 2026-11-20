"""Verify saved visual-text rollouts on CPU, including interrupted runs."""

import sys

from gui_recover.cli import main


if __name__ == "__main__":
    main(["audit-text-recovery", *sys.argv[1:]])
