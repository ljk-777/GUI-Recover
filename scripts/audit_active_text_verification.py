"""Replay recorded visual decisions and verify the bounded tail probe on CPU."""

import sys

from gui_recover.cli import main


if __name__ == "__main__":
    main(["audit-active-text-recovery", *sys.argv[1:]])
