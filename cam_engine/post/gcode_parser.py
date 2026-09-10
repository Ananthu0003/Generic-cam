"""G-Code parse-back verification for testing toolpath fidelity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ParsedBlock:
    line_number: int
    raw_text: str
    command: Optional[str] = None  # G0, G1, G2, G3, etc.
    target_pos: Optional[np.ndarray] = None
    feed: Optional[float] = None
    spindle: Optional[float] = None
    tool: Optional[int] = None
    is_cutting: bool = False


class GCodeValidator:
    """Parses G-code back into 3D moves to check boundaries, collisions, and continuity."""

    def __init__(self):
        self.re_word = re.compile(r"([A-Z])([-+]?[0-9]*\.?[0-9]+)")

    def parse(self, gcode: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        cur_pos = np.array([0.0, 0.0, 0.0])
        cur_feed: Optional[float] = None
        cur_spindle: Optional[float] = None
        cur_tool: Optional[int] = None
        active_motion: Optional[str] = None

        for idx, raw_line in enumerate(gcode.splitlines(), start=1):
            line = raw_line.strip()
            # strip comments in parentheses or after semicolon
            line = re.sub(r"\(.*?\)", "", line)
            line = re.sub(r";.*$", "", line).strip()
            if not line:
                continue

            matches = self.re_word.findall(line.upper())
            words = {letter: float(val) if "." in val else int(val) for letter, val in matches}

            # Motion mode
            if "G" in words:
                g_val = int(words["G"])
                if g_val in (0, 1, 2, 3):
                    active_motion = f"G{g_val}"

            if "F" in words:
                cur_feed = float(words["F"])
            if "S" in words:
                cur_spindle = float(words["S"])
            if "T" in words:
                cur_tool = int(words["T"])

            # Coordinate updates
            new_pos = cur_pos.copy()
            moved = False
            for axis_idx, axis_char in enumerate(["X", "Y", "Z"]):
                if axis_char in words:
                    new_pos[axis_idx] = float(words[axis_char])
                    moved = True

            if moved:
                cur_pos = new_pos

            is_cut = active_motion in ("G1", "G2", "G3")
            blocks.append(
                ParsedBlock(
                    line_number=idx,
                    raw_text=raw_line,
                    command=active_motion if moved else None,
                    target_pos=cur_pos.copy() if moved else None,
                    feed=cur_feed,
                    spindle=cur_spindle,
                    tool=cur_tool,
                    is_cutting=is_cut,
                )
            )

        return blocks
