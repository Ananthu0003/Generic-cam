"""Stock and fixture creation service.

Pure functions — no FastAPI coupling.  Extracted from the old ``utils.py``.
"""

from __future__ import annotations

from typing import Union
import numpy as np

from ...context import Stock, StockKind, Fixture
from ..schemas import StockConfig


def create_stock_from_config(
    cfg: StockConfig,
    bmin: Union[list[float], np.ndarray],
    bmax: Union[list[float], np.ndarray],
) -> Stock:
    """Build a ``Stock`` object from an API ``StockConfig`` and part bounds."""
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)
    mode = getattr(cfg, "stock_mode", "relative_box") or "relative_box"
    off_x = getattr(cfg, "offset_x", 0.0) or 0.0
    off_y = getattr(cfg, "offset_y", 0.0) or 0.0
    off_z = getattr(cfg, "offset_z", 0.0) or 0.0

    if mode in ("relative_cylinder", "cylinder", "fixed_cylinder"):
        return _cylinder_stock(cfg, bmin, bmax, mode, off_x, off_y, off_z)
    elif mode == "fixed_box":
        return _fixed_box_stock(cfg, bmin, bmax, off_x, off_y, off_z)
    else:
        return _relative_box_stock(cfg, bmin, bmax, off_x, off_y, off_z)


def _cylinder_stock(
    cfg: StockConfig,
    bmin: np.ndarray,
    bmax: np.ndarray,
    mode: str,
    off_x: float,
    off_y: float,
    off_z: float,
) -> Stock:
    cx = 0.5 * (bmin[0] + bmax[0]) + off_x
    cy = 0.5 * (bmin[1] + bmax[1]) + off_y
    rx = 0.5 * (bmax[0] - bmin[0])
    ry = 0.5 * (bmax[1] - bmin[1])
    part_r = float(np.hypot(rx, ry))

    top_margin = cfg.cylinder_margin_axial_top if cfg.cylinder_margin_axial_top is not None else 1.0
    bot_margin = cfg.cylinder_margin_axial_bot if cfg.cylinder_margin_axial_bot is not None else 3.0
    z_top = float(bmax[2] + top_margin + off_z)

    if mode == "fixed_cylinder":
        dia = float(cfg.cylinder_diameter if (cfg.cylinder_diameter and cfg.cylinder_diameter > 0) else (2.0 * (part_r + 2.5)))
        radius = dia / 2.0
        if cfg.cylinder_length and cfg.cylinder_length > 0:
            z_bot = z_top - float(cfg.cylinder_length)
        else:
            z_bot = float(bmin[2] - bot_margin + off_z)
    else:
        radial_margin = cfg.cylinder_margin_radial if cfg.cylinder_margin_radial is not None else 2.5
        dia = float(cfg.cylinder_diameter if (cfg.cylinder_diameter and cfg.cylinder_diameter > 0) else (2.0 * (part_r + radial_margin)))
        radius = dia / 2.0
        z_bot = float(bmin[2] - bot_margin + off_z)

    axis_str = str(getattr(cfg, "cylinder_axis", "Z") or "Z").upper()
    axis_map = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0])}
    axis_vec = axis_map.get(axis_str, np.array([0.0, 0.0, 1.0]))

    return Stock.from_cylinder(
        center_xy=np.array([cx, cy]),
        z_min=z_bot,
        z_max=z_top,
        radius=radius,
        axis=axis_vec,
    )


def _fixed_box_stock(
    cfg: StockConfig,
    bmin: np.ndarray,
    bmax: np.ndarray,
    off_x: float,
    off_y: float,
    off_z: float,
) -> Stock:
    part_sx = float(bmax[0] - bmin[0])
    part_sy = float(bmax[1] - bmin[1])
    part_sz = float(bmax[2] - bmin[2])

    fx = float(cfg.fixed_size_x if (cfg.fixed_size_x and cfg.fixed_size_x > 0) else (part_sx + 2 * cfg.margin_x))
    fy = float(cfg.fixed_size_y if (cfg.fixed_size_y and cfg.fixed_size_y > 0) else (part_sy + 2 * cfg.margin_y))
    fz = float(cfg.fixed_size_z if (cfg.fixed_size_z and cfg.fixed_size_z > 0) else (part_sz + cfg.margin_z_top + cfg.margin_z_bottom))

    cx = 0.5 * (bmin[0] + bmax[0]) + off_x
    cy = 0.5 * (bmin[1] + bmax[1]) + off_y
    z_top = float(bmax[2] + cfg.margin_z_top + off_z)
    z_bot = z_top - fz

    stock_min = np.array([cx - fx / 2.0, cy - fy / 2.0, z_bot])
    stock_max = np.array([cx + fx / 2.0, cy + fy / 2.0, z_top])
    return Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)


def _relative_box_stock(
    cfg: StockConfig,
    bmin: np.ndarray,
    bmax: np.ndarray,
    off_x: float,
    off_y: float,
    off_z: float,
) -> Stock:
    stock_min = np.array([
        bmin[0] - cfg.margin_x + off_x,
        bmin[1] - cfg.margin_y + off_y,
        bmin[2] - cfg.margin_z_bottom + off_z,
    ])
    stock_max = np.array([
        bmax[0] + cfg.margin_x + off_x,
        bmax[1] + cfg.margin_y + off_y,
        bmax[2] + cfg.margin_z_top + off_z,
    ])
    return Stock(kind=StockKind.BOX, bounds_min=stock_min, bounds_max=stock_max)


def build_fixtures(sc: "SetupConfig", stock: Stock, cfg: StockConfig) -> list[Fixture]:  # noqa: F821
    """Build fixture (vise jaw / toe clamp) objects for a setup."""
    fixtures: list[Fixture] = []
    stock_min = stock.bounds_min
    stock_max = stock.bounds_max
    clamp_type = getattr(cfg, "clamp_type", "vise_jaws") or "vise_jaws"
    clamp_h = float(getattr(cfg, "clamp_height", cfg.margin_z_bottom) or cfg.margin_z_bottom or 3.0)
    clamp_w = float(getattr(cfg, "clamp_width", 12.0) or 12.0)

    if clamp_type == "vise_jaws" and clamp_h > 0:
        z_bot = float(stock_min[2])
        z_top = z_bot + clamp_h
        fixtures.extend([
            Fixture(
                id=f"{sc.id}_vise_jaw_front",
                name="Front Vise Jaw",
                bounds_min=np.array([stock_min[0] - 5.0, stock_min[1] - clamp_w, z_bot - 10.0]),
                bounds_max=np.array([stock_max[0] + 5.0, stock_min[1], z_top]),
                kind="vise_jaw",
            ),
            Fixture(
                id=f"{sc.id}_vise_jaw_rear",
                name="Rear Vise Jaw",
                bounds_min=np.array([stock_min[0] - 5.0, stock_max[1], z_bot - 10.0]),
                bounds_max=np.array([stock_max[0] + 5.0, stock_max[1] + clamp_w, z_top]),
                kind="vise_jaw",
            ),
        ])

    elif clamp_type == "toe_clamps" and clamp_h > 0:
        z_bot = float(stock_min[2])
        z_top = z_bot + clamp_h
        fixtures.extend([
            Fixture(id=f"{sc.id}_toe_1", name="Toe Clamp 1",
                    bounds_min=np.array([stock_min[0] - 10, stock_min[1] - 10, z_bot]),
                    bounds_max=np.array([stock_min[0] + 5, stock_min[1] + 5, z_top]), kind="toe_clamp"),
            Fixture(id=f"{sc.id}_toe_2", name="Toe Clamp 2",
                    bounds_min=np.array([stock_max[0] - 5, stock_min[1] - 10, z_bot]),
                    bounds_max=np.array([stock_max[0] + 10, stock_min[1] + 5, z_top]), kind="toe_clamp"),
            Fixture(id=f"{sc.id}_toe_3", name="Toe Clamp 3",
                    bounds_min=np.array([stock_min[0] - 10, stock_max[1] - 5, z_bot]),
                    bounds_max=np.array([stock_min[0] + 5, stock_max[1] + 10, z_top]), kind="toe_clamp"),
            Fixture(id=f"{sc.id}_toe_4", name="Toe Clamp 4",
                    bounds_min=np.array([stock_max[0] - 5, stock_max[1] - 5, z_bot]),
                    bounds_max=np.array([stock_max[0] + 10, stock_max[1] + 10, z_top]), kind="toe_clamp"),
        ])

    return fixtures
