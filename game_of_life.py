"""Modern Game of Life with PySide6 GUI.

Run:
  python game_of_life.py

Build single-file exe (Windows or Linux, run on the target OS):
  pip install -r requirements.txt
  pyinstaller --noconfirm --onefile --windowed game_of_life.py --name game-of-life

Controls:
- Start/Pause: toggle simulation
- Step: advance one generation
- Randomize: fill grid with noise (density slider controls % alive)
- Clear: empty grid
- Wrap: toroidal edges on/off
- Speed: update interval in ms
- Patterns: insert predefined shapes at center
- Click/drag on the grid to toggle cells
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from modules.environment import EnvironmentConfig, EnvironmentEngine
from modules.resource_map import ResourceMapManager
from modules.species_policy import SpeciesPolicyManager
from modules.policy_viz import PolicyVizWindow

LOG_PATH = Path(__file__).with_name("debug") / "debug.txt"
logger = logging.getLogger("gol")
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Truncate log each app start so we only see the current session.
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def log_debug(msg: str, **data: object) -> None:
    if data:
        parts = [f"{k}={v}" for k, v in data.items()]
        logger.debug(f"{msg} | " + ", ".join(parts))
    else:
        logger.debug(msg)

# Colors
# UI background lighter dark; grid panel still drawn dark in the widget
BACKGROUND = QtGui.QColor(32, 36, 44)
GRID_COLOR = QtGui.QColor(50, 60, 70)
BARRIER_COLOR = QtGui.QColor(200, 50, 50)
SPECIES_COLORS = [
    QtGui.QColor(0, 200, 255),   # species 1
    QtGui.QColor(255, 180, 0),   # species 2
    QtGui.QColor(120, 255, 80),  # species 3
    QtGui.QColor(200, 120, 255), # species 4
    QtGui.QColor(255, 90, 160),  # species 5
]


def lerp_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = max(0.0, min(1.0, t))
    return QtGui.QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


class LifeWidget(QtWidgets.QWidget):
    def __init__(self, rows: int = 80, cols: int = 110, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.rows = rows
        self.cols = cols
        self.env_engine = EnvironmentEngine(rows, cols, EnvironmentConfig())
        self.resource_map = self.env_engine.resource_map
        # grid values: 0 empty, 1-5 species, -1 barrier
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.age = np.zeros((rows, cols), dtype=np.int32)
        self.wrap = True
        self.view_resources = False
        self._mouse_down = False
        self.setMouseTracking(True)
        self.policy_mgr: SpeciesPolicyManager | None = None
        self.use_policy = False
        self.brush_state: int | None = None  # None means cycle
        self.randomize_resources()
        log_debug("LifeWidget initialized", rows=rows, cols=cols)

    def set_wrap(self, wrap: bool) -> None:
        self.wrap = wrap

    def set_policy_manager(self, mgr: SpeciesPolicyManager | None, enabled: bool) -> None:
        self.policy_mgr = mgr
        self.use_policy = enabled
        log_debug("Policy manager toggled", enabled=enabled)

    def clear(self) -> None:
        self.grid[:] = 0
        self.age[:] = 0
        self.env_engine.reset()
        self.update()
        log_debug("Board cleared")

    def randomize(self, density: float = 0.2, allowed_species: Sequence[int] | None = None) -> None:
        alive_mask = np.random.rand(self.rows, self.cols) < density
        allowed = list(allowed_species) if allowed_species else [1, 2, 3, 4, 5]
        species_choices = np.random.choice(allowed, size=(self.rows, self.cols)).astype(np.int8)
        self.grid = np.where(alive_mask, species_choices, 0)
        self.age[:] = 0
        self.env_engine.reset()
        self.randomize_resources()
        self.update()
        log_debug("Board randomized", density=density, alive=int((self.grid > 0).sum()))

    def randomize_resources(self) -> None:
        self.resource_map.randomize()
        self.update()
        grid = self.resource_map.grid
        log_debug("Resources randomized", rmin=float(grid.min()), rmax=float(grid.max()), rmean=float(grid.mean()))

    def fill_resources(self, value: float) -> None:
        self.resource_map.fill(value)
        self.update()
        grid = self.resource_map.grid
        log_debug("Resources filled", value=value, rmin=float(grid.min()), rmax=float(grid.max()), rmean=float(grid.mean()))

    def set_view_resources(self, enabled: bool) -> None:
        self.view_resources = enabled
        if enabled and float(self.resource_map.grid.max()) <= 0.0:
            self.randomize_resources()
        self.update()
        log_debug("Resource overlay toggled", enabled=enabled, rmax=float(self.resource_map.grid.max()))

    def insert_pattern(self, pattern: List[Tuple[int, int]]) -> None:
        r0 = self.rows // 2
        c0 = self.cols // 2
        for dr, dc in pattern:
            r = (r0 + dr) % self.rows
            c = (c0 + dc) % self.cols
            self.grid[r, c] = 1
            self.age[r, c] = 0
        self.update()

    def step_with_delta(self) -> tuple[np.ndarray, np.ndarray]:
        g_prev = self.grid.copy()
        new_grid, _info = self.env_engine.step(g_prev, self.wrap, self.policy_mgr, self.use_policy)

        same_species = (new_grid == g_prev) & (new_grid > 0)
        births_any = (new_grid > 0) & (~same_species)
        self.age = np.where(new_grid > 0, np.where(births_any, 1, self.age + 1), 0)
        self.grid = new_grid
        self.update()
        log_debug(
            "Step computed",
            wrap=self.wrap,
            alive=int((self.grid > 0).sum()),
            resources_mean=float(self.resource_map.grid.mean()),
        )
        return g_prev, new_grid

    # --- painting ---
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(16, 20, 26))
        w = self.width()
        h = self.height()
        cw = w / self.cols
        ch = h / self.rows

        if not self.view_resources:
            painter.setPen(QtGui.QPen(GRID_COLOR, 1))
            for c in range(self.cols + 1):
                x = int(c * cw)
                painter.drawLine(x, 0, x, h)
            for r in range(self.rows + 1):
                y = int(r * ch)
                painter.drawLine(0, y, w, y)

        if self.view_resources:
            map_max = float(np.max(self.resource_map.grid))
            if map_max <= 0.0:
                # fill with a visible baseline to ensure rendering is obvious
                self.resource_map.fill(self.resource_map.layer.config.max_capacity)
                map_max = float(np.max(self.resource_map.grid))
            scale_max = max(map_max, 0.001)
            any_res = map_max > 0.0
            map_min = float(np.min(self.resource_map.grid))

        for r in range(self.rows):
            for c in range(self.cols):
                rect = QtCore.QRect(int(c * cw) + 1, int(r * ch) + 1, int(cw) - 1, int(ch) - 1)
                if self.view_resources:
                    val = float(self.resource_map.grid[r, c])
                    norm = min(1.0, val / scale_max)
                    g_val = int(10 + 245 * norm)
                    painter.fillRect(rect, QtGui.QColor(0, g_val, 0))
                    continue
                val = self.grid[r, c]
                if val == 0:
                    continue
                if val == -1:
                    painter.fillRect(rect, QtGui.QColor(80, 0, 0, 120))
                    pen = QtGui.QPen(BARRIER_COLOR, 2)
                    painter.setPen(pen)
                    painter.drawLine(rect.topLeft(), rect.bottomRight())
                    painter.drawLine(rect.bottomLeft(), rect.topRight())
                    painter.setPen(QtGui.QPen())
                    continue
                species_idx = val - 1
                base = SPECIES_COLORS[species_idx % len(SPECIES_COLORS)]
                age = self.age[r, c]
                t = min(age / 12.0, 1.0)
                color = lerp_color(base, QtGui.QColor(255, 255, 255), t)
                painter.fillRect(rect, color)

        if self.view_resources:
            painter.setPen(QtGui.QColor(200, 255, 200))
            if not any_res:
                painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No resources — click Gen Resources")
            else:
                painter.drawText(12, 22, f"Resources: min {map_min:.2f} max {map_max:.2f}")
                # draw a small legend bar
                legend_x = 12
                legend_y = 30
                legend_w = 180
                legend_h = 12
                for i in range(legend_w):
                    t = i / max(1, legend_w - 1)
                    g_val = int(10 + 245 * t)
                    painter.fillRect(QtCore.QRect(legend_x + i, legend_y, 1, legend_h), QtGui.QColor(0, g_val, 0))
                painter.drawRect(QtCore.QRect(legend_x, legend_y, legend_w, legend_h))

    # --- mouse input ---
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._mouse_down = True
        self._toggle_from_event(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mouse_down:
            self._toggle_from_event(event)

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._mouse_down = False

    def _toggle_from_event(self, event: QtGui.QMouseEvent) -> None:
        w = self.width()
        h = self.height()
        cw = w / self.cols
        ch = h / self.rows
        c = int(event.position().x() // cw)
        r = int(event.position().y() // ch)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.brush_state is None:
                current = self.grid[r, c]
                sequence = [-1, 0, 1, 2, 3, 4, 5]  # barrier -> empty -> species1..5
                idx = sequence.index(int(current)) if int(current) in sequence else 1
                next_state = sequence[(idx + 1) % len(sequence)]
                self.grid[r, c] = next_state
                self.age[r, c] = 0 if next_state <= 0 else 1
            else:
                self.grid[r, c] = self.brush_state
                self.age[r, c] = 0 if self.brush_state <= 0 else 1
            self.update()

    def set_brush_state(self, state: int | None) -> None:
        self.brush_state = state


class ControlPanel(QtWidgets.QWidget):
    start_clicked = QtCore.Signal()
    pause_clicked = QtCore.Signal()
    step_clicked = QtCore.Signal()
    clear_clicked = QtCore.Signal()
    random_clicked = QtCore.Signal(float)
    wrap_changed = QtCore.Signal(bool)
    speed_changed = QtCore.Signal(int)
    pattern_selected = QtCore.Signal(str)
    policy_toggle = QtCore.Signal(bool)
    res_random_clicked = QtCore.Signal()
    res_fill_clicked = QtCore.Signal(float)
    res_view_toggle = QtCore.Signal(bool)
    policy_viz_clicked = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.random_btn = QtWidgets.QPushButton("Random")
        self.res_random_btn = QtWidgets.QPushButton("Gen Resources")
        self.res_fill_btn = QtWidgets.QPushButton("Fill Resources")
        self.res_view_box = QtWidgets.QCheckBox("View Resources")
        self.policy_viz_btn = QtWidgets.QPushButton("Policy Viz")

        for btn in [self.start_btn, self.pause_btn, self.step_btn, self.clear_btn, self.random_btn]:
            layout.addWidget(btn)

        layout.addWidget(self.res_random_btn)
        layout.addWidget(self.res_fill_btn)
        layout.addWidget(self.res_view_box)
        layout.addWidget(self.policy_viz_btn)

        layout.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 800)
        self.speed_slider.setValue(120)
        self.speed_slider.setFixedWidth(140)
        layout.addWidget(self.speed_slider)

        layout.addWidget(QtWidgets.QLabel("Density:"))
        self.density_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.density_slider.setRange(5, 90)
        self.density_slider.setValue(20)
        self.density_slider.setFixedWidth(120)
        layout.addWidget(self.density_slider)

        self.wrap_box = QtWidgets.QCheckBox("Wrap")
        self.wrap_box.setChecked(True)
        layout.addWidget(self.wrap_box)

        layout.addWidget(QtWidgets.QLabel("Pattern:"))
        self.pattern_box = QtWidgets.QComboBox()
        self.pattern_box.addItems(["None", "Glider", "Small Exploder", "Pulsar", "Gosper Gun"])
        layout.addWidget(self.pattern_box)

        self.policy_box = QtWidgets.QCheckBox("Learning")
        layout.addWidget(self.policy_box)

        layout.addStretch(1)

        self.start_btn.clicked.connect(self.start_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.step_btn.clicked.connect(self.step_clicked)
        self.clear_btn.clicked.connect(self.clear_clicked)
        self.random_btn.clicked.connect(self._emit_random)
        self.res_random_btn.clicked.connect(lambda: self.res_random_clicked.emit())
        self.res_fill_btn.clicked.connect(self._emit_res_fill)
        self.res_view_box.toggled.connect(lambda checked: (log_debug("View resources toggled", checked=checked), self.res_view_toggle.emit(checked))[1])
        self.wrap_box.stateChanged.connect(lambda s: self.wrap_changed.emit(s == QtCore.Qt.CheckState.Checked))
        self.speed_slider.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        self.pattern_box.currentTextChanged.connect(self.pattern_selected)
        # use toggled(bool) to avoid enum quirks
        self.policy_box.toggled.connect(lambda checked: self.policy_toggle.emit(bool(checked)))
        self.policy_viz_btn.clicked.connect(lambda: self.policy_viz_clicked.emit())

    def _emit_random(self) -> None:
        density = self.density_slider.value() / 100.0
        self.random_clicked.emit(density)

    def _emit_res_fill(self) -> None:
        val, ok = QtWidgets.QInputDialog.getDouble(self, "Fill resources", "Value per cell", 1.0, 0.0, 10.0, 2)
        if ok:
            self.res_fill_clicked.emit(val)


PATTERNS: Dict[str, List[Tuple[int, int]]] = {
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "Small Exploder": [(-1, 0), (0, -1), (0, 0), (0, 1), (1, -1), (1, 1), (2, 0)],
    "Pulsar": [
        (-6, -4), (-6, -3), (-6, -2), (-6, 2), (-6, 3), (-6, 4),
        (-4, -6), (-3, -6), (-2, -6), (2, -6), (3, -6), (4, -6),
        (-1, -4), (-1, -3), (-1, -2), (-1, 2), (-1, 3), (-1, 4),
        (1, -4), (1, -3), (1, -2), (1, 2), (1, 3), (1, 4),
        (-4, 6), (-3, 6), (-2, 6), (2, 6), (3, 6), (4, 6),
        (6, -4), (6, -3), (6, -2), (6, 2), (6, 3), (6, 4),
    ],
    "Gosper Gun": [
        (0, 24),
        (1, 22), (1, 24),
        (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
        (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
        (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
        (6, 10), (6, 16), (6, 24),
        (7, 11), (7, 15),
        (8, 12), (8, 13),
    ],
}


class BrushPanel(QtWidgets.QGroupBox):
    brush_changed = QtCore.Signal(object)  # emits int or None

    def __init__(self) -> None:
        super().__init__("Brush")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)

        def add_button(text: str, value: int | None) -> None:
            btn = QtWidgets.QRadioButton(text)
            layout.addWidget(btn)
            self.group.addButton(btn)
            btn.value = value  # type: ignore[attr-defined]
            btn.toggled.connect(lambda checked, v=value: checked and self.brush_changed.emit(v))

        add_button("Cycle", None)
        add_button("Erase", 0)
        add_button("Barrier", -1)
        add_button("Species 1", 1)
        add_button("Species 2", 2)
        add_button("Species 3", 3)
        add_button("Species 4", 4)
        add_button("Species 5", 5)

        # default to Cycle
        self.group.buttons()[0].setChecked(True)
        layout.addStretch(1)
        self.setStyleSheet(
            "QGroupBox { color: white; font-weight: bold; }"
            "QLabel { color: white; }"
            "QRadioButton { color: white; }"
        )


class SpawnPanel(QtWidgets.QGroupBox):
    allowed_changed = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__("Spawn Control")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QtWidgets.QLabel("Species eligible for Random")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        for sid in range(1, 6):
            cb = QtWidgets.QCheckBox(f"Species {sid}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._emit_allowed)
            layout.addWidget(cb)
            self.checkboxes[sid] = cb

        btn_row = QtWidgets.QHBoxLayout()
        sel_all = QtWidgets.QPushButton("All")
        sel_none = QtWidgets.QPushButton("None")
        sel_all.clicked.connect(self._select_all)
        sel_none.clicked.connect(self._select_none)
        btn_row.addWidget(sel_all)
        btn_row.addWidget(sel_none)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        self.setStyleSheet(
            "QGroupBox { color: white; font-weight: bold; }"
            "QLabel { color: white; }"
            "QCheckBox { color: white; }"
            "QPushButton { color: white; background-color: #444; border: 1px solid #666; padding: 4px; }"
            "QPushButton:hover { background-color: #555; }"
        )

    def allowed_species(self) -> list[int]:
        allowed = [sid for sid, cb in self.checkboxes.items() if cb.isChecked()]
        return allowed or [1, 2, 3, 4, 5]

    def _emit_allowed(self) -> None:
        self.allowed_changed.emit(self.allowed_species())

    def _select_all(self) -> None:
        for cb in self.checkboxes.values():
            cb.setChecked(True)
        self._emit_allowed()

    def _select_none(self) -> None:
        for cb in self.checkboxes.values():
            cb.setChecked(False)
        self._emit_allowed()


class StatsPanel(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Stats")
        self.labels: Dict[int, QtWidgets.QLabel] = {}
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        for s in range(1, 6):
            lbl = QtWidgets.QLabel("Alive 0 | Deaths 0 | Kills 0")
            layout.addRow(f"Species {s}", lbl)
            self.labels[s] = lbl
        self.setStyleSheet(
            "QGroupBox { color: white; font-weight: bold; }"
            "QLabel { color: white; }"
        )

    def update_counts(self, alive: np.ndarray, deaths: np.ndarray, kills: np.ndarray) -> None:
        for s in range(1, 6):
            self.labels[s].setText(f"Alive {alive[s-1]} | Deaths {deaths[s-1]} | Kills {kills[s-1]}")


class ConfigPanel(QtWidgets.QGroupBox):
    params_changed = QtCore.Signal(dict)

    def __init__(self, env_cfg: EnvironmentConfig) -> None:
        super().__init__("Settings")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.resource_tab = QtWidgets.QWidget()
        res_form = QtWidgets.QFormLayout(self.resource_tab)
        res_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.res_regen = self._spin(res_form, "Regen / step", env_cfg.resource.regen_rate, 0.0, 5.0, 0.05)
        self.res_max = self._spin(res_form, "Max capacity", env_cfg.resource.max_capacity, 0.1, 20.0, 0.1)
        self.res_base = self._spin(res_form, "Baseline", env_cfg.resource.baseline, 0.0, 10.0, 0.1)
        self.res_rng_low = self._spin(res_form, "Rand low", env_cfg.resource.rng_low, 0.0, 10.0, 0.1)
        self.res_rng_high = self._spin(res_form, "Rand high", env_cfg.resource.rng_high, 0.0, 10.0, 0.1)
        self.tabs.addTab(self.resource_tab, "Resources")

        self.energy_tab = QtWidgets.QWidget()
        en_form = QtWidgets.QFormLayout(self.energy_tab)
        en_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.energy_decay = self._spin(en_form, "Energy decay", env_cfg.energy_decay, 0.0, 2.0, 0.01)
        self.energy_conv = self._spin(en_form, "Resource->Energy", env_cfg.energy_conversion, 0.1, 5.0, 0.05)
        self.cost_stay = self._spin(en_form, "Stay cost", env_cfg.stay_cost, 0.0, 2.0, 0.01)
        self.cost_takeover = self._spin(en_form, "Takeover cost", env_cfg.takeover_cost, 0.0, 3.0, 0.01)
        self.cost_birth = self._spin(en_form, "Birth action cost", env_cfg.birth_action_cost, 0.0, 3.0, 0.01)
        self.tabs.addTab(self.energy_tab, "Energy")

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._emit_params)
        layout.addWidget(apply_btn)
        layout.addStretch(1)

    def _spin(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        value: float,
        lo: float,
        hi: float,
        step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(3)
        box.setValue(value)
        form.addRow(label, box)
        return box

    def _emit_params(self) -> None:
        params = {
            "resource": {
                "regen_rate": self.res_regen.value(),
                "max_capacity": self.res_max.value(),
                "baseline": self.res_base.value(),
                "rng_low": self.res_rng_low.value(),
                "rng_high": self.res_rng_high.value(),
            },
            "energy": {
                "energy_decay": self.energy_decay.value(),
                "energy_conversion": self.energy_conv.value(),
                "stay_cost": self.cost_stay.value(),
                "takeover_cost": self.cost_takeover.value(),
                "birth_action_cost": self.cost_birth.value(),
            },
        }
        self.params_changed.emit(params)


class PanelToggleBar(QtWidgets.QWidget):
    panel_toggled = QtCore.Signal(str, bool)

    def __init__(self, panels: Sequence[Tuple[str, str, bool]]) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(12)
        self._boxes: Dict[str, QtWidgets.QCheckBox] = {}
        for key, label, default in panels:
            box = QtWidgets.QCheckBox(label)
            box.setChecked(default)
            box.toggled.connect(lambda checked, k=key: self.panel_toggled.emit(k, checked))
            layout.addWidget(box)
            self._boxes[key] = box
        layout.addStretch(1)

    def set_checked(self, key: str, value: bool) -> None:
        box = self._boxes.get(key)
        if not box:
            return
        if box.isChecked() == value:
            return
        box.blockSignals(True)
        box.setChecked(value)
        box.blockSignals(False)
        self.panel_toggled.emit(key, value)


class InsightsPanel(QtWidgets.QGroupBox):
    metric_changed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__("Insights")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.currentIndexChanged.connect(self._emit_metric_changed)
        selector_row.addWidget(self.metric_combo, 1)
        layout.addLayout(selector_row)

        self.canvas = FigureCanvas(Figure(figsize=(4, 3), tight_layout=True))
        self.canvas.setMinimumHeight(220)
        self.ax = self.canvas.figure.subplots()
        self.ax.set_facecolor("#101010")
        layout.addWidget(self.canvas, 1)
        layout.addStretch(1)

    def set_metrics(self, metrics: Sequence[Tuple[str, str]]) -> None:
        current = self.current_metric_key()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for label, key in metrics:
            self.metric_combo.addItem(label, key)
        self.metric_combo.blockSignals(False)
        if current and current in [key for _, key in metrics]:
            index = [key for _, key in metrics].index(current)
            self.metric_combo.setCurrentIndex(index)
        elif self.metric_combo.count() > 0:
            self.metric_combo.setCurrentIndex(0)

    def current_metric_key(self) -> str | None:
        idx = self.metric_combo.currentIndex()
        if idx < 0:
            return None
        return self.metric_combo.itemData(idx)

    def plot_series(self, steps: Sequence[int], values: Sequence[int], ylabel: str) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#101010")
        if steps and values:
            self.ax.plot(steps, values, color="#7fc7ff", linewidth=2)
            self.ax.set_xlabel("Steps")
            self.ax.set_ylabel(ylabel)
            self.ax.grid(True, alpha=0.3)
        else:
            self.ax.text(0.5, 0.5, "No data yet", color="#cccccc", ha="center", va="center", transform=self.ax.transAxes)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _emit_metric_changed(self, index: int) -> None:  # noqa: ARG002 - Qt callback
        key = self.metric_combo.itemData(index)
        if key is not None:
            self.metric_changed.emit(key)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Game of Life - Modern")
        self.board = LifeWidget()
        self.controls = ControlPanel()
        self.brush_panel = BrushPanel()
        self.spawn_panel = SpawnPanel()
        self.stats_panel = StatsPanel()
        self.insights_panel = InsightsPanel()
        self.config_panel = ConfigPanel(self.board.env_engine.cfg)
        self.brush_panel.setMinimumWidth(180)
        self.spawn_panel.setMinimumWidth(180)
        self.stats_panel.setMinimumWidth(220)
        self.insights_panel.setMinimumWidth(260)
        self.config_panel.setMinimumWidth(260)

        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(container)
        vlayout.setContentsMargins(4, 4, 4, 4)
        vlayout.setSpacing(6)
        vlayout.addWidget(self.controls)

        self.panel_toggle_bar = PanelToggleBar(
            [
                ("spawn", "Spawn Panel", False),
                ("brush", "Brush Panel", True),
                ("stats", "Stats Panel", True),
                ("insights", "Insights Panel", False),
                ("config", "Config Panel", False),
            ]
        )
        vlayout.addWidget(self.panel_toggle_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        self.spawn_scroll = QtWidgets.QScrollArea()
        self.spawn_scroll.setWidgetResizable(True)
        self.spawn_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.spawn_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.spawn_scroll.setWidget(self.spawn_panel)

        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setWidget(self.brush_panel)

        self.right_scroll = QtWidgets.QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_scroll.setWidget(self.stats_panel)

        self.insights_scroll = QtWidgets.QScrollArea()
        self.insights_scroll.setWidgetResizable(True)
        self.insights_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.insights_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.insights_scroll.setWidget(self.insights_panel)

        self.config_scroll = QtWidgets.QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.config_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.config_scroll.setWidget(self.config_panel)

        splitter.addWidget(self.spawn_scroll)
        splitter.addWidget(self.left_scroll)
        splitter.addWidget(self.board)
        splitter.addWidget(self.right_scroll)
        splitter.addWidget(self.insights_scroll)
        splitter.addWidget(self.config_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 0)
        splitter.setStretchFactor(4, 0)
        splitter.setStretchFactor(5, 0)

        vlayout.addWidget(splitter, 1)

        self.setCentralWidget(container)

        self.panel_widgets = {
            "spawn": self.spawn_scroll,
            "brush": self.left_scroll,
            "stats": self.right_scroll,
            "insights": self.insights_scroll,
            "config": self.config_scroll,
        }
        self.panel_toggle_bar.panel_toggled.connect(self._handle_panel_toggle)
        self.insights_panel.metric_changed.connect(self._update_insights_plot)
        self.spawn_scroll.hide()
        self.insights_scroll.hide()
        self.config_panel.params_changed.connect(self._apply_params)
        self.config_scroll.hide()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)

        # policy manager (per-species).
        self.policy_mgr = SpeciesPolicyManager.create(
            [1, 2, 3, 4, 5],
            hidden_layers=[512, 512, 256],
            reset_arch=True,
        )
        self.board.set_policy_manager(self.policy_mgr, enabled=False)
        self.allowed_species = self.spawn_panel.allowed_species()

        self.res_window: ResourceWindow | None = None
        self.policy_window: PolicyVizWindow | None = None

        # stats trackers
        self.alive_counts = np.zeros(5, dtype=np.int64)
        self.death_counts = np.zeros(5, dtype=np.int64)
        self.kill_counts = np.zeros(5, dtype=np.int64)

        self._recompute_alive()
        self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)

        self._connect_signals()

        self.status = self.statusBar()
        self.gen = 0

        self.metric_options = self._metric_options()
        self.metric_labels = {key: label for label, key in self.metric_options}
        self.metric_keys = [key for _, key in self.metric_options]
        self._init_history_storage()
        self.insights_panel.set_metrics(self.metric_options)
        self._record_history(0)

        self._update_status()

    def _current_interval_ms(self) -> int:
        # Invert slider: higher slider value = faster (smaller interval)
        v = self.controls.speed_slider.value()
        return self.controls.speed_slider.maximum() - v + self.controls.speed_slider.minimum()

    def _metric_options(self) -> List[Tuple[str, str]]:
        options: List[Tuple[str, str]] = [
            ("Alive (total)", "alive_total"),
            ("Deaths (total)", "deaths_total"),
            ("Kills (total)", "kills_total"),
        ]
        for s in range(1, 6):
            options.append((f"Species {s} alive", f"alive_{s}"))
            options.append((f"Species {s} deaths", f"deaths_{s}"))
            options.append((f"Species {s} kills", f"kills_{s}"))
        return options

    def _init_history_storage(self) -> None:
        self.stat_history: Dict[str, List[int]] = {"steps": []}
        for key in self.metric_keys:
            self.stat_history[key] = []

    def _record_history(self, step: int) -> None:
        if not hasattr(self, "stat_history"):
            return
        self.stat_history["steps"].append(step)
        self.stat_history["alive_total"].append(int(self.alive_counts.sum()))
        self.stat_history["deaths_total"].append(int(self.death_counts.sum()))
        self.stat_history["kills_total"].append(int(self.kill_counts.sum()))
        for idx in range(5):
            species = idx + 1
            self.stat_history[f"alive_{species}"].append(int(self.alive_counts[idx]))
            self.stat_history[f"deaths_{species}"].append(int(self.death_counts[idx]))
            self.stat_history[f"kills_{species}"].append(int(self.kill_counts[idx]))
        self._update_insights_plot()

    def _reset_history_and_snapshot(self) -> None:
        self._init_history_storage()
        self._record_history(self.gen)

    def _update_insights_plot(self) -> None:
        if not hasattr(self, "stat_history"):
            return
        metric_key = self.insights_panel.current_metric_key()
        if not metric_key:
            self.insights_panel.plot_series([], [], "")
            return
        steps = self.stat_history.get("steps", [])
        values = self.stat_history.get(metric_key, [])
        label = self.metric_labels.get(metric_key, "Value")
        self.insights_panel.plot_series(steps, values, label)

    def _handle_panel_toggle(self, key: str, visible: bool) -> None:
        widget = self.panel_widgets.get(key)
        if not widget:
            return
        widget.setVisible(visible)
        if key == "insights" and visible:
            self._update_insights_plot()

    def _connect_signals(self) -> None:
        self.controls.start_clicked.connect(self.start)
        self.controls.pause_clicked.connect(self.pause)
        self.controls.step_clicked.connect(self.step_once)
        self.controls.clear_clicked.connect(self.clear)
        self.controls.random_clicked.connect(self.randomize)
        self.controls.res_random_clicked.connect(self._randomize_resources)
        self.controls.res_fill_clicked.connect(self._fill_resources)
        self.controls.res_view_toggle.connect(self._toggle_resource_view)
        self.controls.wrap_changed.connect(self.set_wrap)
        self.controls.speed_changed.connect(self.set_speed)
        self.controls.pattern_selected.connect(self.insert_pattern)
        self.controls.policy_toggle.connect(self.toggle_policy)
        self.controls.policy_viz_clicked.connect(self._show_policy_viz)
        self.brush_panel.brush_changed.connect(self._set_brush)
        self.spawn_panel.allowed_changed.connect(self._update_allowed_species)

    def _tick(self) -> None:
        # ensure toggle state is respected even if a signal was missed
        desired_learning = self.controls.policy_box.isChecked()
        if self.board.use_policy != desired_learning:
            self.board.set_policy_manager(self.policy_mgr, desired_learning)
            log_debug("Learning state synced on tick", desired=desired_learning)
        prev, new = self.board.step_with_delta()
        self._update_stats(prev, new)
        self.gen += 1
        self._update_status()

    def start(self) -> None:
        if not self.timer.isActive():
            self.timer.start(self._current_interval_ms())
        self.status.showMessage("Running", 1500)

    def pause(self) -> None:
        self.timer.stop()
        self.status.showMessage("Paused", 1500)

    def step_once(self) -> None:
        self.timer.stop()
        desired_learning = self.controls.policy_box.isChecked()
        if self.board.use_policy != desired_learning:
            self.board.set_policy_manager(self.policy_mgr, desired_learning)
            log_debug("Learning state synced on step", desired=desired_learning)
        prev, new = self.board.step_with_delta()
        self._update_stats(prev, new)
        self.gen += 1
        self._update_status()

    def clear(self) -> None:
        self.timer.stop()
        self.board.clear()
        self.gen = 0
        self.alive_counts[:] = 0
        self.death_counts[:] = 0
        self.kill_counts[:] = 0
        self._update_status()
        self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)
        self._reset_history_and_snapshot()

    def randomize(self, density: float) -> None:
        self.timer.stop()
        self.board.randomize(density, self.allowed_species)
        self.gen = 0
        self.death_counts[:] = 0
        self.kill_counts[:] = 0
        self._recompute_alive()
        self._update_status()
        self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)
        self._reset_history_and_snapshot()

    def _update_allowed_species(self, allowed: list[int]) -> None:
        self.allowed_species = allowed or [1, 2, 3, 4, 5]
        self.status.showMessage(f"Random spawns: {', '.join(map(str, self.allowed_species))}", 1500)

    def set_wrap(self, wrap: bool) -> None:
        self.board.set_wrap(wrap)

    def _randomize_resources(self) -> None:
        self.board.randomize_resources()
        self.status.showMessage("Resources regenerated", 1200)

    def _fill_resources(self, value: float) -> None:
        self.board.fill_resources(value)
        self.status.showMessage("Resources filled", 1200)

    def _toggle_resource_view(self, enabled: bool) -> None:
        if enabled and float(self.board.resource_map.grid.max()) <= 0.0:
            self.board.randomize_resources()
            self.status.showMessage("Resources generated for view", 1500)
        log_debug("Resource view toggle requested", enabled=enabled)
        self.board.set_view_resources(False)
        if enabled:
            if self.res_window is None:
                self.res_window = ResourceWindow(self.board)
                log_debug("Resource window created")
            self.res_window.match_board_size()
            self.res_window.show()
            self.res_window.raise_()
            self.res_window.activateWindow()
            log_debug("Resource window shown")
        else:
            if self.res_window is not None:
                self.res_window.close()
                log_debug("Resource window closed")
                self.res_window = None

    def _show_policy_viz(self) -> None:
        if self.policy_mgr is None:
            self.status.showMessage("No policy manager available", 1500)
            return
        if self.policy_window is None:
            self.policy_window = PolicyVizWindow(self.policy_mgr, self)
        self.policy_window.set_learning_enabled(self.board.use_policy)
        self.policy_window.show()
        self.policy_window.raise_()
        self.policy_window.activateWindow()
        log_debug("Policy viz window shown")

    def _apply_params(self, params: dict) -> None:
        env = self.board.env_engine
        res_cfg = env.resources.config
        res_params = params.get("resource", {})
        res_cfg.regen_rate = float(res_params.get("regen_rate", res_cfg.regen_rate))
        res_cfg.max_capacity = float(res_params.get("max_capacity", res_cfg.max_capacity))
        res_cfg.baseline = float(res_params.get("baseline", res_cfg.baseline))
        res_cfg.rng_low = float(res_params.get("rng_low", res_cfg.rng_low))
        res_cfg.rng_high = float(res_params.get("rng_high", res_cfg.rng_high))
        env.resources.grid[:] = np.clip(env.resources.grid, 0.0, res_cfg.max_capacity)

        energy_params = params.get("energy", {})
        env.cfg.energy_decay = float(energy_params.get("energy_decay", env.cfg.energy_decay))
        env.cfg.energy_conversion = float(energy_params.get("energy_conversion", env.cfg.energy_conversion))
        env.cfg.stay_cost = float(energy_params.get("stay_cost", env.cfg.stay_cost))
        env.cfg.takeover_cost = float(energy_params.get("takeover_cost", env.cfg.takeover_cost))
        env.cfg.birth_action_cost = float(energy_params.get("birth_action_cost", env.cfg.birth_action_cost))

        self.status.showMessage("Parameters applied", 1500)

    def set_speed(self, ms: int) -> None:
        if self.timer.isActive():
            self.timer.start(self._current_interval_ms())

    def insert_pattern(self, name: str) -> None:
        if name == "None":
            return
        pattern = PATTERNS.get(name)
        if pattern:
            self.board.insert_pattern(pattern)
            self._recompute_alive()
            self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)

    def toggle_policy(self, enabled: bool) -> None:
        self.board.set_policy_manager(self.policy_mgr, enabled)
        if self.policy_window is not None:
            self.policy_window.set_learning_enabled(enabled)
        self.status.showMessage(f"Learning {'on' if enabled else 'off'}", 1200)
        log_debug("Learning checkbox toggled", enabled=enabled)

    def _set_brush(self, state: int | None) -> None:
        self.board.set_brush_state(state)

    def _update_status(self) -> None:
        alive = int((self.board.grid > 0).sum())
        self.status.showMessage(f"Gen {self.gen} | Alive {alive} | Wrap {'on' if self.board.wrap else 'off'}")

    def _update_stats(self, prev: np.ndarray, new: np.ndarray) -> None:
        for s in range(1, 6):
            self.alive_counts[s - 1] = int((new == s).sum())
        deaths_mask = (prev > 0) & (new != prev)
        for s in range(1, 6):
            self.death_counts[s - 1] += int(((prev == s) & deaths_mask).sum())
        kills_mask = (prev > 0) & (new > 0) & (new != prev)
        for s in range(1, 6):
            self.kill_counts[s - 1] += int(((new == s) & kills_mask).sum())
        self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)
        self._record_history(self.gen + 1)

    def _recompute_alive(self) -> None:
        for s in range(1, 6):
            self.alive_counts[s - 1] = int((self.board.grid == s).sum())


class ResourceWindow(QtWidgets.QMainWindow):
    def __init__(self, board: LifeWidget) -> None:
        super().__init__()
        self.board = board
        self.setWindowTitle("Resource Map")
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(320, 220)
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setCentralWidget(self.label)
        self.match_board_size()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(250)
        self._refresh()
        log_debug("ResourceWindow initialized")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.timer.stop()
        log_debug("ResourceWindow closeEvent")
        super().closeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        log_debug("ResourceWindow showEvent")
        self.match_board_size()
        QtCore.QTimer.singleShot(0, self._refresh)

    def match_board_size(self) -> None:
        size = self.board.size()
        if size.width() <= 0 or size.height() <= 0:
            self.resize(500, 400)
            return
        self.resize(size)

    def _refresh(self) -> None:
        grid = self.board.resource_map.grid
        if grid.size == 0:
            return
        gmax = float(grid.max())
        gmin = float(grid.min())
        gmax = max(gmax, 0.001)
        norm = np.clip((grid - gmin) / gmax, 0.0, 1.0)
        rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.uint8)
        rgba[..., 1] = (norm * 255).astype(np.uint8)
        rgba[..., 3] = 255
        h, w, _ = rgba.shape
        target_size = self.label.size()
        if target_size.width() < 10 or target_size.height() < 10:
            target_size = self.size()
        if target_size.width() < 10 or target_size.height() < 10:
            target_size = QtCore.QSize(w, h)
        image = QtGui.QImage(rgba.data, w, h, QtGui.QImage.Format.Format_RGBA8888).copy()
        pix = QtGui.QPixmap.fromImage(image).scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.FastTransformation,
        )
        self.label.setPixmap(pix)
        log_debug(
            "ResourceWindow refresh",
            shape=grid.shape,
            rmin=gmin,
            rmax=float(grid.max()),
            rmean=float(grid.mean()),
            target_w=target_size.width(),
            target_h=target_size.height(),
        )


def main() -> None:
    log_debug("App start")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Game of Life")
    app.setStyle("Fusion")
    font = app.font()
    font.setBold(True)
    app.setFont(font)
    palette = app.palette()
    text_color = QtGui.QColor(255, 255, 255)
    palette.setColor(QtGui.QPalette.ColorRole.Window, BACKGROUND)
    palette.setColor(QtGui.QPalette.ColorRole.Base, BACKGROUND)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(58, 64, 76))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, text_color)
    if hasattr(QtGui.QPalette.ColorRole, "PlaceholderText"):
        palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, text_color)
    app.setPalette(palette)
    app.setStyleSheet("* { color: #ffffff; font-weight: 600; }")

    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
