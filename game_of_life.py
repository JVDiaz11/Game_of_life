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

import random
import sys
from typing import Dict, List, Sequence, Tuple

import torch

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from agent_policy import PolicyManager

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
        # grid values: 0 empty, 1-5 species, -1 barrier
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.age = np.zeros((rows, cols), dtype=np.int32)
        self.wrap = True
        self._mouse_down = False
        self.setMouseTracking(True)
        self.policy_mgr: PolicyManager | None = None
        self.use_policy = False
        self.brush_state: int | None = None  # None means cycle

    def set_wrap(self, wrap: bool) -> None:
        self.wrap = wrap

    def set_policy_manager(self, mgr: PolicyManager | None, enabled: bool) -> None:
        self.policy_mgr = mgr
        self.use_policy = enabled

    def clear(self) -> None:
        self.grid[:] = 0
        self.age[:] = 0
        self.update()

    def randomize(self, density: float = 0.2) -> None:
        alive_mask = np.random.rand(self.rows, self.cols) < density
        species_choices = np.random.randint(1, 6, size=(self.rows, self.cols), dtype=np.int8)
        self.grid = np.where(alive_mask, species_choices, 0)
        self.age[:] = 0
        self.update()

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
        if self.use_policy and self.policy_mgr is not None:
            new_grid = self._policy_step(g_prev)
        else:
            new_grid = self._rule_step(g_prev)

        same_species = (new_grid == g_prev) & (new_grid > 0)
        births_any = (new_grid > 0) & (~same_species)
        self.age = np.where(new_grid > 0, np.where(births_any, 1, self.age + 1), 0)
        self.grid = new_grid
        self.update()
        return g_prev, new_grid

    def _rule_step(self, g: np.ndarray) -> np.ndarray:
        neighbor_counts = []
        for s in range(1, 6):
            mask = g == s
            if self.wrap:
                n = sum(
                    np.roll(np.roll(mask, i, axis=0), j, axis=1)
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                    if not (i == 0 and j == 0)
                )
            else:
                n = np.zeros_like(mask, dtype=np.int8)
                n[1:-1, 1:-1] = (
                    mask[:-2, :-2]
                    + mask[:-2, 1:-1]
                    + mask[:-2, 2:]
                    + mask[1:-1, :-2]
                    + mask[1:-1, 2:]
                    + mask[2:, :-2]
                    + mask[2:, 1:-1]
                    + mask[2:, 2:]
                )
            neighbor_counts.append(n.astype(np.int16))
        counts_stack = np.stack(neighbor_counts, axis=-1)

        new_grid = np.zeros_like(g)
        barrier_mask = g == -1
        new_grid[barrier_mask] = -1

        pred_map = np.array([0, 5, 1, 2, 3, 4], dtype=np.int8)
        species_mask = g > 0
        predators = pred_map[g.clip(0, 5)]
        flat_counts = counts_stack.reshape(-1, 5)
        pred_idx = np.clip(predators.reshape(-1) - 1, 0, 4)
        pred_counts = flat_counts[np.arange(flat_counts.shape[0]), pred_idx].reshape(g.shape)
        predation_mask = species_mask & (pred_counts >= 3)

        for s in range(1, 6):
            s_mask = g == s
            same = counts_stack[..., s - 1]
            survive = s_mask & (~predation_mask) & ((same == 2) | (same == 3))
            new_grid[survive] = s

        new_grid[predation_mask] = predators[predation_mask]

        empty_mask = g == 0
        max_count = counts_stack.max(axis=-1)
        argmax = counts_stack.argmax(axis=-1) + 1
        tie_mask = (counts_stack == max_count[..., None]).sum(axis=-1) > 1
        birth_mask = empty_mask & (max_count >= 3) & (~tie_mask)
        new_grid[birth_mask] = argmax[birth_mask]
        return new_grid

    def _policy_step(self, g: np.ndarray) -> np.ndarray:
        # barriers stay barriers
        barriers = g == -1
        grid_t = torch.from_numpy(g.astype(np.int64))
        actions = self.policy_mgr.infer_next_states(grid_t)
        new_grid = actions.numpy().astype(np.int8)
        new_grid[barriers] = -1
        return new_grid

    # --- painting ---
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        # Grid background stays dark
        painter.fillRect(self.rect(), QtGui.QColor(16, 20, 26))
        w = self.width()
        h = self.height()
        cw = w / self.cols
        ch = h / self.rows

        # Grid lines (lightweight)
        painter.setPen(QtGui.QPen(GRID_COLOR, 1))
        for c in range(self.cols + 1):
            x = int(c * cw)
            painter.drawLine(x, 0, x, h)
        for r in range(self.rows + 1):
            y = int(r * ch)
            painter.drawLine(0, y, w, y)

        # Cells and barriers
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                if val == 0:
                    continue
                rect = QtCore.QRect(int(c * cw) + 1, int(r * ch) + 1, int(cw) - 1, int(ch) - 1)
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

        for btn in [self.start_btn, self.pause_btn, self.step_btn, self.clear_btn, self.random_btn]:
            layout.addWidget(btn)

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

        self.policy_box = QtWidgets.QCheckBox("NN policy")
        layout.addWidget(self.policy_box)

        layout.addStretch(1)

        self.start_btn.clicked.connect(self.start_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.step_btn.clicked.connect(self.step_clicked)
        self.clear_btn.clicked.connect(self.clear_clicked)
        self.random_btn.clicked.connect(self._emit_random)
        self.wrap_box.stateChanged.connect(lambda s: self.wrap_changed.emit(s == QtCore.Qt.CheckState.Checked))
        self.speed_slider.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        self.pattern_box.currentTextChanged.connect(self.pattern_selected)
        self.policy_box.stateChanged.connect(lambda s: self.policy_toggle.emit(s == QtCore.Qt.CheckState.Checked))

    def _emit_random(self) -> None:
        density = self.density_slider.value() / 100.0
        self.random_clicked.emit(density)


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
        self.stats_panel = StatsPanel()
        self.insights_panel = InsightsPanel()
        self.brush_panel.setMinimumWidth(180)
        self.stats_panel.setMinimumWidth(220)
        self.insights_panel.setMinimumWidth(260)

        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(container)
        vlayout.setContentsMargins(4, 4, 4, 4)
        vlayout.setSpacing(6)
        vlayout.addWidget(self.controls)

        self.panel_toggle_bar = PanelToggleBar(
            [
                ("brush", "Brush Panel", True),
                ("stats", "Stats Panel", True),
                ("insights", "Insights Panel", False),
            ]
        )
        vlayout.addWidget(self.panel_toggle_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

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

        splitter.addWidget(self.left_scroll)
        splitter.addWidget(self.board)
        splitter.addWidget(self.right_scroll)
        splitter.addWidget(self.insights_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)

        vlayout.addWidget(splitter, 1)

        self.setCentralWidget(container)

        self.panel_widgets = {
            "brush": self.left_scroll,
            "stats": self.right_scroll,
            "insights": self.insights_scroll,
        }
        self.panel_toggle_bar.panel_toggled.connect(self._handle_panel_toggle)
        self.insights_panel.metric_changed.connect(self._update_insights_plot)
        self.insights_scroll.hide()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)

        # policy manager (untrained by default). You can load trained weights here.
        self.policy_mgr = PolicyManager.create([1, 2, 3, 4, 5])
        self.board.set_policy_manager(self.policy_mgr, enabled=False)

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
        self.controls.wrap_changed.connect(self.set_wrap)
        self.controls.speed_changed.connect(self.set_speed)
        self.controls.pattern_selected.connect(self.insert_pattern)
        self.controls.policy_toggle.connect(self.toggle_policy)
        self.brush_panel.brush_changed.connect(self._set_brush)

    def _tick(self) -> None:
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
        self.board.randomize(density)
        self.gen = 0
        self.death_counts[:] = 0
        self.kill_counts[:] = 0
        self._recompute_alive()
        self._update_status()
        self.stats_panel.update_counts(self.alive_counts, self.death_counts, self.kill_counts)
        self._reset_history_and_snapshot()

    def set_wrap(self, wrap: bool) -> None:
        self.board.set_wrap(wrap)

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


def main() -> None:
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
