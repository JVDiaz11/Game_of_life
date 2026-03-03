from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .species_policy import SpeciesPolicyManager, NUM_ACTIONS


class PolicyVizWindow(QtWidgets.QMainWindow):
    """Live view of per-species policy stats (architecture, activations, gradients)."""

    def __init__(self, policy_mgr: SpeciesPolicyManager, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Policy Visualizer")
        self.policy_mgr = policy_mgr
        self.grad_history: Dict[int, list[float]] = {}
        self.learning_on = False
        self.setMinimumWidth(520)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Species:"))
        self.species_box = QtWidgets.QComboBox()
        self.species_box.addItems(["1", "2", "3", "4", "5"])
        controls.addWidget(self.species_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.info_label = QtWidgets.QLabel("No data yet")
        layout.addWidget(self.info_label)

        self.fig = Figure(figsize=(6, 4), facecolor="#111622")
        self.canvas = FigureCanvas(self.fig)
        self.ax_actions = self.fig.add_subplot(2, 1, 1)
        self.ax_grad = self.fig.add_subplot(2, 1, 2)
        layout.addWidget(self.canvas)

        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._refresh)
        self.timer.start()

    def set_learning_enabled(self, enabled: bool) -> None:
        self.learning_on = enabled

    def _refresh(self) -> None:
        sid = int(self.species_box.currentText())
        snap = self.policy_mgr.debug_snapshot()
        data = snap.get(sid)
        if not data:
            self.info_label.setText(
                "No stats yet — ensure Learning is on and simulation is running"
            )
            self._clear_plots()
            return

        hidden = data.get("hidden", 0)
        params = data.get("param_count", 0)
        logits_mean = data.get("logits_mean", 0.0)
        logits_std = data.get("logits_std", 0.0)
        value_mean = data.get("value_mean", 0.0)
        value_std = data.get("value_std", 0.0)
        buffer_fill = data.get("buffer_fill", 0.0)
        last_update = data.get("last_update", {})
        grad_norm = last_update.get("grad_norm", 0.0)

        learning_txt = "on" if self.learning_on else "off"
        self.info_label.setText(
            f"Learning {learning_txt} | Hidden {hidden} | Params {params} | logits μ {logits_mean:.3f} σ {logits_std:.3f} | "
            f"value μ {value_mean:.3f} σ {value_std:.3f} | buffer {buffer_fill*100:.0f}% | grad {grad_norm:.3f}"
        )

        # action distribution
        act_dist = np.array(data.get("action_dist", np.zeros(NUM_ACTIONS)))
        self.ax_actions.clear()
        self.ax_actions.bar(np.arange(NUM_ACTIONS), act_dist, color="#4cb0ff")
        self.ax_actions.set_ylim(0, max(0.01, act_dist.max() * 1.1))
        self.ax_actions.set_title("Action distribution")

        # gradient norm history
        hist = self.grad_history.setdefault(sid, [])
        if grad_norm:
            hist.append(grad_norm)
            if len(hist) > 200:
                del hist[:-200]
        self.ax_grad.clear()
        if hist:
            self.ax_grad.plot(hist, color="#ff9f43")
        self.ax_grad.set_title("Grad norm (last updates)")
        self.ax_grad.set_ylim(bottom=0)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _clear_plots(self) -> None:
        self.ax_actions.clear()
        self.ax_grad.clear()
        self.canvas.draw_idle()
