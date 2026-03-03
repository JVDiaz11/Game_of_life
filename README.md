# Game of Life (PySide6)

A modern Life variant with 5 species, barriers, and a PySide6 GUI: start/pause, step, randomize, clear, wrap toggle, speed slider, density slider, and pattern insertion (Glider, Small Exploder, Pulsar, Gosper Gun). Left brush panel selects what you paint (cycle/erase/barrier/species 1-5). Right stats panel shows per-species alive/deaths/kills. Click/drag to paint. Species interact with cyclic dominance; colors fade with age. Optional Learning toggle (per-species small MLP) lives in agent_policy.py.

## Run from source
```bash
cd Toys/Game_of_life
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python game_of_life.py
```

### Learning (per-species PPO)

- Each species owns a single PPO policy/value net; weights save under `species/` as `species{n}.pt`.
- Policies mutate over time: when a species policy ages it can widen hidden layers and apply small noise to encourage exploration.
- Action space: stay, die, move (up/down/left/right), or claim an empty cell.

## Build a single-file app (PyInstaller)
Build on the target OS (Windows for .exe, Linux for ELF):
```bash
cd Toys/Game_of_life
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed game_of_life.py --name game-of-life
```
The binary will be in `dist/` (e.g., `dist/game-of-life.exe` on Windows). Double-click to run.

## Controls
- Start / Pause: run or halt simulation
- Step: advance one generation
- Random: fill with noise (density slider sets % alive)
- Clear: empty the board
- Wrap: toroidal edges on/off
- Speed: update interval (ms)
- Patterns: drop Glider, Small Exploder, Pulsar, or Gosper Gun at center
- Mouse: click/drag to cycle barrier/empty/species
- Learning: toggle in UI to let per-species policies pick next state (random weights by default; load/train via species_policy)
- Resources: generate a resource map (random or fill), toggle View Resources to see the heatmap; resources regen each step and gate births/upkeep/energy.
- Policy Viz: click the Policy Viz button to open a live window with per-species policy stats (action distribution, activation stats, grad norms, hidden size).
- Config Panel: adjust resource regen/baseline/capacity/random bounds and energy costs/decay at runtime; open via the panel toggles.
- Brush panel (left): choose what the cursor paints (cycle, erase, barrier, species 1-5)
- Stats panel (right): shows per-species alive, deaths, kills (updates each step)
- Window toolbar (top-right) shows Minimize / Maximize / Close buttons; native title bar controls also work.

## Notes
- Default grid: 80x110; colors fade with age per species.
- Wrapping uses toroidal neighbors; unwrapped mode treats edges as dead.
- For larger grids, consider increasing the window size or reducing rows/cols in `LifeWidget`.
- Barriers stay fixed; policies cannot change them.
- Simulation internals now live in `modules/`:
	- `modules/environment.py` applies ecological rules, predation, resource gates, and hooks into policies.
	- `modules/resources.py` tracks cell food with regen/baseline values and per-species costs.
	- `modules/resource_map.py` owns resource map generation/management used by the environment.
	- `modules/memory.py` holds per-cell memory channels (last reward, time since food).
	- `modules/species_policy.py` houses per-species PPO policy/value nets; `agent_policy.py` is a thin compatibility shim.
	- `modules/ppo_buffer.py` and `modules/ppo_trainer.py` implement online PPO rollouts/updates.
