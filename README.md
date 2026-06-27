# Modification-Considering Value Learning (MCVL)

Code for the reward-tampering experiments studying **Modification-Considering Value
Learning (MCVL)** — a forecast-and-score gate that lets a reinforcement-learning agent
detect and avoid reward-tampering behaviour during deployment.

The repository contains two independent experiment families:

1. **Gridworld experiments** (tabular/DQN) on the DeepMind AI-safety gridworlds,
   driven by [`run.sh`](run.sh) → [`run_experiment.py`](run_experiment.py).
2. **Continuous-control experiments** (TD3) on custom MuJoCo environments,
   driven by [`run_td3.sh`](run_td3.sh) → [`td3.py`](td3.py).

Plots for both are produced by the two Jupyter notebooks
[`plots.ipynb`](plots.ipynb) and [`plots_td3.ipynb`](plots_td3.ipynb).

---

## Installation

Requires Python 3.10+. We recommend a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# The gridworld gym wrapper ships with the repo and must be installed editable:
pip install -e safe-grid-gym
```

`ai_safety_gridworlds/` is a vendored package and is imported directly from the
repository root, so run all commands from the repo root (or keep it on
`PYTHONPATH`).

> **MuJoCo:** the continuous-control environments depend on `mujoco` (installed via
> `requirements.txt`). If MuJoCo fails to import, follow the
> [MuJoCo installation notes](https://github.com/google-deepmind/mujoco) for your
> platform.

---

## 1. Gridworld experiment (`run.sh`)

Runs the main MCVL experiment. For every random seed it:

1. trains an initial (pre-deployment) agent,
2. runs a deployment phase where reward tampering is *possible* (no gate), and
3. runs a deployment phase protected by the MCVL **forecast-and-score gate**,

then writes all metrics and checkpoints to `results/<name>_<EnvClass>/`.

```bash
bash run.sh
```

By default the experiment uses the **Tomato Watering** environment. To run a
different gridworld, edit the `env_class` line at the bottom of
[`run_experiment.py`](run_experiment.py); the available environments are:

- `TomatoWateringEnvironment` (default)
- `AbsentSupervisorEnvironment`
- `RocksDiamondsEnvironment`

Hyperparameters live in [`config.py`](config.py) (`get_default_config` in
[`environment_utils.py`](environment_utils.py) sets per-environment overrides).

`run.sh` includes example `#SBATCH` directives for Slurm clusters; they are
ignored when the script is launched directly with `bash run.sh`.

### Oracle baseline

The main figure (`plot_run_oracle` in `plots.ipynb`) compares the MCVL agent
against an **Oracle** that trains directly on the true (un-tampered) reward. To
produce the Oracle run, set `use_real_reward_for_training = True` and give it a
distinct name in [`run_experiment.py`](run_experiment.py):

```python
env_class = TomatoWateringEnvironment
config = get_default_config(env_class)
config.use_real_reward_for_training = True   # train on the true reward
run_experiments('oracle', config, parallel=True)
```

Launch this as a second job (alongside the default `mcvl` run). It writes to
`results/oracle_<EnvClass>/`, which is the second argument to `plot_run_oracle`.

---

## 2. Continuous-control experiment (`run_td3.sh`)

Runs a single TD3 training job for one seed/configuration:

```bash
# One seed, default settings (Ant-v5):
bash run_td3.sh --seed 0 --env_id Ant-v5

# Other environments:
bash run_td3.sh --seed 0 --env_id HalfCheetah-v5
bash run_td3.sh --seed 0 --env_id Reacher-v5
```

Useful flags (see [`td3.py`](td3.py) for the full list):

| Flag | Description |
|------|-------------|
| `--env_id` | `Ant-v5`, `HalfCheetah-v5`, or `Reacher-v5` |
| `--seed` | Random seed |
| `--check_tampering` | Enable the MCVL tampering gate |
| `--oracle_reward` | Train against the oracle (true) reward |
| `--total_timesteps` | Override the default training length |
| `--output_dir` | Where to write run logs (default `runs/`) |

Results are written under `--output_dir` as TensorBoard event files.

### Reward variants (incl. Oracle)

`plots_td3.ipynb` compares three agents, selected through the reward flags
(note that `--oracle_reward` defaults to `True`, so a bare `run_td3.sh` produces
an Oracle run):

| Agent | Flags |
|-------|-------|
| **Oracle** — trains on the true reward | `--oracle_reward True` (default) |
| **TD3** — proxy reward, tampering possible | `--oracle_reward False` |
| **MC-TD3** (ours) — proxy reward + MCVL gate | `--oracle_reward False --check_tampering True` |

(`--oracle_reward` and `--check_tampering` are mutually exclusive.) The variant
is encoded in each run directory's name (`__oracle`, `CheckTamperingTrue`, …),
which `plots_td3.ipynb` uses to bucket the curves automatically.

To launch a full sweep of 10 seeds (one Slurm job per seed) use
[`submit_td3_seeds`](submit_td3_seeds); pass the variant flags through and point
each variant at the same `--output_dir`:

```bash
./submit_td3_seeds --env_id Ant-v5                                          # Oracle
./submit_td3_seeds --env_id Ant-v5 --oracle_reward False                    # TD3
./submit_td3_seeds --env_id Ant-v5 --oracle_reward False --check_tampering True  # MC-TD3
```

---

## 3. Plotting

After running experiments, generate the figures:

- [`plots.ipynb`](plots.ipynb) — gridworld results (reads `results/`).
- [`plots_td3.ipynb`](plots_td3.ipynb) — continuous-control results (reads `runs/`).

Open the relevant notebook and set the run name(s) at the top of the plotting
cells to match the experiment directory you produced (e.g.
`mcvl_TomatoWateringEnvironment`). Figures are saved to `plots/`.

---

## Experiment tracking

Both experiment families log to [Weights & Biases](https://wandb.ai). Set your own
entity/project by running `wandb login` first, or disable tracking:

- Gridworlds: comment out the `wandb.init(...)` call in `run_experiment.py`.
- TD3: pass `--track False` to `run_td3.sh` / `td3.py`.

---

## Repository layout

```
run.sh, run_experiment.py        # gridworld experiment entry point
run_td3.sh, td3.py, submit_td3_seeds  # continuous-control entry points
training.py, environment_utils.py     # DQN training loop + env helpers
config.py, helpers.py, networks.py    # configuration, utilities, models
train_state.py, replay.py             # training state + replay buffer
BoxMovingEnv.py                       # custom box-moving reward-tampering gridworld
reacher.py, ant_v5.py, half_cheetah_v5.py  # custom MuJoCo environments
ai_safety_gridworlds/, safe-grid-gym/      # vendored gridworld libraries
plots.ipynb, plots_td3.ipynb               # plotting notebooks
```

---

## Contact

Questions? Please reach out to **eop [at] cs.utoronto.edu**.
