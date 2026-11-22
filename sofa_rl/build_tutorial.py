"""Generate ../SoftRobot_Ventricle_PPO.ipynb (run from the repository root: python sofa_rl/build_tutorial.py)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Steering a soft robot through the brain ventricles with PPO (SOFA + Stable-Baselines3)

A tutorial on deep reinforcement learning for a surgical soft robot. A two-section pneumatic
continuum robot enters the skull at **Kocher's point**, leaves a rigid sheath at the frontal horn
of the right lateral ventricle, and must reach a target 80 mm along a planned path without leaving
the ventricle. The robot and the ventricle wall are simulated in **SOFA**. A **PPO** policy learns
to control insertion and bending.

**You will learn**

1. how patient data become an RL **environment** (SOFA scene + Gymnasium API),
2. how the **observation** and **action** spaces are designed (residual control on a nominal schedule),
3. how the **reward function** is built, term by term, and what each term does in a real episode,
4. what the **policy** is (actor-critic MLP) and how to query a trained one without the simulator,
5. how **PPO training** is set up and how to read its learning curves,
6. how two reward designs compare (**v1**: tip only, **v2**: tip + robot-body shape).

**Runs without SOFA.** Every result below is computed from files in `sofa_rl/results/` (training logs,
trained models, recorded episodes). Cells that call the simulator are guarded by `RUN_SOFA = False`;
see `sofa_rl/ENVIRONMENT.md` to build SOFA and retrain.

| Folder | Contents |
|---|---|
| `sofa_rl/rl/` | Gymnasium environment, PPO training, evaluation, policy-driven SOFA scene |
| `sofa_rl/cosserat_robot_01_from_matlab.py` | SOFA scene: robot, sheath, wall contact, pneumatic actuation |
| `sofa_rl/data/`, `poses/`, `walls/` | Patient 44 path, ventricle mesh, Kocher fiducials, design case, collision wall |
| `sofa_rl/results/` | Training logs, models (v1, v2), recorded rollouts and episodes |
| `sofa_rl/app.py` | Gradio demo |
""")

md("## 0. Setup")
code(r"""
# pip install -r sofa_rl/requirements.txt
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch                      # import before numpy/matplotlib (avoids an OpenMP clash on some Macs)
import ast, csv, glob, json, platform, struct, zipfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = 'sofa_rl'
RES, DATA = f'{ROOT}/results', f'{ROOT}/data/patients_dataset'
RUN_SOFA = False                  # True: re-run SOFA (needs the build in sofa_rl/ENVIRONMENT.md)
torch.set_num_threads(1)

V1, V2, BASE, INK, MUTED, WALL = '#2a78d6', '#eb6834', '#8a8a86', '#0b0b0b', '#52514e', '#c9c8c3'
plt.rcParams.update({'figure.dpi': 110, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.edgecolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'lines.linewidth': 2})
import stable_baselines3, gymnasium
print('python', platform.python_version(), '| numpy', np.__version__, '| torch', torch.__version__,
      '| stable-baselines3', stable_baselines3.__version__, '| gymnasium', gymnasium.__version__)
print('reported runs: SOFA v26.06, Python 3.12.13, torch 2.7.1+cpu, SB3 2.6.0, gymnasium 1.1.1, Ubuntu 24.04')

def show(fn, name):
    # print the source of a function or method from a file without importing it (no SOFA needed)
    tree = ast.parse(open(fn).read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            print(ast.get_source_segment(open(fn).read(), node)); return
    print('not found:', name)
""")

md(r"""
## 1. The task and its data

The anatomy comes from an MRI-derived dataset of planned paths, ventricle surface meshes and 3D Slicer
fiducials (40 patients, left and right). This tutorial uses **patient 44, right ventricle** (shared with
the data owners' permission):

* **planned path** — centre path from the frontal-horn entry **F**; the goal is the path point at 80 mm
  (the robot's length);
* **ventricle mesh** — the wall: contacts, clearance, and the "left the lumen" failure test;
* **fiducials** — the coronal suture gives **Kocher's point K**; the rigid sheath is aimed from K to F.

The data are not a supervised training set. They define the **simulator**, and the agent generates its
own experience by interacting with it.
""")
code(r"""
def read_stl(fn):
    b = open(fn, 'rb').read(); n = struct.unpack('<I', b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    return a['v'].astype(float)                                    # [n, 3 vertices, xyz] mm

def read_fcsv(fn):
    return np.array([[float(v) for v in l.split(',')[1:4]] for l in open(fn) if l.strip() and not l.startswith('#')])

tris = read_stl(f'{DATA}/ventricles/patient44_right.stl')
path = np.loadtxt(f'{DATA}/paths/patient44_right.csv', delimiter=',')
suture = read_fcsv(f'{DATA}/kochers/patient44/OC_1.fcsv')
K = suture[-1]                                                     # lateral end of the suture line
ep = np.load(f'{RES}/episodes/episode_v2.npz')
goal = ep['goal']
roll = np.load(f'{RES}/rollouts/unity_v2.npz')                     # recorded rod shapes; sheath axis d, entry F
F, d = roll['F'], roll['d']
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
print(f'mesh {len(tris)} triangles | path {len(path)} points, {s[-1]:.1f} mm | goal at 80 mm')
print(f'Kocher K -> entry F: {-(K - F) @ d:.1f} mm along the sheath axis, '
      f'{np.linalg.norm((K - F) - ((K - F) @ d) * d):.2f} mm off it')

fig = plt.figure(figsize=(7, 5.5)); ax = fig.add_subplot(projection='3d')
ax.add_collection3d(Poly3DCollection(tris, alpha=0.08, facecolor='#3fb6d8', edgecolor='none'))
ax.plot(*path.T, color=INK, lw=2, label='planned path')
ax.plot(*np.array([F - 80 * d, F]).T, color='#9aa0a8', lw=5, alpha=0.6, label='rigid sheath (80 mm)')
for p, m, c, l in [(K, 'D', '#e0a800', 'Kocher point K'), (F, 's', '#1baf7a', 'entry F'), (goal, '*', '#e34948', 'goal')]:
    ax.scatter(*p, marker=m, s=60, color=c, label=l, depthshade=False)
lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
c0, r0 = (lo + hi) / 2, (hi - lo).max() / 2 + 10
ax.set(xlim=(c0[0] - r0, c0[0] + r0), ylim=(c0[1] - r0, c0[1] + r0), zlim=(c0[2] - r0, c0[2] + r0),
       xlabel='L (mm)', ylabel='P (mm)', zlabel='S (mm)'); ax.view_init(18, -125)
ax.legend(loc='upper left', fontsize=7, frameon=False); plt.show()
""")

md(r"""
## 2. The environment

**Simulator (SOFA v26.06).** The robot is a beam-FEM rod with 41 nodes, two bending sections (25 + 55 mm,
radius 3 mm). Nodes not yet inserted are held on the sheath axis behind F; inserting moves them out.
Bending comes from chamber pressures computed through a Yeoh hyperelastic actuator model. Wall contact is
solved as a linear complementarity problem (Gauss-Seidel, tolerance 1e-6). This tolerance is part of
the task: a coarser solver sends the robot into a different ventricle branch at a bifurcation.

**Gymnasium wrapper.** `VentricleNavEnv` (`sofa_rl/rl/ventricle_env.py`) exposes the usual API.
One RL step = 10 SOFA steps = 0.1 s. Each environment runs in its own process because the SOFA scene
reads its configuration at import.
""")
code(r"""
env_src = f'{ROOT}/rl/ventricle_env.py'
show(env_src, 'reset')
""")
code(r"""
if RUN_SOFA:     # needs the SOFA build; run from sofa_rl/ with rl/run.sh (macOS) or rl/run_server.sh (Linux)
    import subprocess
    subprocess.run('rl/run_server.sh rl/check_env.py', shell=True, cwd=ROOT, check=True)
""")

md(r"""
## 3. Observation space (35 values)

| Index | Content | Scale |
|---|---|---|
| 0–2 | tip − goal | ÷ 50 mm |
| 3–5 | tip − path point at the current insertion depth | ÷ 50 mm |
| 6–8 | tip direction (unit vector) | – |
| 9 | inserted depth / robot length | 0–1 |
| 10–13 | commanded curvatures (2 sections × 2 bending planes) | ÷ κ<sub>max</sub> |
| 14–31 | 6 robot nodes relative to F | ÷ 50 mm |
| 32–33 | wall clearance at the tip and minimum along the robot | ÷ 10 mm |
| 34 | total wall-contact force | ÷ 0.05 |

The scales keep every input near unit size. The policy also sees its own curvature commands (10–13),
because the action changes them incrementally.
""")
code(r"""
show(env_src, '_obs')
obs = ep['obs']
groups = {'tip - goal (x50 mm)': slice(0, 3), 'tip - path (x50 mm)': slice(3, 6), 'depth / L': slice(9, 10),
          'curvature commands': slice(10, 14), 'clearance (x10 mm)': slice(32, 34)}
fig, axs = plt.subplots(1, len(groups), figsize=(12, 2.4), constrained_layout=True)
for ax, (t, sl) in zip(axs, groups.items()):
    ax.plot(obs[:, sl], lw=1.4); ax.set_title(t, fontsize=8, loc='left'); ax.set_xlabel('RL step')
fig.suptitle('Observations along the v2 episode', fontsize=9, x=0.01, ha='left'); plt.show()
""")

md(r"""
## 4. Action space (5 values in [−1, 1])

* **a₀: insertion** — 0.5·(1 + a₀) mm per step: forward only, 0–1 mm.
* **a₁…a₄: curvature residuals** — each step adds a₁…a₄ × 0.01 1/mm to a residual on the four curvature
  components (section 1 and 2, two bending planes each).

The commanded curvature is **depth-scaled nominal schedule + accumulated residual**, clipped to
κ<sub>max</sub> = 0.12 1/mm:

$$\kappa_t = \frac{d_t}{L}\,\kappa_{\text{fit}} + \sum_{\tau \le t} 0.01\,a_{1..4,\tau}$$

This is **residual RL**. The nominal schedule κ<sub>fit</sub> (fitted to the planned path) already gives the
gross shape, so the policy only learns corrections. On its own, the schedule ends in the wrong ventricle
branch, 45 mm from the goal.
""")
code(r"""
show(env_src, 'apply_action')
episodes = {'v2 · tip + shape': (np.load(f'{RES}/episodes/episode_v2.npz'), V2),
            'v1 · tip only': (np.load(f'{RES}/episodes/episode_v1.npz'), V1),
            'nominal schedule': (np.load(f'{RES}/episodes/episode_baseline.npz'), BASE)}
fig, axs = plt.subplots(1, 3, figsize=(12, 2.8), constrained_layout=True)
for name, (e, col) in list(episodes.items())[:2]:
    axs[0].plot(0.5 * (1 + e['action'][:, 0]), color=col, lw=1.4, label=name)
    axs[1].plot(np.abs(e['action'][:, 1:]).mean(1), color=col, lw=1.4, label=name)
for name, (e, col) in episodes.items():
    axs[2].plot(np.linalg.norm(e['kappa'], axis=2).max(1) / 1e3, color=col, lw=1.4, label=name)
for a, t in zip(axs, ['insertion per step (mm)', 'mean |curvature residual action|', 'largest commanded curvature (1/mm)']):
    a.set_title(t, fontsize=8, loc='left'); a.set_xlabel('RL step')
axs[2].axhline(0.12, color=MUTED, lw=0.8, ls=':')
axs[0].legend(frameon=False, fontsize=7); axs[2].legend(frameon=False, fontsize=7); plt.show()
""")
md(r"""
Both policies always insert the full 1 mm per step (the v2 line is hidden under v1), so all the control
goes into the curvature residuals. They use them very differently: **v2 drives a section to the
curvature limit (0.12 1/mm) for most of the insertion**, bending harder to keep the body on the path,
while v1 stays near 0.07 1/mm. The nominal schedule only scales its fitted curvature with depth.
""")

md(r"""
## 5. Reward function

$$r_t = \underbrace{0.2\,\Delta d_t}_{\text{progress}}
 \;\underbrace{-\,0.002\,e^{\text{tip}}_t}_{\text{tip tracking}}
 \;\underbrace{-\,w_{\text{shape}}\,e^{\text{shape}}_t}_{\text{body on path (v2)}}
 \;\underbrace{-\,20\,f_t}_{\text{contact}}
 \;\underbrace{-\,0.01}_{\text{time}}
 \;+\;\underbrace{10\,e^{-e^{\text{goal}}/10}}_{\text{at full insertion}}
 \;\underbrace{-\,10}_{\text{left the lumen}}$$

* $\Delta d_t$ is the inserted millimetres. Completing the insertion is always worth more than stalling.
* $e^{\text{tip}}_t$ is the tip distance to the path point at the same depth (a moving reference).
* $e^{\text{shape}}_t$ is the mean distance of all inserted nodes to the path at equal arc length.
  **v1: $w_{\text{shape}} = 0$; v2: $w_{\text{shape}} = 0.01$ 1/mm.**
* $f_t$ is the total wall-contact force. The terminal bonus decays smoothly with goal error.
  Success means goal error ≤ 3 mm; it is a metric, not a reward step.
""")
code(r"""
show(env_src, 'step')
""")
code(r"""
terms = ['r_depth', 'r_tip', 'r_shape', 'r_force', 'r_time', 'r_terminal']
labels = ['progress', 'tip tracking', 'body shape', 'contact', 'time', 'terminal']
print(f'{"episode":<18}' + ''.join(f'{l:>13}' for l in labels) + f'{"return":>10}{"goal err":>10}')
for name, (e, _) in episodes.items():
    print(f'{name:<18}' + ''.join(f'{e[t].sum():13.2f}' for t in terms) + f'{e["reward"].sum():10.2f}{e["goal_err_mm"][-1]:9.2f} mm')
print('\n(each episode is scored with the reward it was trained with: w_shape = 0.01 for v2, 0 otherwise)')

fig, axs = plt.subplots(1, 3, figsize=(12, 2.8), constrained_layout=True)
for name, (e, col) in episodes.items():
    axs[0].plot(e['tip_err_mm'], color=col, lw=1.4, label=name)
    axs[1].plot(e['shape_err_mm'], color=col, lw=1.4, label=name)
    axs[2].plot(np.cumsum(e['reward']), color=col, lw=1.4, label=name)
for a, t in zip(axs, ['tip error vs path at current depth (mm)', 'body off path, e_shape (mm)', 'cumulative reward']):
    a.set_title(t, fontsize=8, loc='left'); a.set_xlabel('RL step')
axs[2].legend(frameon=False, fontsize=7); plt.show()
""")
md(r"""
The nominal schedule collects the same progress reward as the policies but almost no terminal bonus,
because it ends 45 mm from the goal. It also pays more than ten times the contact penalty, because it
presses into the wall of the wrong branch. v1 and v2 track the tip about equally well; only v2 keeps the
**body** near the path, because only v2 pays for it.
""")

md(r"""
## 6. The policy

PPO uses an **actor-critic**. The **actor** maps the 35-value observation to the mean of a Gaussian over
the 5 actions (state-independent log-std). The **critic** estimates the state value V(s) used for
advantages. Both are 2 × 128 tanh MLPs. Below, the trained v2 model is loaded from its zip file and
queried on the recorded observations. It reproduces the recorded actions to float precision, so you
can analyse the policy without SOFA. (v1's return, 23.5, is higher than v2's, 22.2, only because v2 is also
charged for its body shape: returns from different reward functions are not comparable.)
""")
code(r"""
from stable_baselines3 import PPO
models = {'v2': PPO.load(f'{RES}/ppo_p44_server_v2_shape/final_model.zip', device='cpu'),
          'v1': PPO.load(f'{RES}/ppo_p44_server_v1/model_190k.zip', device='cpu')}
m = models['v2']
print(m.policy)
print('parameters:', sum(p.numel() for p in m.policy.parameters()),
      '| action std:', np.round(m.policy.log_std.exp().detach().numpy(), 3))

e = episodes['v2 · tip + shape'][0]
pred, _ = m.predict(e['obs'][:-1], deterministic=True)
print(f'max |predicted - recorded action| over {len(pred)} steps: {np.abs(pred - e["action"]).max():.2e}')

with torch.no_grad():
    values = m.policy.predict_values(torch.as_tensor(e['obs'], dtype=torch.float32)).numpy().ravel()
plt.figure(figsize=(5, 2.6)); plt.plot(values, color=V2, lw=1.6)
plt.title('critic V(s) along the v2 episode (normalised-reward units)', fontsize=8, loc='left'); plt.xlabel('RL step'); plt.show()
""")

md(r"""
## 7. Training with PPO

PPO collects rollouts with the current policy, then takes several gradient steps on the **clipped
surrogate objective**

$$L(\theta) = \mathbb{E}_t\big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\big],
\quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

plus a value-function loss. The clip keeps each update close to the policy that collected the data.
Advantages $A_t$ come from generalised advantage estimation (GAE). Rewards are normalised online
(VecNormalize); observations are not normalised, since they are already scaled.

**Training system.** 5 SOFA simulators run in parallel processes (`SubprocVecEnv`), about 60 steps/s on 6
CPU cores, so 200k steps take about 1 h. Each episode reset rebuilds the SOFA scene, and the worker
processes leak memory: the first run was killed by the out-of-memory killer at 198.4k steps. Training
now runs one rollout per `learn()` call and respawns the workers when one exceeds 1.5 GB, keeping the
policy, optimiser and normaliser, so no steps are lost. `rl/train_supervised.sh` resumes from the newest
checkpoint after a crash.
""")
code(r"""
hp = {k: getattr(m, k) for k in ['n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm']}
# clip_range / lr_schedule are pickled functions in the zip: calling them under a different Python version than
# the one that saved the model (3.12) can crash the interpreter, so the clip value is read from train_ppo.py
clip = [l for l in open(f'{ROOT}/rl/train_ppo.py') if 'clip_range=' in l][0].split('clip_range=')[1].split(',')[0]
hp.update(clip_range=float(clip), n_envs=5, rollout_size=5 * m.n_steps, net_arch='[128, 128] actor and critic, tanh')
for k, v in hp.items():
    print(f'{k:<15} {v}')
cmds = {'v1': 'rl/run_server.sh rl/train_ppo.py --name ppo_p44_server_v1 --steps 200000 --envs 5 --n-steps 256 --ckpt 10000 --max-episode-steps 250',
        'v2': 'rl/train_supervised.sh ppo_p44_server_v2_shape 200000 --envs 5 --n-steps 256 --ckpt 10000 --max-episode-steps 250 --w-shape 0.01'}
if RUN_SOFA:
    import subprocess
    for c in cmds.values():
        subprocess.run(c, shell=True, cwd=ROOT, check=True)          # ~1 h each on 6 cores
""")
code(r"""
def load_run(run):
    files = sorted(glob.glob(f'{RES}/{run}/monitor_*.monitor.csv'))
    eps = []
    for fn in files:
        with open(fn) as f:
            next(f); rows = list(csv.DictReader(f))
        n = 0
        for r in rows:
            n += int(r['l'])
            eps.append(dict(step=n * len(files), success=r['success'] == 'True', goal=float(r['goal_err_mm']),
                            shape=float(r['shape_err_mm'])))
    return eps

def binned(eps, key, agg, width=10_000):
    edges = np.arange(0, max(e['step'] for e in eps) + width, width)
    out = [(b / 1000, agg([e[key] for e in eps if a < e['step'] <= b])) for a, b in zip(edges[:-1], edges[1:])
           if sum(a < e['step'] <= b for e in eps) >= 10]
    return np.array(out).T

runs = {'v1 · tip reward': ('ppo_p44_server_v1', V1), 'v2 · tip + shape reward': ('ppo_p44_server_v2_shape', V2)}
logs = {k: load_run(r) for k, (r, _) in runs.items()}
fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
for ax, (key, title, agg) in zip(axs, [('success', 'success rate (%)', lambda v: 100 * np.mean(v)),
                                       ('goal', 'goal error at episode end (mm)', np.mean),
                                       ('shape', 'body off path (mm)', np.mean)]):
    for k, (r, col) in runs.items():
        x, y = binned(logs[k], key, agg); ax.plot(x, y, color=col, label=k)
    ax.set_title(title, fontsize=8, loc='left'); ax.set_xlabel('training steps (thousands)')
    if key != 'success':
        ax.set_yscale('log'); t = [1, 2, 3, 5, 10, 20, 40]; ax.set_yticks(t); ax.set_yticklabels(t); ax.minorticks_off()
axs[0].legend(frameon=False, fontsize=7); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 2.6), constrained_layout=True)
for k, (r, col) in runs.items():
    rows = [x for f in sorted(glob.glob(f'{RES}/{r}/progress*.csv'), key=lambda f: ('upto' not in f, f)) for x in csv.DictReader(open(f))]
    st = np.array([float(x['time/total_timesteps']) for x in rows]) / 1000
    for ax, key in zip(axs, ['train/std', 'train/explained_variance']):
        ax.plot(st, [float(x[key]) if x.get(key) else np.nan for x in rows], color=col, lw=1.3, label=k)
axs[0].set_title('policy action std (exploration)', fontsize=8, loc='left'); axs[1].set_title('critic explained variance', fontsize=8, loc='left')
axs[1].set_ylim(0, 1.02)
for a in axs: a.set_xlabel('training steps (thousands)')
plt.show()
""")
md(r"""
**Reading the curves.** Both runs escape the wrong branch within about 40k steps: goal error falls from
about 45 mm to under 10 mm. v1's body stays about 4.7 mm off the path for the whole run, because nothing
in its reward refers to the body. v2 halves that and also reaches the goal more accurately: a body on
the path keeps the tip where the goal is reachable. Exploration noise (action std) shrinks smoothly, and
the critic explains about 97 % of the return variance. Both are signs of healthy PPO training.
""")

md("## 8. Results")
code(r"""
def summary(fn):
    z = np.load(fn); n = int(z['n'])
    g = np.array([float(z[f'goal_err_{i}']) for i in range(n)])
    dev = np.array([np.min(np.linalg.norm(z[f'rod_{i}'][:, None] - z['path'][None], axis=2), 1).mean() for i in range(n)])
    return g, dev
print(f'{"policy":<22}{"deterministic goal err":>24}{"test success (n=30)":>22}{"test goal err median [p90]":>30}{"body off path":>16}')
for name, det, ev in [('v1 · tip reward', 'v1_190k_det', 'eval30_v1'), ('v2 · tip + shape', 'v2_final_det', 'eval30_v2')]:
    gd, _ = summary(f'{RES}/rollouts/{det}.npz'); g, dev = summary(f'{RES}/rollouts/{ev}.npz')
    print(f'{name:<22}{gd[0]:21.2f} mm{100 * np.mean(g <= 3):20.0f} %{np.median(g):19.2f} [{np.percentile(g, 90):.2f}] mm{dev.mean():13.2f} mm')
print(f'{"nominal schedule":<22}{float(np.load(f"{RES}/rollouts/unity_baseline.npz")["goal_err_0"]):21.2f} mm{0:20.0f} %')
print('\ntest = 30 new episodes per policy with sampled actions (seed 7), same anatomy; success = goal error <= 3 mm')
""")
code(r"""
c = path.mean(0); _, _, Vt = np.linalg.svd(path - c); P = lambda x: (x - c) @ Vt.T
wall_pts = tris.reshape(-1, 3)[::3]
near = wall_pts[np.min(np.linalg.norm(wall_pts[:, None] - path[None, ::3], axis=2), 1) < 25]
fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, gridspec_kw={'width_ratios': [1.25, 1]})
for ax, (i, j, title) in zip(axs, [(0, 1, 'main plane of the path'), (0, 2, 'side view (out of plane)')]):
    q = P(near); ax.scatter(q[:, i], q[:, j], s=1.2, color=WALL, lw=0)
    for f, col, lab in [('unity_baseline', BASE, 'nominal schedule'), ('unity_v1', V1, 'v1 · tip'), ('unity_v2', V2, 'v2 · tip + shape')]:
        z = np.load(f'{RES}/rollouts/{f}.npz'); tip = P(z['rods_0'][:, -1]); rod = P(z['rods_0'][-1])
        ax.plot(tip[:, i], tip[:, j], color=col, lw=1.2, ls='--' if 'baseline' in f else '-', label=lab)
        if 'baseline' not in f:
            ax.plot(rod[:, i], rod[:, j], color=col, lw=6, alpha=0.25)
    q = P(path); ax.plot(q[:, i], q[:, j], color=INK, lw=1.4, label='planned path')
    ax.set_aspect('equal'); ax.set_title(title, fontsize=8, loc='left'); ax.set_xlabel('mm')
axs[0].legend(frameon=False, fontsize=7, loc='lower left', ncol=2); plt.show()
print('thin: tip path per RL step; wide bands: final robot body (deterministic policies)')
""")
md(r"""
| | Nominal schedule | PPO v1 (tip) | PPO v2 (tip + shape) |
|---|---|---|---|
| Deterministic goal error | 45.4 mm (wrong branch) | 1.2 mm | **0.7 mm** |
| Success, 30 test episodes | 0 % | 90 % | **97 %** |
| Body off path | 8.3 mm | 4.6 mm | **2.5 mm** |

Results repeat exactly on one platform, not across platforms: the deterministic v2 rollout ends 0.67 mm
from the goal on Linux x86-64 (where these numbers were produced) and 0.74 mm on macOS arm64, because
the contact solver's floating-point results differ slightly.
""")

md(r"""
## 9. Interactive demo (Gradio)

`sofa_rl/app.py` has three tabs: **Replay** (3D replay of the recorded insertions with a step slider),
**Results**, and **Live SOFA** (runs a new simulation with a chosen policy, action mode and seed; shown only
where SOFA is installed).
""")
code(r"""
LAUNCH_APP = False
if LAUNCH_APP:
    import sys; sys.path.insert(0, ROOT)
    from app import demo
    demo.launch(inline=True, height=900)          # or: python sofa_rl/app.py   (add --share for a public link)
""")

md(r"""
## 10. Takeaways and limitations

* **Residual RL** on a physically fitted schedule makes the problem learnable in about 10⁵ simulator steps.
* **Reward design matters most.** A body-to-path term halved the body deviation and improved the task metric.
* **Simulator settings are part of the task** (contact-solver tolerance), and long simulation-in-the-loop
  training needs **resource management** (worker respawning).

Limitations: one anatomy for training and testing, rigid wall, no sensor noise or domain randomisation,
30 test episodes per policy. Next steps: train on many patients and test on held-out ones; randomise wall
stiffness, friction, actuator gain and registration; add a history-dependent (poroviscoelastic) wall;
compare against non-learned baselines.
""")

nb = nbf.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'SoftRobot_Ventricle_PPO.ipynb')
print('wrote SoftRobot_Ventricle_PPO.ipynb with', len(C), 'cells')
