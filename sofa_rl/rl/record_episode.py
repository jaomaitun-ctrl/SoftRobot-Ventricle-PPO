"""
Record one episode step by step for analysis/teaching: observations, actions, every reward
term, and the state measurements behind them.

    rl/run_server.sh rl/record_episode.py --ckpt <model.zip> [--w-shape 0.01] [--stochastic --seed 3] --out ep.npz
    rl/run_server.sh rl/record_episode.py --baseline --out ep_baseline.npz

Saved arrays (T = number of RL steps):
  obs [T+1, 35]   observation before each action (and the final one)
  action [T, 5]   policy action in [-1, 1] (insertion, 4 curvature residuals)
  reward [T]      env reward; terms: r_depth, r_tip, r_shape, r_force, r_time, r_terminal (sum = reward)
  depth_mm, tip_err_mm, goal_err_mm, shape_err_mm, force, clear_min_mm [T]   after each step
  kappa [T, 2, 2] commanded curvatures (1/m), pressures_kpa [T, 2, n_chambers]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--w-shape', type=float, default=0.0, help='reward weight used to score the episode')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import ventricle_env as ve
    env = ve.VentricleNavEnv(max_steps=250, w_shape=a.w_shape)
    model = None
    if not a.baseline:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        model = PPO.load(a.ckpt, device='cpu')
        if a.stochastic:
            model.set_random_seed(a.seed)

    obs, info = env.reset(seed=a.seed)
    rec = {k: [] for k in ['obs', 'action', 'reward', 'r_depth', 'r_tip', 'r_shape', 'r_force', 'r_time',
                           'r_terminal', 'depth_mm', 'tip_err_mm', 'goal_err_mm', 'shape_err_mm', 'force',
                           'clear_min_mm', 'kappa', 'pressures_kpa']}
    rec['obs'].append(obs)
    done = False
    while not done:
        if model is None:
            act = np.array([1.0, 0, 0, 0, 0], np.float32)
        else:
            act, _ = model.predict(obs, deterministic=not a.stochastic)
        prev = env.ctrl.cmd_depth
        obs, r, term, trunc, info = env.step(act)
        st = env._state()
        terms = dict(r_depth=ve.W_DEPTH * (env.ctrl.cmd_depth - prev) * 1e3, r_tip=-ve.W_TIP * st['tip_err'],
                     r_shape=-env.w_shape * st['shape_err'], r_force=-ve.W_FORCE * st['force'], r_time=-ve.W_TIME)
        terms['r_terminal'] = r - sum(terms.values())
        for k, v in terms.items():
            rec[k].append(v)
        rec['obs'].append(obs)
        rec['action'].append(np.asarray(act, float))
        rec['reward'].append(r)
        for k in ['depth_mm', 'tip_err_mm', 'goal_err_mm', 'shape_err_mm', 'force', 'clear_min_mm']:
            rec[k].append(info[k] if k in info else st[k.replace('_mm', '')])
        rec['kappa'].append(env.kappa.copy())
        rec['pressures_kpa'].append(np.array([np.asarray(p, float) for p in env.ctrl.cmd_pressures]) / 1e3)
        done = term or trunc
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(success=info['success'], w_shape=a.w_shape, goal=env.goal, path=env.path,
               constants=np.array([ve.W_DEPTH, ve.W_TIP, ve.W_FORCE, ve.W_TIME, ve.SUCCESS_BONUS, ve.FAIL_PENALTY,
                                   ve.GOAL_SCALE_MM, ve.GOAL_TOL_MM, ve.DEPTH_STEP_MM, ve.DKAPPA, ve.KAPPA_MAX]))
    np.savez(a.out, **out)
    print(f'{len(out["action"])} steps, return {out["reward"].sum():.2f}, goal error {out["goal_err_mm"][-1]:.2f} mm, '
          f'success {info["success"]}', flush=True)
    env.close()


if __name__ == '__main__':
    main()
