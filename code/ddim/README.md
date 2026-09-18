# ddim — variance-preserving diffusion, deterministic DDIM sampler

`process.py` implements the [`common.process.Process`](../common/process.py) contract:

- **schedule** — continuous-time VP, `abar(t) = exp(-(beta_min t + (beta_max - beta_min) t²/2))`
  sampled at `sweep.T_train` points (`ddim.beta_min`, `ddim.beta_max`).
- **train_model** — eps-prediction (DDPM objective) on samples of the reference GMM, optimised by
  `Process.fit` (cosine lr, gradient clipping, EMA weights).
- **sample** — deterministic DDIM, step index `T-1 -> 0`: the learned sampler, the ground truth.
- **true_forward / true_backward** — the same DDIM recursion driven by the closed-form score of the
  noised mixture, with a Heun predictor-corrector step (`ddim.true_solver`) so the two passes are
  exact inverses. Used for the analytic control and the anchor backtrack.

```bash
python code/ddim/main.py                                  # train -> atlas -> atlas_viz, sweep=full
python code/ddim/main.py sweep=d2 classifier=fast         # d = 2, four predictors
python code/ddim/main.py stages=[atlas_viz]               # re-draw
python code/ddim/main.py --cfg job                        # print the composed config
```
