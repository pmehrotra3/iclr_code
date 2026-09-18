# flow — OT flow matching, deterministic ODE sampler

`process.py` implements the [`common.process.Process`](../common/process.py) contract:

- **train_model** — conditional flow matching with the OT path (Lipman et al., Eq. 20-23,
  `flow.sigma_min`), optimised by `Process.fit` (cosine lr, gradient clipping, EMA weights).
- **sample** — integrate the learned velocity `t: 0 -> 1` with `flow.solver` (euler by default):
  the learned sampler, the ground truth.
- **true_forward / true_backward** — the closed-form marginal OT velocity of the reference GMM,
  integrated with `flow.true_solver` (heun) so the two passes are exact inverses. Used for the
  analytic control and the anchor backtrack.

```bash
python code/flow/main.py                                  # train -> atlas -> atlas_viz, sweep=full
python code/flow/main.py sweep=d2 classifier=fast
python code/flow/main.py flow.sigma_min=0.001 run_tag=smin1e-3
```
