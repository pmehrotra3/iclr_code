# flow — Flow Matching with the Optimal-Transport conditional path

`process.py` implements the [`common.process.Process`](../common/process.py) contract following
Lipman et al. (2022), Example II (`t = 0` noise, `t = 1` data):

- **train_model** — conditional flow matching: `psi_t(x0) = (1 - (1 - sigma_min) t) x0 + t x1`,
  target `x1 - (1 - sigma_min) x0` (`flow.sigma_min`).
- **sample** — integrate the learned velocity on a uniform grid of `sweep.T_train` points with
  `flow.solver` ∈ {euler, midpoint, rk4}.
- **true_forward** — integrate the closed-form marginal OT velocity of the reference GMM
  (responsibility-weighted conditional velocities) with the same solver.

```bash
python code/flow/main.py                                  # full pipeline, defaults in conf/config.yaml
python code/flow/main.py flow.solver=rk4 stages=[evaluate,visualize]
python code/flow/main.py classifier=ladder sweep=ladder
python code/flow/main.py eval.labels=true eval.tag=true   # exact-velocity control
```

All shared knobs are documented in [`common/conf/base.yaml`](../common/conf/base.yaml) and the
top-level README.
