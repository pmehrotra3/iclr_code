# ddim — variance-preserving diffusion, deterministic DDIM sampler

`process.py` implements the [`common.process.Process`](../common/process.py) contract:

- **schedule** — continuous-time VP, `abar(t) = exp(-(beta_min t + (beta_max - beta_min) t²/2))`
  sampled at `sweep.T_train` points (`ddim.beta_min`, `ddim.beta_max`).
- **train_model** — eps-prediction (DDPM objective) on samples of the reference GMM.
- **sample** — deterministic DDIM, step index `T-1 → 0`.
- **true_forward** — the same DDIM recursion driven by the closed-form score of the noised
  mixture (the ideal sampler; used for the `analytic` predictor and for `eval.labels=true`).

```bash
python code/ddim/main.py                                  # full pipeline, defaults in conf/config.yaml
python code/ddim/main.py stages=[evaluate,visualize] classifier=ladder sweep=ladder
python code/ddim/main.py stages=[seedmap] classifier=polar
python code/ddim/main.py eval.labels=true eval.tag=true   # exact-score control
python code/ddim/main.py --cfg job                        # print the composed config
```

All shared knobs (`sweep`, `classifier`, `train`, `eval`, `seedmap`, `paths`) are documented in
[`common/conf/base.yaml`](../common/conf/base.yaml) and the top-level README.
