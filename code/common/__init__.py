"""common — shared library for the seed-fate experiments.

A *process* (ddim/, flow/, ...) is a generative sampler that maps a Gaussian seed to a
data-space endpoint. The library owns everything that does not depend on the process:
the Gaussian-mixture reference (gmm), the network backbone (nets), the seed -> fate
classifiers (fate), the process contract + registry (process), checkpoint I/O, the
pipeline stages, and the shared Hydra config groups under common/conf.
"""
from common import gmm, nets, fate, process, checkpoint, utils  # noqa: F401

__all__ = ["gmm", "nets", "fate", "process", "checkpoint", "utils"]