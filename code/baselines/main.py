"""code/baselines/main.py — entry point for the RODS / IQ baseline processes.

    python code/baselines/main.py process=rods_cas stages=[evaluate,visualize]
    python code/baselines/main.py process=rods_sas stages=[evaluate,visualize]
    python code/baselines/main.py process=iq       stages=[evaluate,visualize]

The explicit `import baselines.process` below matters: common.process.load_process
falls back to `importlib.import_module(f"{name}.process")`, which assumes the
process name equals its folder name. Three processes live in one folder here, so
we populate the registry up front and that fallback is never reached.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baselines.process  # noqa: F401,E402  -- registers rods_cas / rods_sas / iq
from common.stages import STAGES  # noqa: E402
from baselines import basin  # noqa: E402
from common.main import run  # noqa: E402

# `basin` compares every sampler against the analytic reference on shared seeds.
# STAGES is a plain dict, so adding to it is all the registration there is.
STAGES["basin"] = basin.run

if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf"))