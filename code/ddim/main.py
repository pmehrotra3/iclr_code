"""code/ddim/main.py — run the pipeline for the ddim process. See conf/config.yaml and README."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.main import run  # noqa: E402

if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf"))
