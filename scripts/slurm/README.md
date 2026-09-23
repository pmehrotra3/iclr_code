# Running the unweighted sweep on a SLURM cluster

Files: `train.sbatch` (48-task array, one per (process, d, K) cell, 3 seeds each),
`eval.sbatch` (6-task array, one per (process, T_true)), `submit.sh` (submits both, eval
chained after train with `--dependency=afterok`). Same grid as `scripts/main.sh` with
`DIMS="2 4 8 16 32 64" KS="2 4 8 16" TTRAIN=500 WEIGHTED=false`.

## 1. Find out what the cluster offers (run once, on the login node)

```bash
sinfo -s                                  # partitions, their state and node counts
sinfo -o "%P %G %D %c %m %l"              # per partition: GRES (gpu types), nodes, cpus, RAM, max walltime
sinfo -p <partition> -o "%N %G %t"        # which nodes in a partition have which GPUs, and their state
scontrol show partition <partition>       # MaxTime, DefaultTime, MaxNodes, AllowAccounts ...
sacctmgr show assoc user=$USER format=account,partition,qos   # accounts/QOS you may submit under
squeue -p <partition> | wc -l             # how busy it is
module avail cuda                         # CUDA modules (only if your python env needs one)
module avail python anaconda miniconda    # if you use a module-provided python/conda
```

## 2. Set up the environment

```bash
git clone <repo> && cd iclr_code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt           # torch with CUDA wheels matching the cluster's driver
python -c "import torch; print(torch.cuda.is_available())"   # on a GPU node / srun
```

Quick interactive test on one GPU before submitting the array:

```bash
srun -p <partition> --gres=gpu:1 --mem=16G --time=00:20:00 --pty bash
source venv/bin/activate
python code/main.py process=ddim data.weighted=false sweep.d=[2] sweep.K=[2] n_seeds=3 \
    stages=[train] run_id=test device=cuda
```

## 3. Edit the `<-- site` lines in train.sbatch and eval.sbatch

| line | set to |
|---|---|
| `--partition` | a GPU partition from `sinfo -s` |
| `--gres` | `gpu:1` or the typed form your site uses, e.g. `gpu:a100:1` |
| `--time` | below the partition MaxTime; d=64,K=16 (3 seeds) took ~25 min on an L40S, so 06:00:00 is generous |
| `--account` / `--qos` | add `#SBATCH --account=<acct>` if `sacctmgr` shows you need one |
| `module load` / `source venv` | whatever activates your python |
| `%8` in `--array=0-47%8` | how many cells to run concurrently |

## 4. Submit and watch

```bash
./scripts/slurm/submit.sh                 # prints both job ids and the run_id
squeue -u $USER                           # PD = pending, R = running, CG = completing
sacct -j <train_jobid> --format=JobID,State,Elapsed,MaxRSS -X   # per-task state, wall time, RAM
tail -f slurm/atlas-train_<train_jobid>_0.out
scancel <jobid>                           # whole array;  scancel <jobid>_7  one task
```

Outputs: checkpoints in `data/<process>/unweighted/checkpoints/`, ground truth in
`data/<process>/unweighted/gt_cache/`, results in `output/<run_id>/<process>/unweighted/T<T>/`.

## Resubmitting

Training is idempotent (`train.force_retrain=false`): a cell whose 3 checkpoints exist is
skipped, so after a time-limit kill just resubmit the failed indices, e.g.

```bash
sacct -j <train_jobid> -X --format=JobID,State | grep -v COMPLETED       # find them
sbatch --array=23,47 --export=ALL,RUN_ID=<run_id> scripts/slurm/train.sbatch
sbatch --export=ALL,RUN_ID=<run_id> scripts/slurm/eval.sbatch             # then eval
```

Task index -> cell: `i = process_index*24 + d_index*4 + K_index` with processes (ddim, flow),
d in (2,4,8,16,32,64), K in (2,4,8,16); so 0..23 are ddim, 24..47 flow, and K varies fastest.

## Copying results back

```bash
rsync -av <cluster>:<path>/iclr_code/output/<run_id>/ output/<run_id>/
rsync -av <cluster>:<path>/iclr_code/checkpoints/ checkpoints/   # trained models + gt caches, optional
```
