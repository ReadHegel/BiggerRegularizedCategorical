# Entropy
U need to scan files entropy and change user name accordigly 

## Running jobs
- Avoid computations on the Entropy login node (*asusgpu0*):
  use [`srun` or `sbatch`](https://entropy-doc.mimuw.edu.pl/submittingjobs.html).

Example training:
- `cd` (the repository in your entropy home directory).
- Check and edit: `entropy/submit.sh`
- Submit job: `./entropy/submit.sh`
- Show logs: `JOB_ID=$(sacct -XPno jobid | tail -1); tail --follow logs/slurm-"$JOB_ID"-*`
- Cancel/interrupt: `scancel "$JOB_ID"`

## Disk usage
- Larger stuff can be stored in node-local volumes [`/storage_*`](https://entropy-doc.mimuw.edu.pl/inputdata.html#storage-type-n-directories).