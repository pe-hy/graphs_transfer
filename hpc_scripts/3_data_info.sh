#!/bin/bash
#SBATCH --job-name=data_info                         # Job name
#SBATCH --output=logs/data_info/data_info_%j.out     # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/data_info/data_info_%j.err      # Error log
#SBATCH --time=0:15:00                               # Time limit hrs:min:sec
#SBATCH --account=project_465001424
#SBATCH --nodes=1                                    # Number of nodes requested
#SBATCH --ntasks=1                                   # Number of tasks (processes)
#SBATCH --gpus=1                                     # Number of GPUs requested
#SBATCH --cpus-per-task=16                           # Number of CPU cores per task
#SBATCH --mem=64GB                                   # Memory limit
#SBATCH --partition=small-g                          # Partition name

# export SINGULARITY_BIND="$SINGULARITY_BIND,/usr/bin/sacct,/usr/bin/sacctmgr,/usr/bin/salloc,/usr/bin/sattach,/usr/bin/sbatch,/usr/bin/sbcast,/usr/bin/scancel,/usr/bin/scontrol,/usr/bin/scrontab,/usr/bin/sdiag,/usr/bin/sinfo,/usr/bin/sprio,/usr/bin/squeue,/usr/bin/sreport,/usr/bin/srun,/usr/bin/sshare,/usr/bin/sstat,/usr/bin/strigger,/usr/bin/sview,/usr/bin/sgather,/usr/lib64/slurm/,/etc/slurm,/etc/passwd,/usr/lib64/libmunge.so.2,/run/munge,/var/lib/misc,/etc/nsswitch.conf"



singularity exec \
    $SIF \
    python utils/get_data_info.py