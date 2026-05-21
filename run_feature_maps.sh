#!/bin/bash
#SBATCH --partition=cpu-max          # CPU-only partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=wtbs-fm
#SBATCH --output=/home/angel.encalada/Documents/ESPOL/Research/WTBSegmentation/bs-loss-function-wtbs/Pretraining/logs/runs/%x_%j.out
#SBATCH --error=/home/angel.encalada/Documents/ESPOL/Research/WTBSegmentation/bs-loss-function-wtbs/Pretraining/logs/runs/%x_%j.err

# ── paths ─────────────────────────────────────────────────────────────────────
#HOME_DATA=/home/angel.encalada/Documents/ESPOL/Research/WTBSegmentation/Datasets/Blade30
#HOME_MODELS=/home/angel.encalada/Documents/ESPOL/Research/WTBSegmentation/bs-loss-function-wtbs/Pretraining
REPO=/home/angel.encalada/Documents/ESPOL/Research/WTBSegmentation/bs-loss-function-wtbs
SCRIPT_DIR=${REPO}/Pretraining

# $SCRATCH is set by CEDIA's SLURM to /scratch/angel.encalada/$SLURM_JOB_ID
# it exists only during the job and is wiped afterwards — perfect for fast I/O
#echo "[$(date)] SCRATCH dir: ${SCRATCH}"

# ── 1. create scratch dirs ────────────────────────────────────────────────────
#mkdir -p ${SCRATCH}/data
#mkdir -p ${SCRATCH}/models

# ── 2. copy dataset from NFS home → scratch ───────────────────────────────────
#echo "[$(date)] Copying dataset to scratch..."
#cp -r ${HOME_DATA} ${SCRATCH}/data
#echo "[$(date)] Done — $(du -sh ${SCRATCH}/data | tail -1)"

# ── 3. activate environment ───────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
#conda activate pywakesim

# ── 4. train using scratch paths ─────────────────────────────────────────────
#export DATA_DIR=${SCRATCH}/data
#export MODEL_DIR=${SCRATCH}/models

cd ${SCRIPT_DIR}

echo "[$(date)] Starting plotting on $(hostname)..."
python "WTBSegmentation_PT_FeatureMaps.py"

echo "[$(date)] Plotting complete."

# ── 5. copy models back to permanent NFS storage ──────────────────────────────
#echo "[$(date)] Copying models back to ${HOME_MODELS}..."
#cp -r ${SCRATCH}/models/* ${HOME_MODELS}/
#echo "[$(date)] Done."
# no need to rm -rf — SLURM wipes $SCRATCH automatically when the job ends
