#!/bin/bash
# embedded options to bsub - start with #BSUB

# -- job name --
#BSUB -J pfl_simulation

# -- queue --
#BSUB -q hpc

# -- number of CPU cores --
#BSUB -n 20

# -- keep all cores on one node for shared-memory Python/TensorFlow work --
#BSUB -R "span[hosts=1]"

# -- memory requested per core/slot; adjust if the job needs more --
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB

# -- wall clock time: hh:mm --
#BSUB -W 24:00

# -- output and error files; %J is replaced by the LSF job id --
#BSUB -o pfl_simulation_%J.out
#BSUB -e pfl_simulation_%J.err

set -euo pipefail

cd "${LS_SUBCWD:-$(dirname "$0")}"

CORES="${LSB_DJOB_NUMPROC:-20}"

export OMP_NUM_THREADS="$CORES"
export MKL_NUM_THREADS="$CORES"
export OPENBLAS_NUM_THREADS="$CORES"
export NUMEXPR_NUM_THREADS="$CORES"
export TF_NUM_INTRAOP_THREADS="$CORES"
export TF_NUM_INTEROP_THREADS=2

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON:-python3}"

echo "Starting simulation on $(hostname)"
echo "Working directory: $(pwd)"
echo "Using ${CORES} CPU cores"
echo "Python: $("$PYTHON_BIN" --version)"

"$PYTHON_BIN" -m simulations.simulation.py
