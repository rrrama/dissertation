#!/bin/bash
#
#SBATCH --job-name=math_baseline
#SBATCH --partition=gecko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --mem=120G

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
conda activate llada
module load CUDA/12.6.2

export HF_HOME=/dcs/large/u5751903/models/

pip install colorlog python-dotenv
pip install "datasets==2.19.0" "numpy<2.0"

srun bash /dcs/pg25/u5751903/dev/Memento/run_math_baseline.sh
