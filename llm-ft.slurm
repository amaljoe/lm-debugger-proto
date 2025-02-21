#!/bin/bash
#SBATCH --job-name=llm_finetune            # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (processes)
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --mem=8G                          # Total memory per node
#SBATCH --time=00:30:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                    # Partition name
#SBATCH --output=finetune_output.log       # Standard output log
#SBATCH --error=finetune_error.log         # Standard error log

# Load necessary modules
module load mldl_modules/miniconda

conda init
source ~/.bashrc
conda activate myenv

# Navigate to project directory
cd /home/IITB/cfilt/24m0797/llm-ft

# Run fine-tuning script
python3 gpt2.py