#!/bin/bash
#SBATCH --job-name=hbbb_exp1               # Nome do job
#SBATCH --output=hbbb_exp1.log             # Arquivo para log de saída
#SBATCH --error=hbbb_exp_erro1.log         # Arquivo para log de erros
#SBATCH --partition=long
#SBATCH --ntasks=1                      # Número de tarefas (geralmente 1 para Python)
#SBATCH --cpus-per-task=16              # CPUs por tarefa
#SBATCH --mem=32G                       # Quantidade de memória alocada

source $HOME/HBBB-Experiments/venv/bin/activate
python3 evaluation/analysis.py
