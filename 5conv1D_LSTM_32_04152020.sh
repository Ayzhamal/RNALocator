#!/bin/bash
#SBATCH --job-name=5Conv1D_LSTM_32_FOR
#SBATCH --output=output_5Conv1D_LSTM_32.txt #output of your pogram prints here
#SBATCH --mail-user=zhamangaraea1@gator.uhd.edu #email
#SBATCH --error=error_5Conv1D_LSTM_32.txt #file where any error will be written
#SBATCH --mail-type=ALL

python /home/angiez1/Conv1D_mRNA/scripts/FOR_04152020_5Conv1D_LSTM_f32.py
