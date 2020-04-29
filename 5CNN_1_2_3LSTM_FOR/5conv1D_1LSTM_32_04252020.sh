#!/bin/bash
#SBATCH --job-name=5Conv_LSTM_32_FOR
#SBATCH --output=output_5Conv_LSTM_32.txt #output of your pogram prints here
#SBATCH --mail-user=zhamangaraea1@gator.uhd.edu #email
#SBATCH --error=error_5Conv_LSTM_32.txt #file where any error will be written
#SBATCH --mail-type=ALL

python /home/angiez1/Senior_Project/FOR_04252020_5Conv_1LSTM_f32.py
