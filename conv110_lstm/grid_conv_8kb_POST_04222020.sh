#!/bin/bash
#SBATCH --job-name=grid_conv_8kb_POST
#SBATCH --output=output_grid_conv_8kb_POST.txt #output of your pogram prints here
#SBATCH --mail-user=zhamangaraea1@gator.uhd.edu #email
#SBATCH --error=error_grid_conv_8kb_POST.txt #file where any error will be written
#SBATCH --mail-type=ALL

python /home/angiez1/Senior_Project/grid_conv_8kb_POST_04222020.py
