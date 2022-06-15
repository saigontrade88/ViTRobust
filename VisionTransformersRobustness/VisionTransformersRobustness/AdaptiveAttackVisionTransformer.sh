#! /bin/bash
#SBATCH --job-name="RaySAttackViT"
#SBATCH --output=out_ViT_%j.txt
#SBATCH --error=err_ViT_%j.txt
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=snsm_itn19  
#SBATCH --qos=snsm19_special

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python3 VisionTransformersRobustness.py 2

#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

conda deactivate





