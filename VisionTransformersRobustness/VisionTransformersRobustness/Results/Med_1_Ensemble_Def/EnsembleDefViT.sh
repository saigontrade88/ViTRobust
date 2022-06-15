#! /bin/bash
#SBATCH --job-name="EnsembleDefViT"
#SBATCH --output=out_ViT_RayS_Def_%j.txt
#SBATCH --error=err_ViT__RayS_Def_%j.txt
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --mem=90G
#SBATCH --time=20:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RayS Attack Against the Shuffle Ensemble Defense"

python3 VisionTransformersRobustness.py 1
#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

conda deactivate





