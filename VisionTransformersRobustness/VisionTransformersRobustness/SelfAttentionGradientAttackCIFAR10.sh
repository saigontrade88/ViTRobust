#! /bin/bash
#SBATCH --job-name="SAGA_Attack_ViT"
#SBATCH --output=out_SAGA_CIFAR10_%j.txt
#SBATCH --error=err_SAGA_CIFAR10_%j.txt
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --mem=90G
#SBATCH --time=10:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=snsm_itn19  
#SBATCH --qos=snsm19_special

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python3 VisionTransformersRobustness.py

#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

conda deactivate





