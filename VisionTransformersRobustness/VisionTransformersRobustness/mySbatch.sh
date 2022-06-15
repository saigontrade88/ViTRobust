#! /bin/bash
#SBATCH --job-name="ViT"
#SBATCH --output=out_ViT_%j.txt
#SBATCH --error=err_ViT_%j.txt
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:2
#SBATCH --mem=90G
#SBATCH --time=20:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short
#SBATCH --array=0-1

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

METHODS=(3)
echo "JOBID: $SLURM_JOB_ID"
echo "METHOD: ${METHODS[$SLURM_ARRAY_TASK_ID]}"


python3 VisionTransformersRobustness.py ${METHODS[$SLURM_ARRAY_TASK_ID]}

wait

#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short
#SBATCH --partition=snsm_itn19  
#SBATCH --qos=snsm19_special

conda deactivate





