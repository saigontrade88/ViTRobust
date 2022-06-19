#! /bin/bash
#SBATCH --job-name="ParalleRayS"
#SBATCH --output=out_ParalleRayS_%j.txt
#SBATCH --error=err_ParalleRayS_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python3 main_RayS.py -m 1 -g 3 -b 32 -d Cifar10\
			    -numClasses 10 \
			    -trainSampleNum 50000 \
				-validSampleNum 10000 \
				-attackSampleNum 100 \
				-epsMax 0.031 \
				-qLimit 10000 \
				-savedFilePath data/0_RayS_Attack/Cifar-10/Distributed


#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short
#SBATCH --partition=snsm_itn19  
#SBATCH --qos=snsm19_special

conda deactivate





