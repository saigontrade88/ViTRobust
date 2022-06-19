#import standard lib
import argparse
import os
from datetime import datetime
import sys
import numpy as np

# #VIT robustness
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersRayS
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels
from collections import OrderedDict
#from DefaultMethods import LoadViTLAndCIFAR10

#torch
import torch
import torch.distributed as dist

import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler as DDP_sampler

#utils
from utils import rank_print, setup_distrib

transformTest = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
])

runID = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

test_set = datasets.CIFAR10(root='./data', 
	train=False, download=True, 
	transform=transformTest)

def net_setup():
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '5000'

#Save a dataloader to the directory 
def SaveLoader(filepath, timestamp, tag, dataLoader):

    if not os.path.isdir(filepath): #If not there, make the directory 
        os.makedirs(filepath)

    #Torch limits the amount of data we can save to disk so we must use numpy to save 
    #torch.save(dataLoader, self.homeDir+dataLoaderName)
    #First convert the tensor to a dataloader 
    xDataPytorch, yDataPytorch = DMP.DataLoaderToTensor(dataLoader)
    #Second conver the pytorch arrays to numpy arrays for saving 
    xDataNumpy = xDataPytorch.cpu().detach().numpy()
    yDataNumpy = yDataPytorch.cpu().detach().numpy()
    #Save the data using numpy

    dataLoaderPathX = '{}/{}_X_{}'.format(filepath, tag, timestamp)
    dataLoaderPathY = '{}/{}_Y_{}'.format(filepath, tag, timestamp)

    np.save(dataLoaderPathX, xDataNumpy)
    np.save(dataLoaderPathY, yDataNumpy)
    
    rank_print(f'Successfully save {tag} at {dataLoaderPathX}')
    #Delete the dataloader and associated variables from memory 
    del dataLoader
    del xDataPytorch
    del yDataPytorch
    del xDataNumpy
    del yDataNumpy

#Load the ViT-L-16 and CIFAR-10 dataset 
def LoadViTLAndCIFAR10(xData, yData, valLoader, validSampleNum, device):
    #Basic variable and data setup
  
    #rank_print("Device {device}")
    numClasses = 10
    imgSize = 224
    batchSize = 8
    #Load the CIFAR-10 data
    if xData and yData:
        print('Default_Methods::LoadViTLAndCIFAR10 :: Load data from file at {}'.format(xData))
        xData = np.load(xData)
        yData = np.load(yData)
        valLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), 
                                        torch.from_numpy(yData), 
                                        transforms = None, 
                                        batchSize = batchSize, 
                                        randomizer = None)
    elif valLoader is not None:
    	pass
    else:
        valLoader = DMP.GetCIFAR10Validation(validSampleNum, imgSize, batchSize)

    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    dir = "Models/ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    rank_print('Default_Methods::Load pre-train model weight.')
    model.eval() # Set model to evaluation mode

    #Wrap the model in the ModelPlus class
    modelPlus = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    rank_print('Default_Methods::Wrap the model in the ModelPlus class.')
    return valLoader, modelPlus
	
def test(local_rank, args):

	print(f"Process for rank: {local_rank} has been spawned")
	#Setup the distributed processing
	world_size = args.machines*args.gpus

	#args.mid: machine id number
	if args.machines == 1:
		rank = local_rank
		device = setup_distrib(local_rank, world_size)
	else:
		rank = args.mid * args.gpus + local_rank
		#There is a timeout param in here. Careful.
		dist.init_process_group('nccl', rank =rank, world_size = world_size)
		torch.cuda.set_device(local_rank)

	#Attack params
	numClasses = args.numClasses
	attackSampleNum = args.attackSampleNum #1000
	epsMax = args.epsMax#0.031
	queryLimit = args.qLimit #10000

	#Data partition and loading
	print(f"Load the test partition into GPU {rank}")
	local_test_sampler = DDP_sampler(test_set, 
									rank = rank, 
									num_replicas = world_size)


	local_test_loader = torch.utils.data.DataLoader(test_set, 
							batch_size=args.batch_size,
							shuffle = False, 
							sampler = local_test_sampler,
							num_workers=1,
							pin_memory=True)

	rank_print(f"Test data received")
	rank_print(f'Batch size: {local_test_loader.batch_size}')
	rank_print(f'Number of batches: {len(local_test_loader)}')
	
	# xValData, yValData = DMP.DataLoaderToTensor(local_test_loader)

	# rank_print(f'xValData: {xValData.shape}')
	# rank_print(f'yValData: {yValData.shape}')
	
	#Load the ViT model
	valLoader, defense = LoadViTLAndCIFAR10(None, None, local_test_loader,
													args.validSampleNum, device)

	rank_print(f'valLoader Batch size: {valLoader.batch_size}')
	rank_print(f'valLoader Number of batches: {len(valLoader)}')

	cleanAcc, tp_5 = defense.validateD(valLoader)
	rank_print(f"Before attack :: Clean acc:{cleanAcc}\n")
	rank_print(f"Before attack :: Top-5 acc:{tp_5}\n")

	# dist.barrier()
	# #Only attack correctly classified examples
	if args.xCleanPath and args.yCleanPath:
		xClean = args.xCleanPath + '/{}_cleanLoader_X_{}.npy'.format(args.data, rank)  
		yClean = args.yCleanPath + '/{}_cleanLoader_Y_{}.npy'.format(args.data, rank)

		xData = np.load(xClean)
		yData = np.load(yClean)
		cleanLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), 
		                            torch.from_numpy(yData), 
		                            transforms = None, 
		                            batchSize = valLoader.batch_size, 
		                            randomizer = None)
		rank_print(f'RaySAttackVisionTransformer :: Load correctly classified examples from file at {yClean}')
	else:
		
		cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, 
                                                                args.attackSampleNum, 
                                                                valLoader, 
                                                                args.numClasses)
		filepath = os.path.join(args.savedFilePath, runID)

		SaveLoader(filepath, rank, 'Cifar10_cleanLoader_{}'.format(args.batch_size), cleanLoader)

	data_inputs, data_labels = next(iter(cleanLoader))
	rank_print(f'Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
	rank_print(f'Data inputs {data_inputs.shape}')
	rank_print(f'Data labels {data_labels.shape}')
	rank_print(f'Length {len(cleanLoader)}')
	rank_print(f'Batch size {cleanLoader.batch_size}')

	#dist.barrier()
	
	###########Attack#####################
	begin = datetime.now()
	beginStr = begin.strftime('%Y-%m-%d %H:%M:%S')
	rank_print(f"Begin attack at {beginStr}\n")

	advLoader = AttackWrappersRayS.RaySAttack(defense, 
	                                epsMax, 
	                                queryLimit, 
	                                cleanLoader)

	now = datetime.now()
	nowStr = now.strftime('%Y-%m-%d %H:%M:%S')
	rank_print(f"Complete at {nowStr}\n")
	rank_print(f"--- {now - begin} seconds ---\n")

	# #Check the results 
	robustAcc, tp_5_adv = defense.validateD(advLoader)
	cleanAcc, tp_5 = defense.validateD(valLoader)
	# #print the results  
	rank_print(f"RaySAttackVisionTransformer :: Queries used:{queryLimit}\n")
	rank_print(f"RaySAttackVisionTransformer :: Robust acc:{robustAcc}\n")
	rank_print(f"RaySAttackVisionTransformer :: Top-5 acc on adversarial images:{tp_5_adv}\n")
	rank_print(f"RaySAttackVisionTransformer :: Clean acc:{cleanAcc}\n")
	rank_print(f"RaySAttackVisionTransformer :: Top-5 acc on clean images:{tp_5}\n")

	filepath = os.path.join(args.savedFilePath, runID)
	SaveLoader(filepath, rank, 'Cifar10_advLoader', advLoader)

	rank_print("Attack Done!")
	dist.destroy_process_group()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'distributed data parallel Rays Attack')
	parser.add_argument('-m', '--machines', default=1, type=int, help='number of machines')
	parser.add_argument('-g', '--gpus', default = 2, type=int, help='number of GPUs in a machine')
	parser.add_argument('-id', '--mid', default = 0, type=int, help='machine id number')
	parser.add_argument('-b', '--batch_size', default = 8, type = int, help='batch size')
	parser.add_argument('-d','--data', default='Cifar-10', type=str)

	parser.add_argument('-trainSampleNum','--trainSampleNum', default=1000, type=int)
	parser.add_argument('-validSampleNum','--validSampleNum', default=1000, type=int)


	parser.add_argument('-numClasses', '--numClasses', default = 10, type = int, help='numClasses')
	parser.add_argument('-attackSampleNum', '--attackSampleNum', default = 10, type = int, help='attackSampleNum')
	parser.add_argument('-qLimit', '--qLimit', default = 100, type = int, help='qLimit ')
	parser.add_argument('-epsMax', '--epsMax', default = 0.031, type = float, help='epsMax')

	parser.add_argument('-savedFilePath', '--savedFilePath', type = str, help='savedFilePath')

	#Debug
	parser.add_argument('-xCleanPath', '--xCleanPath', type=str, help='xClean correctly classified data')
	parser.add_argument('-yCleanPath', '--yCleanPath', type=str, help='yClean correctly classified data')

	args = parser.parse_args()

	start_time = datetime.now() 
	print("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))

	net_setup()
	mp.spawn(test, nprocs=args.gpus, args=(args,))

	now=datetime.now() 
	print("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
	print("--- {} seconds ---".format(now - start_time))
	sys.stdout.flush()
