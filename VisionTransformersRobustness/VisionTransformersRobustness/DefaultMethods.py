#In this file we provide different methods to run attacks on different models 
import torch
import numpy
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersRayS
import AttackWrappersAdaptiveBlackBox
import AttackWrappersSAGA
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels
from collections import OrderedDict

import logging

#Set up logger
logging.basicConfig(filename='info.log',
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)-6s %(name)s :: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

#Load the ViT-L-16 and CIFAR-10 dataset 
def LoadViTLAndCIFAR10():
    #Basic variable and data setup
    device = torch.device("cuda")
    logger.info("Device {}".format(device))
    numClasses = 10
    imgSize = 224
    batchSize = 8
    #Load the CIFAR-10 data
    logger.info('Load the CIFAR-10 data')
    valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)
    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    dir = "Models/ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    logger.info('Load pre-train model weight and then model evaluation')
    model.eval() # Set model to evaluation mode

    #Wrap the model in the ModelPlus class
    modelPlus = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    logger.info('Wrap the model in the ModelPlus class.')
    return valLoader, modelPlus

#Load the shuffle defense containing ViT-L-16 and BiT-M-R101x3
#For all attacks except SAGA, vis should be false (makes the Vision tranformer return the attention weights if true)
def LoadShuffleDefenseAndCIFAR10(vis=False):
    modelPlusList = []
    #Basic variable and data setup
    device = torch.device("cuda")
    logger.info("Device {}".format(device))
    numClasses = 10
    imgSize = 224
    batchSize = 8 
    #Load the CIFAR-10 data
    valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)
    data_inputs, data_labels = next(iter(valLoader))
    logger.info('LoadShuffleDefenseAndCIFAR10 :: Sucessfully load the validation data. Print basic info of the first batch')
    logger.info('LoadShuffleDefenseAndCIFAR10 :: Data inputs {}'.format(data_inputs.shape))
    logger.info('LoadShuffleDefenseAndCIFAR10 :: Data labels {}'.format(data_labels.shape))

    #Save some input for debugging if necessary

    #Load ViT-L-16 (Vision Transformer)
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
    dir = "Models/ViT-L_16,cifar10,run0_15K_checkpoint.bin"

    dict = torch.load(dir)
    model.load_state_dict(dict)

    model.eval() #Set model to eval mode
    #Wrap the model in the ModelPlus class
    modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    modelPlusList.append(modelPlusV)
    logger.info('LoadShuffleDefenseAndCIFAR10:: Successfully load pre-train model {} to device'.format(dir, device))

    #Load the BiT-M-R101x3 (Big Transfer Model)
    dirB = "Models/BiT-M-R101x3-Run0.tar" #checkpoint
    modelB = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)

    #Get the checkpoint 
    checkpoint = torch.load(dirB, map_location="cpu")
    #Remove module so that it will load properly
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    #Load the learnable parameters
    modelB.load_state_dict(new_state_dict)
    modelB.eval() #Set model to eval mode

    #Wrap the model in the ModelPlus class
    #Here we hard code the Big Transfer Model Plus class input size to 160x128 (what it was trained on)
    modelBig101Plus = ModelPlus("BiT-M-R101x3", modelB, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
    modelPlusList.append(modelBig101Plus)
    logger.info('LoadShuffleDefenseAndCIFAR10 :: Successfully load pre-train model {} to device'.format(dirB, device))
    
    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    logger.info('LoadShuffleDefenseAndCIFAR10 :: Successfully create a ShuffleDefense object')

    return valLoader, defense

#Method to do the RayS attack - query based blackbox attack on a single Vision Transformers
def RaySAttackVisionTransformer():
    #Load the model and dataset
    logger.info('Load the model (model architecture and model weights), dataset, and model evaluation')
    valLoader, defense = LoadViTLAndCIFAR10()
    #Get the clean samples
    numClasses = 10
    attackSampleNum = 100 #1000
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)
    #Set the attack parameters 
    epsMax = 0.031
    queryLimit = 10000 #10000
    #The next line does the actual attack on the defense 
    advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader)
    #Check the results 
    robustAcc = defense.validateD(advLoader)
    cleanAcc = defense.validateD(valLoader)
    #logger.info the results 
    logger.info("Queries used: {}".format(queryLimit))
    logger.info("Robust acc: {}".format(robustAcc))
    logger.info("Clean acc: {}".format(cleanAcc))

#Here we do the RayS attack on a shuffle defense comprised of two models, ViT-L-16 and BiT-M-R101x3
def RaySAttackShuffleDefense():
    #Load the model and dataset
    valLoader, defense = LoadShuffleDefenseAndCIFAR10()
    #Get the clean samples
    numClasses = 10
    attackSampleNum = 10 #1000
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)

    data_inputs, data_labels = next(iter(cleanLoader))
    logger.info('Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
    logger.info('Data inputs {}'.format(data_inputs.shape))
    logger.info('Data labels {}'.format(data_labels.shape))

    #Set the attack parameters 
    epsMax = 0.031
    queryLimit = 100 #10000
    #The next line does the actual attack on the defense 
    advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader)

    data_inputs, data_labels = next(iter(advLoader))
    logger.info('Sucessfully generate adversarial examples that fool BOTH models. Print basic info of the first batch')
    logger.info('Data inputs {}'.format(data_inputs.shape))
    logger.info('Data labels {}'.format(data_labels.shape))

    #Check the results
    logger.info('Robust accuracy of the ensemble model')
    robustAcc = defense.validateD(advLoader)
    logger.info('Clean accuracy of the ensemble model')
    cleanAcc = defense.validateD(valLoader)

    #Print the results 
    logger.info("Queries used: {}".format(queryLimit))
    logger.info("Robust acc: {}".format(robustAcc))
    logger.info("Clean acc: {}".format(cleanAcc))

#Run the 100% strength adaptive attack on ViT-L-16, transfer based blackbox attack
def AdaptiveAttackVisionTransformer():
    #Corresponding tag for saving files
    #First part indicates the type of defense, second part indidcates the synthetic model and last part indicates the strenght of the attack (100%)
    saveTag = "ViT-L-16, ViT-32(ImageNet21K), p100" 
    device = torch.device("cuda")
    logger.info("Device {}".format(device))

    #Attack parameters 
    numAttackSamples = 100 # 1000
    epsForAttacks = 0.031
    clipMin = 0.0 
    clipMax = 1.0
    #Parameters of training the synthetic model 
    imgSize = 224
    batchSize = 32
    numClasses = 10
    numIterations = 1 # 4
    epochsPerIteration = 1 #1 
    epsForAug = 0.1 #when generating synthetic data, this value is eps for FGSM used to generate synthetic data
    learningRate = (3e-2) / 2 #Learning rate of the synthetic model 
    #Load the training dataset, validation dataset and the defense 
    valLoader, defense = LoadViTLAndCIFAR10()

    trainLoader = DMP.GetCIFAR10Training(imgSize, batchSize)
    data_inputs, data_labels = next(iter(trainLoader))
    logger.info('Sucessfully load the train data. Print basic info of the first batch')
    logger.info('Data inputs {}'.format(data_inputs.shape))
    logger.info('Data labels {}'.format(data_labels.shape))

    #Get the clean data 
    xTest, yTest = DMP.DataLoaderToTensor(valLoader)

    #Clean validation loader
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
    data_inputs, data_labels = next(iter(cleanLoader))
    logger.info('Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
    logger.info('Data inputs {}'.format(data_inputs.shape))
    logger.info('Data labels {}'.format(data_labels.shape))


    #Create the synthetic model to generate adversarial examples
    syntheticDir = "Models//imagenet21k_ViT-B_32.npz" #imagenet21k_ViT-B_32.npz
    config = CONFIGS["ViT-B_32"]
    syntheticModel = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    syntheticModel.load_from(numpy.load(syntheticDir))  
    syntheticModel.to(device)
    logger.info('Successfully load pre-train model {} to device'.format(syntheticDir, device))

    #Do the attack 
    oracle = defense
    dataLoaderForTraining = trainLoader
    optimizerName = "sgd"
    #Last line does the attack 
    AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, 
                                            numIterations, epochsPerIteration, epsForAug, 
                                            learningRate, optimizerName, dataLoaderForTraining, 
                                            cleanLoader, numClasses, epsForAttacks, clipMin, clipMax)

#Run the 100% strength adaptive attack on shuffle defense
def AdaptiveAttackShuffleDefense():
    #Corresponding tag for saving files
    #First part indicates the type of defense, second part indidcates the synthetic model and last part indicates the strenght of the attack (100%)
    saveTag = "Adaptive_Attack_Defense_ViT-L-16, ViT-32(ImageNet21K), p100" 
    device = torch.device("cuda")
    logger.info("Device {}".format(device))

    #Attack parameters 
    numAttackSamples = 10 #1000
    epsForAttacks = 0.031
    clipMin = 0.0 
    clipMax = 1.0

    #Parameters of training the synthetic model 
    imgSize = 224
    batchSize = 32
    numClasses = 10
    numIterations = 1 #4
    epochsPerIteration = 1 #10
    epsForAug = 0.1 #when generating synthetic data, this value is eps for FGSM used to generate synthetic data
    learningRate = (3e-2) / 2 #Learning rate of the synthetic model

    #Load the training dataset, validation dataset and the defense 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10()
    logger.info('AdaptiveAttackShuffleDefense :: Sucessfully load the ensemble defense. Print basic info of the first batch')

    trainLoader = DMP.GetCIFAR10Training(imgSize, batchSize)
    logger.info('AdaptiveAttackShuffleDefense :: Sucessfully load the train data. Print basic info of the first batch')
    logger.info('AdaptiveAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
    logger.info('AdaptiveAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

    #Get the clean data 
    xTest, yTest = DMP.DataLoaderToTensor(valLoader)
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)

    logger.info('AdaptiveAttackShuffleDefense :: Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
    logger.info('AdaptiveAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
    logger.info('AdaptiveAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

    #Create the synthetic model 
    syntheticDir = "Models//imagenet21k_ViT-B_32.npz"
    config = CONFIGS["ViT-B_32"]
    syntheticModel = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    syntheticModel.load_from(numpy.load(syntheticDir))  
    syntheticModel.to(device)
    logger.info('AdaptiveAttackShuffleDefense :: Successfully load pre-train model {} to device'.format(syntheticDir, device))

    #Do the attack 
    oracle = defense
    dataLoaderForTraining = trainLoader
    optimizerName = "sgd"
    #Last line does the attack 
    AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, 
                                                    numIterations, epochsPerIteration, epsForAug, 
                                                    learningRate, optimizerName, dataLoaderForTraining, 
                                                    cleanLoader, numClasses, epsForAttacks, clipMin, clipMax)

#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def SelfAttentionGradientAttackCIFAR10():
    logger.info("Running Self-Attention Gradient Attack on ViT-L-16 and BiT-M-R101x3")
    #Set up the parameters for the attack 
    attackSampleNum = 10 #1000
    numClasses = 10
    coefficientArray = torch.zeros(2)
    secondcoeff = 2.0000e-04
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    logger.info("Coeff Array:")
    logger.info(coefficientArray)
    device = torch.device("cuda")
    logger.info("Device {}".format(device))

    epsMax = 0.031
    clipMin = 0.0
    clipMax = 1.0
    numSteps = 10
    #Load the models and the dataset
    #Note it is important to set vis to true so the transformer's model output returns the attention weights 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)
    modelPlusList = defense.modelPlusList
    #Note that the batch size will effect how the gradient is computed in PyTorch
    #Here we use batch size 8 for ViT-L and batch size 2 for BiT-M. Other batch sizes are possible but they will not generate the same result
    modelPlusList[0].batchSize = 8
    modelPlusList[1].batchSize = 2
    #Get the clean examples 
    cleanLoader =AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, attackSampleNum, numClasses, valLoader, modelPlusList)
    #Do the attack
    advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, numSteps, modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax)
    #Go through and check the robust accuray of each model on the adversarial examples 
    for i in range(0, len(modelPlusList)):
        acc = modelPlusList[i].validateD(advLoader)
        logger.info('model Name {} Robust acc: {}'.format(modelPlusList[i].modelName, acc))