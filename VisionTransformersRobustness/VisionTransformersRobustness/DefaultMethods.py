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
from datetime import datetime
import os
import sys
## Imports for plotting
import matplotlib.pyplot as plt
%matplotlib inline 
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

import logging

#Set up logger
# logging.basicConfig(filename='info.log',
#                     level=logging.INFO,
#                     format='%(asctime)s.%(msecs)03d %(levelname)-6s %(name)s :: %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger(__name__)

def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']  
    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        #img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()
    
    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None and adv_img is None:
        print('Case 1')
        fig, ax = plt.subplots(1, 2, figsize=(15,2), gridspec_kw={'width_ratios': [1, 1]})
    elif noise is None and adv_img is not None:
        print('Case 2')
        fig, ax = plt.subplots(1, 4, figsize=(15,2), 
                               gridspec_kw={'width_ratios': [1, 1, 1, 2]})
    else:
        print('Case 3')
        fig, ax = plt.subplots(1, 5, figsize=(15,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})
    
    #Draw the clean image
    ax[0].imshow(img)
    ax[0].set_title(class_names[label])
    ax[0].axis('off')
    
    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        #adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        noise = noise.cpu().permute(1, 2, 0).numpy()
        noise = noise * 0.5 + 0.5 # Scale between 0 to 1 
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')
    if adv_img is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        #adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # # Visualize noise
        ax[2].axis('off')
        # # buffer
        #ax[3].axis('off')
    
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([class_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')
    
    #plt.show()
    if adv_img is not None:
      print('Save adversarial images')
      plt.savefig('adv_img_{}.pdf'.format(class_names[label]), bbox_inches='tight')
    else:
      plt.savefig('clean_img_{}.pdf'.format(class_names[label]), bbox_inches='tight')
    plt.close()

#Load the ViT-L-16 and CIFAR-10 dataset 
def LoadViTLAndCIFAR10(xData, yData):
    #Basic variable and data setup
    device = torch.device("cuda")
    print("Device {}".format(device))
    numClasses = 10
    imgSize = 224
    batchSize = 8
    #Load the CIFAR-10 data
    if xData and yData:
        print('Default_Methods::LoadViTLAndCIFAR10 :: Load data from file at {}'.format(xData))
        xData = numpy.load(xData)
        yData = numpy.load(yData)
        valLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), 
                                        torch.from_numpy(yData), 
                                        transforms = None, 
                                        batchSize = batchSize, 
                                        randomizer = None)
    else:
        valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)

    #Load ViT-L-16
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    dir = "Models/ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    print('Default_Methods::Load pre-train model weight and then model evaluation')
    model.eval() # Set model to evaluation mode

    #Wrap the model in the ModelPlus class
    modelPlus = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    print('Default_Methods::Wrap the model in the ModelPlus class.')
    return valLoader, modelPlus

#Load the shuffle defense containing ViT-L-16 and BiT-M-R101x3
#For all attacks except SAGA, vis should be false (makes the Vision tranformer return the attention weights if true)
def LoadShuffleDefenseAndCIFAR10(vis=False, xData, yData):
    modelPlusList = []
    #Basic variable and data setup
    device = torch.device("cuda")
    print("Device {}".format(device))
    numClasses = 10
    imgSize = 224
    batchSize = 8 
    #Load the CIFAR-10 data
    if xData and yData:
        print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Load data from file at {}'.format(xData))
        xData = numpy.load(xData)
        yData = numpy.load(yData)
        valLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), 
                                        torch.from_numpy(yData), 
                                        transforms = None, 
                                        batchSize = batchSize, 
                                        randomizer = None)
    else:
        valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)

    data_inputs, data_labels = next(iter(valLoader))
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Sucessfully load the validation data. Print basic info of the first batch')
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Data inputs {}'.format(data_inputs.shape))
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Data labels {}'.format(data_labels.shape))

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
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10:: Successfully load pre-train model {} to device'.format(dir, device))

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
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Successfully load pre-train model {} to device'.format(dirB, device))
    
    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
    print('Default_Methods::LoadShuffleDefenseAndCIFAR10 :: Successfully create a ShuffleDefense object')

    return valLoader, defense

#Method to do the RayS attack - query based blackbox attack on a single Vision Transformers
def RaySAttackVisionTransformer(xValData, yValData, xCleanData, yCleanData, qLimit):
    #Load the model and dataset
    saveTag = 'RaySAttack'
    start_time = datetime.now() 
    fileName = '{}_{}.txt'.format(saveTag, start_time)
    with open(fileName, 'a+') as wf:
        
        wf.write("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        print('Default_Methods::RaySAttackVisionTransformer :: Load the model (model architecture and model weights), dataset, and model evaluation')
        valLoader, defense = LoadViTLAndCIFAR10(xValData, yValData)

        cleanAcc, tp_5 = defense.validateD(valLoader)
        wf.write("RaySAttackVisionTransformer :: Before attack :: Clean acc:{}".format(cleanAcc))
        wf.write("RaySAttackVisionTransformer :: Before attack :: Top-5 error:{}".format(tp_5))

        #Get the clean samples
        numClasses = 10
        attackSampleNum = 100 #1000
        
        #Only attack correctly classified examples
        if xCleanData and yCleanData:
            print('Default_Methods::RaySAttackVisionTransformer :: Load correctly classified examples from file at {}'.format(xCleanData))
            xData = numpy.load(xCleanData)
            yData = numpy.load(yCleanData)
            cleanLoader = DMP.TensorToDataLoader(torch.from_numpy(xCleanData), 
                                            torch.from_numpy(yCleanData), 
                                            transforms = None, 
                                            batchSize = valLoader.batch_size, 
                                            randomizer = None)
        else:
            cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, 
                                                                attackSampleNum, 
                                                                valLoader, 
                                                                numClasses)

        #Set the attack parameters 
        epsMax = 0.031
        queryLimit = qLimit #10000

        #The next line does the actual attack on the defense 
        begin = datetime.now()
        wf.write("Begin attack at {}".format(begin.strftime('%Y-%m-%d %H:%M:%S')))
        advLoader = AttackWrappersRayS.RaySAttack(defense, 
                                            epsMax, 
                                            queryLimit, 
                                            cleanLoader)
        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - begin))

        #Check the results 
        robustAcc, tp_5_adv = defense.validateD(advLoader)
        cleanAcc, tp_5 = defense.validateD(valLoader)
        #print the results 
        # print("RaySAttackVisionTransformer :: Queries used: {}".format(queryLimit))
        # print("RaySAttackVisionTransformer :: Robust acc: {}".format(robustAcc))
        # print("RaySAttackVisionTransformer :: Clean acc: {}".format(cleanAcc))

        #Print the results 
        wf.write("RaySAttackVisionTransformer :: Queries used:{}".format(queryLimit))
        wf.write("RaySAttackVisionTransformer :: Robust acc:{}".format(robustAcc))
        wf.write("RaySAttackVisionTransformer :: Top-5 acc on adversarial images:{}".format(tp_5_adv))
        wf.write("RaySAttackVisionTransformer :: Clean acc:{}".format(cleanAcc))
        wf.write("RaySAttackVisionTransformer :: Top-5 acc on clean images:{}".format(tp_5))

        #Visualization
        with torch.no_grad():
            adv_preds = defense.predictD(advLoader, numClasses)

        #Extract first batch of orig data
        exmp_batch, label_batch = next(iter(cleanLoader))

        #Extract first batch of adv data
        adv_exmp_batch, adv_label_batch = next(iter(advLoader))
        for i in range(valLoader.batch_size):
            show_prediction(exmp_batch[i], 
                          label_batch[i].int(), 
                          adv_preds[i], 
                          5, 
                          adv_exmp_batch[i], 
                          None)

        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - start_time))
        
        sys.stdout.flush()

#Here we do the RayS attack on a shuffle defense comprised of two models, ViT-L-16 and BiT-M-R101x3
def RaySAttackShuffleDefense():
    #Load the model and dataset
    saveTag = 'RaySAttackShuffleDefense'
    start_time = datetime.now()
    fileName = '{}_{}.txt'.format(saveTag, start_time)
    
    with open(fileName, 'a+') as wf:
        
        wf.write("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        #Load the model and dataset
        valLoader, defense = LoadShuffleDefenseAndCIFAR10()
        #Get the clean samples
        numClasses = 10
        attackSampleNum = 1000 #1000
        cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)

        data_inputs, data_labels = next(iter(cleanLoader))
        print('Default_Methods::RaySAttackShuffleDefense :: Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
        print('Default_Methods::RaySAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::RaySAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

        #Set the attack parameters 
        epsMax = 0.031
        queryLimit = 10000 #10000
        #The next line does the actual attack on the defense 
        begin = datetime.now()
        wf.write("Begin attack at {}".format(begin.strftime('%Y-%m-%d %H:%M:%S')))
        advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader)
        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - begin))

        data_inputs, data_labels = next(iter(advLoader))
        print('Default_Methods::RaySAttackShuffleDefense :: Sucessfully generate adversarial examples that fool BOTH models. Print basic info of the first batch')
        print('Default_Methods::RaySAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::RaySAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

        #Check the results
        print('Default_Methods::RaySAttackShuffleDefense :: Robust accuracy of the ensemble model')
        robustAcc = defense.validateD(advLoader)
        print('Default_Methods::RaySAttackShuffleDefense :: Clean accuracy of the ensemble model')
        cleanAcc = defense.validateD(valLoader)

        #Print the results 
        wf.write("Default_Methods::RaySAttackShuffleDefense :: Queries used: {}".format(queryLimit))
        wf.write("Default_Methods::RaySAttackShuffleDefense :: Robust acc: {}".format(robustAcc))
        wf.write("Default_Methods::RaySAttackShuffleDefense :: Clean acc: {}".format(cleanAcc))

        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - start_time))

        sys.stdout.flush()

#Run the 100% strength adaptive attack on ViT-L-16, transfer based blackbox attack
def AdaptiveAttackVisionTransformer():

    #Corresponding tag for saving files
    #First part indicates the type of defense, second part indidcates the synthetic model and last part indicates the strenght of the attack (100%)
    saveTag = "AdaptiveAttackVisionTransformer_ViT-L-16, ViT-32(ImageNet21K), p100" 
    device = torch.device("cuda")
    print("Device {}".format(device))
    start_time = datetime.now()

    fileName = '{}_{}.txt'.format(saveTag, start_time)
    with open(fileName, 'a+') as wf:
        #Attack parameters       
        wf.write("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        numAttackSamples = 1000 # 1000
        epsForAttacks = 0.031
        clipMin = 0.0 
        clipMax = 1.0
        #Parameters of training the synthetic model 
        imgSize = 224
        batchSize = 32
        numClasses = 10
        numIterations = 4 # 4
        epochsPerIteration = 1 #1 
        epsForAug = 0.1 #when generating synthetic data, this value is eps for FGSM used to generate synthetic data
        learningRate = (3e-2) / 2 #Learning rate of the synthetic model 
        #Load the training dataset, validation dataset and the defense 
        valLoader, defense = LoadViTLAndCIFAR10()

        trainLoader = DMP.GetCIFAR10Training(imgSize, batchSize)
        data_inputs, data_labels = next(iter(trainLoader))
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Sucessfully load the train data. Print basic info of the first batch')
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Data labels {}'.format(data_labels.shape))

        #Get the clean data 
        xTest, yTest = DMP.DataLoaderToTensor(valLoader)

        #Clean validation loader
        cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)
        data_inputs, data_labels = next(iter(cleanLoader))
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Data labels {}'.format(data_labels.shape))


        #Create the synthetic model to generate adversarial examples
        syntheticDir = "Models//imagenet21k_ViT-B_32.npz" #imagenet21k_ViT-B_32.npz
        config = CONFIGS["ViT-B_32"]
        syntheticModel = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
        syntheticModel.load_from(numpy.load(syntheticDir))  
        syntheticModel.to(device)
        print('Default_Methods::AdaptiveAttackVisionTransformer :: Successfully load pre-train model {} to device'.format(syntheticDir, device))

        #Do the attack 
        oracle = defense
        dataLoaderForTraining = trainLoader
        optimizerName = "sgd"
        #Last line does the attack
        begin = datetime.now()
        wf.write("Begin attack at {}".format(begin.strftime('%Y-%m-%d %H:%M:%S')))
        AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, 
                                                numIterations, epochsPerIteration, epsForAug, 
                                                learningRate, optimizerName, dataLoaderForTraining, 
                                              cleanLoader, numClasses, epsForAttacks, clipMin, clipMax)
        now = datetime.now()
        wf.write("Complete the attack at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - begin))

        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - start_time))

        sys.stdout.flush()

#Run the 100% strength adaptive attack on shuffle defense
def AdaptiveAttackShuffleDefense():
    #Corresponding tag for saving files
    #First part indicates the type of defense, second part indidcates the synthetic model and last part indicates the strenght of the attack (100%)
    saveTag = "Adaptive_Attack_Defense_ViT-L-16, ViT-32(ImageNet21K), p100" 
    device = torch.device("cuda")
    print("Device {}".format(device))
    start_time = datetime.now()
    
    fileName = '{}_{}.txt'.format(saveTag, start_time)
    with open(fileName, 'a+') as wf:  
        wf.write("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))

        #Attack parameters 
        numAttackSamples = 1000 #1000
        epsForAttacks = 0.031
        clipMin = 0.0 
        clipMax = 1.0

        #Parameters of training the synthetic model 
        imgSize = 224
        batchSize = 32
        numClasses = 10
        numIterations = 4 #4
        epochsPerIteration = 10 #10
        epsForAug = 0.1 #when generating synthetic data, this value is eps for FGSM used to generate synthetic data
        learningRate = (3e-2) / 2 #Learning rate of the synthetic model

        #Load the training dataset, validation dataset and the defense 
        valLoader, defense = LoadShuffleDefenseAndCIFAR10()
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Sucessfully load the ensemble defense. Print basic info of the first batch')

        trainLoader = DMP.GetCIFAR10Training(imgSize, batchSize)
        data_inputs, data_labels = next(iter(trainLoader))
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Sucessfully load the train data. Print basic info of the first batch')
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

        #Get the clean data 
        xTest, yTest = DMP.DataLoaderToTensor(valLoader)
        cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, numAttackSamples, valLoader, numClasses)

        data_inputs, data_labels = next(iter(cleanLoader))
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Sucessfully generate clean examples that are correctly classified by BOTH models. Print basic info of the first batch')
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Data labels {}'.format(data_labels.shape))

        #Create the synthetic model 
        syntheticDir = "Models//imagenet21k_ViT-B_32.npz"
        config = CONFIGS["ViT-B_32"]
        syntheticModel = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
        syntheticModel.load_from(numpy.load(syntheticDir))  
        syntheticModel.to(device)
        print('Default_Methods::AdaptiveAttackShuffleDefense :: Successfully load pre-train model {} to device'.format(syntheticDir, device))

        #Do the attack 
        oracle = defense
        dataLoaderForTraining = trainLoader
        optimizerName = "sgd"
        #Last line does the attack
        begin = datetime.now()
        wf.write("Begin attack at {}".format(begin.strftime('%Y-%m-%d %H:%M:%S')))
        AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, 
                                                        numIterations, epochsPerIteration, epsForAug, 
                                                        learningRate, optimizerName, dataLoaderForTraining, 
                                                        cleanLoader, numClasses, epsForAttacks, clipMin, clipMax)
        now = datetime.now()
        wf.write("Complete the attack at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - begin))

        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - start_time))

        sys.stdout.flush()

#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def SelfAttentionGradientAttackCIFAR10():
    #Load the model and dataset
    saveTag = 'SelfAttentionGradientAttackCIFAR10'
    start_time = datetime.now()
    
    fileName = '{}_{}.txt'.format(saveTag, start_time)
    with open(fileName, 'a+') as wf:
        wf.write("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        print("Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Running Self-Attention Gradient Attack on ViT-L-16 and BiT-M-R101x3")
        #Set up the parameters for the attack 
        attackSampleNum = 10
        numClasses = 10
        coefficientArray = torch.zeros(2)
        secondcoeff = 2.0000e-04
        coefficientArray[0] = 1.0 - secondcoeff
        coefficientArray[1] = secondcoeff
        print("Coeff Array:")
        print(coefficientArray)
        device = torch.device("cuda")
        print("Device {}".format(device))

        epsMax = 0.031
        clipMin = 0.0
        clipMax = 1.0
        numSteps = 10
        #Load the models and the dataset
        #Note it is important to set vis to true so the transformer's model output returns the attention weights 
        valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)

        data_inputs, data_labels = next(iter(valLoader))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Sucessfully load the valid data. Print basic info of the first batch')
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data labels {}'.format(data_labels.shape))

        modelPlusList = defense.modelPlusList
        #Note that the batch size will effect how the gradient is computed in PyTorch
        #Here we use batch size 8 for ViT-L and batch size 2 for BiT-M. Other batch sizes are possible but they will not generate the same result
        modelPlusList[0].batchSize = 8
        modelPlusList[1].batchSize = 2

        #Get the clean examples 
        cleanLoader =AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, 
                                    attackSampleNum, numClasses, valLoader, modelPlusList)

        data_inputs, data_labels = next(iter(cleanLoader))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Sucessfully load the clean data that are classified corrected by the component models. Print basic info of the first batch')
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data labels {}'.format(data_labels.shape))

        #Do the attack
        begin = datetime.now()
        wf.write("Begin attack at {}".format(begin.strftime('%Y-%m-%d %H:%M:%S')))

        advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, 
            numSteps, modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax)

        now = datetime.now()
        wf.write("Complete the attack at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - begin))

        data_inputs, data_labels = next(iter(advLoader))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Sucessfully generate the adversarial examples using the proposed attack. Print basic info of the first batch')
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data inputs {}'.format(data_inputs.shape))
        print('Default_Methods::SelfAttentionGradientAttackCIFAR10 :: Data labels {}'.format(data_labels.shape))

        #Go through and check the robust accuray of each model on the adversarial examples 
        for i in range(0, len(modelPlusList)):
            acc = modelPlusList[i].validateD(advLoader)
            wf.write('model Name {} Robust acc: {}'.format(modelPlusList[i].modelName, acc))

        now = datetime.now()
        wf.write("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
        wf.write("--- {} seconds ---".format(now - start_time))

        sys.stdout.flush()