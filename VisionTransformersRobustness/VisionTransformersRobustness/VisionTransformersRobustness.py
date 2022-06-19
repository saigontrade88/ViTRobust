import os
import sys
import DefaultMethods
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Main method to do one of the CIFAR-10 experiments 
#Uncomment any one of the following lines to run an attack (RayS, Adaptive or SAGA) 
#The attack can be run on either ViT-L-16 or a defense made up of ViT-L-16 and BiT-M-R101x3 
def main():
    parser = argparse.ArgumentParser(description = 'ViT and Robustness')
    parser.add_argument('-methodID', '--methodID', default=0, type=int, help='method ID')
    parser.add_argument('-xVal', '--xVal', type=str, help='xVal data')
    parser.add_argument('-yVal', '--yVal', type=str, help='yVal data')
    
    parser.add_argument('-xClean', '--xClean', type=str, help='xClean correctly classified data')
    parser.add_argument('-yClean', '--yClean', type=str, help='yClean correctly classified data')

    parser.add_argument('-qLimit', '--qLimit', default=1000, type=int, help='queryLimit')
    parser.add_argument('-xAdv', '--xAdv', type=str, help='xAdv data')
    parser.add_argument('-yAdv', '--yAdv', type=str, help='yAdv data')
    
    parser.add_argument('-attackSampleNum', '--attackSampleNum', default=1000, type=int)
    parser.add_argument('-validSampleNum', '--validSampleNum', default=1000, type=int)
    parser.add_argument('-trainSampleNum', '--trainSampleNum', default=1000, type=int)
    parser.add_argument('-savedFilePath', '--savedFilePath', type=str)

    args = parser.parse_args()


    # Collect arguments
    myInt = args.methodID
    xVal = args.xVal
    yVal = args.yVal

    xClean = args.xClean
    yClean = args.yClean

    qLimit = args.qLimit
    attackSampleNum=args.attackSampleNum

    xAdv = args.xAdv
    yAdv = args.yAdv

    #Debug
    trainSampleNum=args.trainSampleNum
    validSampleNum=args.validSampleNum

    savedFilePath = args.savedFilePath


    if myInt == 0:
        #Uncomment next line to do the RayS attack on the Vision Transformer, ViT-L-16
        # RaySAttackVisionTransformer(xValData, yValData, xCleanData, yCleanData)
        DefaultMethods.RaySAttackVisionTransformer(xVal, yVal, 
                                        qLimit, attackSampleNum, 
                                        xClean, yClean,
                                        trainSampleNum, validSampleNum, 
                                        savedFilePath)

    elif myInt == 1:
        #Uncomment next line to do the RayS attack on the Shuffle Defense (ViT-L-16 and BiT-M-R101x3)
        #Currently running and double the time limit. 
        # Find a way to save the ongoing result. Incase time limit issue. What else methods?
        DefaultMethods.RaySAttackShuffleDefense(xVal, yVal, 
                                    qLimit, attackSampleNum, 
                                            xClean, yClean,
                                            trainSampleNum, validSampleNum,
                                            savedFilePath)

    elif myInt == 2:
        #Uncomment next line to do the Adaptive attack on the Vision Transformer, ViT-L-16
        #Currently running and double the time limit. 
        DefaultMethods.AdaptiveAttackVisionTransformer(xVal, yVal, 
                                    qLimit, attackSampleNum, 
                                            xClean, yClean,
                                            trainSampleNum, validSampleNum,
                                            savedFilePath)

    elif myInt == 3:
        #Uncomment next line to do the Adaptive attack on the Shuffle Defense (ViT-L-16 and BiT-M-R101x3)
        #Check if you use CIFAR-10 dataset or not.
        #Read the error message
        DefaultMethods.AdaptiveAttackShuffleDefense(xVal, yVal, 
                                    qLimit, attackSampleNum, 
                                            xClean, yClean,
                                            trainSampleNum, validSampleNum,
                                            savedFilePath)

    elif myInt == 4:
        #Uncomment next line to do the self-attention gradient on the Shuffle Defense
        DefaultMethods.SelfAttentionGradientAttackCIFAR10(xVal, yVal, 
                                    qLimit, attackSampleNum, 
                                            xClean, yClean,
                                            trainSampleNum, validSampleNum,
                                            savedFilePath)
    else:
        print('Invalid choice.')

if __name__ == "__main__":

    main()
    