import os
import DefaultMethods
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

#Main method to do one of the CIFAR-10 experiments 
#Uncomment any one of the following lines to run an attack (RayS, Adaptive or SAGA) 
#The attack can be run on either ViT-L-16 or a defense made up of ViT-L-16 and BiT-M-R101x3 
def main(myInt):
    if myInt == 0:
        #Uncomment next line to do the RayS attack on the Vision Transformer, ViT-L-16
        DefaultMethods.RaySAttackVisionTransformer()

    elif myInt == 1:
        #Uncomment next line to do the RayS attack on the Shuffle Defense (ViT-L-16 and BiT-M-R101x3)
        #DefaultMethods.RaySAttackShuffleDefense()

    elif myInt == 2:
        #Uncomment next line to do the Adaptive attack on the Vision Transformer, ViT-L-16
        DefaultMethods.AdaptiveAttackVisionTransformer()

    elif myInt == 3:

        #Uncomment next line to do the Adaptive attack on the Shuffle Defense (ViT-L-16 and BiT-M-R101x3)
        DefaultMethods.AdaptiveAttackShuffleDefense()

    elif myInt == 4:
        #Uncomment next line to do the self-attention gradient on the Shuffle Defense
        DefaultMethods.SelfAttentionGradientAttackCIFAR10()
    else:
        print('Invalid choice.')

if __name__ == "__main__":

    main(int(sys.argv[1]))
    