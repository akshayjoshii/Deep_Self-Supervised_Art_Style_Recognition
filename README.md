# Saarland HLCV Project
Implementation of Auto Encoding Transformers for Art Style Recognition

# CIFAR10 Usage:
python3 main.py --mode=0 -F=tmp_data/cifar --choose=0 --lr=0.002 --lr1=0.1 --batch_size=128 --num_workers=4 --type=0 --KL_Lambda=1.0 --lambda=10.0 --lambda1=7.5 --lambda2=5.0 --lambda3=2.0 --lambda4=0.5 --max_lambda=1 --max_lambda1=0.75 --max_lambda2=0.5 --max_lambda3=0.2 --max_lambda4=0.05 --portion=0.005 --beta=75 --mix_mode=1  --Mixmatch_warm=50 --dataset=cifar10

#Colab Commands:
!cp "/content/drive/My Drive/Dataset/train.pickle" "train.pickle"
!cp "/content/drive/My Drive/Dataset/test.pickle" "test.pickle"

# Param usage definitions (For CIFAR10)
python3 main.py -h
--mode default:0, default mode to run
-F training data path(Automatically download to this path)
--choose use gpu id 
--lr default:0.002 learning rate for Adam optimizer for main backbone network
--lr1 default:0.1 learning rate for SGD optimizer for AET regularization network
--batch_size default:128 (Actually 256 is better, but one gpu can't support)
--num_workers default:16 number of data loading workers for pytorch dataloader
--type default:0 0:Wide ResNet-28-2, 1:Wide ResNet-28-2-Large
--KL_Lambda default:1.0 hyper parameter for KL divergence to control consistency in the framework
--lambda: warm factor for projective transformation AET regularization
--lambda1: warm factor for affine transformation AET regularization
--lambda2: warm factor for similarity transformation AET regularization
--lambda3: warm factor for euclidean transformation AET regularization
--lambda4: warm factor for CCBS transformation AET regularization
--max_lambda: hyper-parameter for projective transformation in AET regularization.
--max_lambda1: hyper-parameter for affine transformation in AET regularization.
--max_lambda2: hyper-parameter for similarity transformation in AET regularization.
--max_lambda3: hyper-parameter for eculidean transformation in AET regularization.
--max_lambda4: hyper-parameter for CCBS transformation in AET regularization.
--portion: specify the portion of data used as labeled data
--beta: hyper parameter for the consistency loss in MixMatch part
--mix_mode: default:1 specify to use Mosaic augmentation in MixMatch or not
--Mixmatch_warm: warm factor for MixMatch beta hyper parameter
--dataset: specify the dataset you will use for training
