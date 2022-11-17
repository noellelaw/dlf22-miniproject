# Deep Learning Mini-Project, Fall 22

Goal: Identify a modified residual network architecture to achieve highest test accuracy on CIFAR-10, with no more than 5 million trainable parameters.

Access training notebook here: https://colab.research.google.com/drive/140wivCC7Htwe_QRP1jMfluBxYrwR7SX7?usp=sharing.

The code base is adapted from: https://github.com/kuangliu/pytorch-cifar. 

The following models can be found for training:
1. **resnet.py** -- ResNet with 1 residual block per residual layer, i.e. ResNet10(). This model was used as a starting point for accuracy and for an abalation study in tuning learning rate and accuracy for all following models. It was found that a learning rate of 5e-3 and batch size of 64 yielded consistently the best results. As such, that was what was used to train all further models.
2. **resnet_F5x5.py** -- Augments ResNet10(), aforementioned, by increasing the convolutional kernel size of the ith layer to a 5x5 kernel. This aims to test if including additional context increases accuracy on the CIFAR-10 dataset.
3. **resnet_L3.py** -- Augments ResNet by using 3 residual layers of channel size [64, 128, 256], respectively. This aims to test the impact of having a deeper network with with lesser-channel blocks on accuracy. It was found to increase accuracy, and that is likely a result due to CIFAR-10 images being smaller, and thus allowing smaller features to be extracted, rendering the 512x512 layer to be of limited use. This is the current home of the highest accuracy model: ResNet38().
4. **resnet_L3K3.py** -- Augments resnet_L3.py by augmenting kernel size in the ith skip connection to 3x3. 
5. **resnet_L2.py** -- Augments ResNet by using 2 residual layers of channel size [64, 128], [64, 256], [128, 256]respectively. This aims to test the impact of having a deeper network with with lesser-channel blocks on accuracy. In contrast to resnet_L3.py, removing an additional layer, was found to overall decrease accuracy.
