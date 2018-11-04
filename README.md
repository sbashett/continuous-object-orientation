# Designing Deep Convolutional Neural Networks for Continuous Object Orientation Estimation
This is a tensorflow impelementation of this [paper](https://arxiv.org/pdf/1702.01499.pdf). The official code is available in caffe framework.
This is one of my summer project and is not an official release form the authors. Please cite the above paper if you wish to use their work.

*Implemented on python3.5 with tensorflow and trained on google cloud with one tesla k80 GPU*

Please refer to the train.py for information on changing the default variables and giving path locations. To run the code you can just use
```
python3 ./train.py
```
The base network used in the paper is resnet but the code contains variables to switch between resnet and squeezenet as base network. Please refer the paper for network structure.

**Preprocess data**: Multi-class labels have to be generated based on the method described in the [paper](https://arxiv.org/pdf/1702.01499.pdf). The scripts for epfl dataset are in ```epfl_prep_data folder```. The ```epfl_augment.py``` is used to introduce vertical flipping or other transformations to augment the existing data. The ```epfl_extractor.py``` is used to assign multi-class labels for orientations. Slight modification might be required when using for other datasets. Please refer to official [epfl dataset](https://cvlab.epfl.ch/data/data-pose-index-php/) page for instructions to download and extract dataset.

The data.csv file in epfl_prep_data folder stores the location of images along with ground truth orientations. This file is used as input in the epfl_augment. The epfl_extractor generates multi-class discrete labels as described in paper and stores the pre-processed image locations along with the labels as csv file. Refer to the sample csv file data_train_labelM09N08.csv. These csv files are used by data_input.py script for feeding data to the network.

* ***data_input.py***: script used to feed input to network.
* ***orient_layer_train***: contains the layer description using tf api for the orientation layer to be added at the end of base network.
* ***mean_shift_layer***: contains the mean shift code to combine the multiple classifications from the network to get a continuous orientation estimate.
* ***squeezenet.py***: defines network for the squeezenet architecture. This script is taken and slightly modified from the official squeezenet repo.
* ***train.py***: contains the script for training and testing network. Refer to code to change default variables and assign paths.

The ```summaries``` folder contains the tensorflow summary files while training. Use ```tensorboard``` to visualize the summaries.

**Please do not forget to download the resnet and squeezenet trained parameters and verify the filenames in train.py**

**This is an initial commit. Working on adding comments and cleanup of code. So, please excuse if some parts of code are sloppy.**
