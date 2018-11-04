# Designing Deep Convolutional Neural Networks for Continuous Object Orientation Estimation
This is a tensorflow impelementation of this [paper](https://arxiv.org/pdf/1702.01499.pdf). The official code is available in caffe framework.
This is one of my summer project and is not an official release form the authors. Please cite the above paper if you wish to use their work.

*Implemented on python3.5 with tensorflow and trained on google cloud with one tesla k80 GPU*

Please refer to the train.py for information on changing the default variables and giving path locations. To run the code you can just use 
```
python3 ./train.py
```
The base network used in the paper is resnet but the code contained variables to switch between resnet and squeezenet as base network. Please refer the paper for network structure.
