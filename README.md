
# MMML : Multiple Module for Modular Learning

MMML is a versatile GitHub repository dedicated to the field of Deep learning. With a focus on flexibility and modularity, this project aims to demonstrate how one could separate the configuration of various experiments from the actual code definition, for complex neural architecture that feature sevral building blocks. Leveraging the power of PyTorch, MMML enables users to easily experiment with new building blocks without the need to modify the entire training procedure.

The unique aspect of this repo lies in its approach to decomposing the components of a neural network. By separating the backbone, head, metric, and loss functions and defining their configuration using a graph structure, users can tailor the architecture to suit their specific needs. This modular approach facilitates the creation and sharing of experiments, making it effortless to explore new techniques and incorporate different components into the learning pipeline.


## Installation step :


requirements :
kornia
pillow
tensorboard
gdown : (installation of large file from github)
torchmetric
Installation of learn2learn :
1. try usuall installation (pip install learn2learn)
2. if not working, try the following workaround :
(broken qpth pypi version due to setup.py for version >0.2.0)
-> module is only used in metaoptnet (you will need additional installation of you want to use it)

```cmd
pip install qpth==0.0.2
pip install numpy>=1.15.4 gym>=0.14.0 torch>=1.1.0 torchvision>=0.3.0 scipy requests gsutil tqdm
pip install learn2learn --no-deps 
```
## Datasets :

Classical dataset : 
Torchvision provides plenty of dataset. You can use a symbolic link/ absolute link to put all the necessary dataset into a common folder.

Learn2learn module include dataset for fewshot. If you are like me and incouter a warning when downloading a large file from google drive, checkout the utils/dataset_utils.py for an exemple of download with cifar-fs.



## experiments :


- CifarFS : 

S2M2R : main.py --file_name manifold-rotations-fs --save_name S2M2R --save --command --backbone resnet12 --scheduler SGDR --mix_aug manifold-mixup --lr 0.05 --use_rotation_pretext

S2M2R : python main.py --file_name manifold-rotations-fs --save_name S2WA --save --command --backbone resnet12 --scheduler SGDR --mix_aug manifold-mixup-cutmix --lr 0.05 --use_rotation_pretext --use_swa 
