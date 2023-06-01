
# MMML : Multiple Module for Modular Learning

MMML is a versatile GitHub repository dedicated to the field of Deep learning. With a focus on flexibility and modularity, this project aims to demonstrate how one could separate the configuration of various experiments from the actual code definition, for complex neural architecture that feature sevral building blocks. Leveraging the power of PyTorch, MMML enables users to easily experiment with new building blocks without the need to modify the entire training procedure.

The unique aspect of FlexiLearnNet lies in its approach to decomposing the components of a neural network. By separating the backbone, head, metric, and loss functions and defining their configuration using a graph structure, users can tailor the architecture to suit their specific needs. This modular approach facilitates the creation and sharing of experiments, making it effortless to explore new techniques and incorporate different components into the learning pipeline.


# Installation step :


requirements :
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


# structure of the code :

## division in modules
Division of Backbone / head / metric / Loss. Metric and loss are implemented using torchmetrics (it automaticly aggregate the loss)

Modules are connected between each other using a graph format, that is interpreted by a ForwardModule. The SplitConfigurationBuilder is used to define such graph. Nodes (torch modules or other callable) inputs and outputs are defined. Special node includes metrics (Metrics objets that are saved using tensorboard) and loss (a callable that will produce a loss function). 

## loss logging :

By convinience, the torchmetric library is used. It allows to easily aggregate the results. In order to log a loss, provide a name when connecting it to the graph (only work with Metric)s:

```python 
split_config.connect_loss(
    classification_loss, ["feature"], metric_name = "classification_loss"
)
```
## few shot dataset
Learn2learn module include dataset for fewshot. If you are like me and incouter a warning when downloading a large file from google drive, checkout the utils/dataset_utils.py for an exemple of download with cifar-fs.


# Features supported :

## Rotation prediction:


For task prediction, a module is responsible for generating the task before feeding the resulting task to the backbone / head. 
checkout : [rotation](./train_exemple/rotation.py)


## Manifold mixup :

For manifold mixup, the backbone behave differently at training time than usuall training. It take the target as an additional input, and produce a transformed version of the target in addition to the feature. The loss need to support this additional information. Loss Mixup will compute $\lambda  L(y_hat, y_{1}) + (1- \lambda ) *  L(y_hat, y_{2}) $ (for cross_entropy, that actually the same). checkout [manifold-rotations](./train_exemple/manifold-rotations.py) 



# To come :
## Contrastiveapproach

In contrastive approach, sevral views of the data are generated, and the view are compared between each other using a special loss, as well as a custom loss (that use either the target information, either the information about the provenance of each view in the batch). Need to implement a multiview, a dispatcher for this view, and the constrastive loss.

## Few shot learning.

As it was demonstrated in [EASY](https://arxiv.org/abs/2201.09699), an effective training of the backbone can lead to high quality dataset. Using simple meta-learning algorithm on the dataset composed by the feature of the backbone is often enough for a robust classification.  

Implementing the steps in this repo is actually not straightforward since you need to change the training steps.

Training step : 
For training : two backward-forward step
For validation : 
1. creation of the feature dataset
2. sampling of few shot task from this feature dataset
3. apply a few shot module


## Maybe

- closure for optimizer
- teacher - student architecture / EMA (should be implemented using a callback)
- experiments on augmentation
