# DNU

This repository provides the code for the papers 
*DNU: Deep Non-local Unrolling for Computational Spectral Imaging* (CVPR 2020)
*Deep Unrolling for Computational Spectral Imaging* (Submitted to TPAMI)

## Environment

Firstly, use Anaconda to create a virtual Python 3.9 environment with necessary dependencies from the **pytorch_environment.yaml** file in the code.

```
conda env create -f ./pytorch_environment.yaml
```

Then, activate the created environment and continue to train or test.

## Train

### Dataset Preparation

To train the DNU model for hyperspectral imaging, the datasets should be downloaded to your computer in advance.
(e.g., [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/), [KAIST](http://vclab.kaist.ac.kr/siggraphasia2017p1/), [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/), and [Harvard](http://vision.seas.harvard.edu/hyperspec/index.html).)

In synthetic simulations, for  ICVL dataset,  you can randomly select 100 spectral images for training and 50 spectral images for testing.  For  Harvard dataset, you should remove 6 deteriorated spectral images with large-area saturated pixels, and randomly select 35 spectral images for training and 9 spectral images for testing.
The training and test images in the ICVL dataset and  Harvard dataset are 48 * 48 inclined image blocks. 

In semi-physical simulations,  the CAVE dataset is used for training and the KAIST dataset is used for test. Then, you should modify the original CAVE and KAIST datasets by spectral interpolation, which have 28 spectral bands ranging from 450nm to 650nm. The patch size for training is 48*48, and the patch size for test is 256*256.

The **Cu_48.mat** and  "mask.mat" in **./mask/** are used for synthetic simulations and semi-physical simulations respectively.

Finally, edit the ```DATASET_PATH``` and ```MASK_PATH``` in **train.py** to indicate the name and path to your dataset and mask. Here is an example:
```
DATASET_PATH = "/PATH/DATSET", 
MASK_PATH = "/PATH/MASK", 
}
```
And there should be two directories in your dataset path: [train, test] to indicate which part should be used for training and testing.

### Argument Configuration

After the dataset is prepared, configure the ```dataset_name```, ```mask_name``` and ```dataset_loader_func_name```  in **train.py** and  **test.py** .

The ```dataset_loader_func_name``` can be any function provided in **datasets.py** or any function you implement using Pytorch Dataset. 
(You should also put your customized dataset loader function in *datasets.py* and set the ```dataset_loader_func_name``` to your customized function name. So that the model can automatically import and use it.)


Current **train.py** and **test_ICVL.py** have already provided an example configuration for training using the ICVL dataset.


### Start Training

After the configuration, the training can be started with the following commands:
```bash
python train.py
```

When the training starts, the trainer will save checkpoints into **./ckpt/** 

The checkpoint of a DNU model trained on ICVL is provided in **./ckpts_ICVL/**. You can directly apply **ep_160.pth** to subsequent test.

## Test

After training, reconstruction image can be generated using the following commands:
```bash
# For Harvard dataset, using the test set
python test_Harvard.py 

# For ICVL dataset, using the test set
python test_ICVL.py 

# For KAIST dataset, using the test set
python test_KAIST.py 
```

Then you can obtain  test metrics and visualized sRGB images by matlab codes  in **./visual**.

# Citation
If our code is useful in your reseach work, please consider citing our paper.
```
@inproceedings{wang2020dnu,
  title={DNU: deep non-local unrolling for computational spectral imaging},
  author={Wang, Lizhi and Sun, Chen and Zhang, Maoqing and Fu, Ying and Huang, Hua},
  booktitle={CVPR},
  pages={1661--1671},
  year={2020}
}
```
