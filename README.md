# Improved Adversarial Training through Adaptive Instance-wise Loss Smoothing	

This repository contains the code of algorithm, **Instance adaptive Smoothness Enhanced Adversarial Training (ISEAT)**, and pre-trained models from the paper "Improved Adversarial Training through Adaptive Instance-wise Loss Smoothing".

# Pre-trained Models

Please find the pre-trained models through this [link](https://emckclac-my.sharepoint.com/:f:/g/personal/k19010102_kcl_ac_uk/EpqHUi-_NyVFtebb7hSTnKsByiMQLft0RbZdAKbiYkVSNQ?e=AwReAc).

# Files

* `data`: data files
* `model`: model checkpoints
  * `trained`: saved model checkpoints
* `output`: experiment logs
* `src`: source code
  * `train.py`: training models
  * `adversary.py`: evaluating adversarial robustness
  * `utils`: shared utilities such training, evaluation, log, printing, adversary, multiprocessing distribution
    * `train.py`: **implementation code of AT and ISEAT**.
  * `model`: model architectures
  * `data`: dataset classes
  * `config`: configurations for training and adversarial evaluation



# Requirements

The development environment is:

1. Python 3.8.13
2. PyTorch 1.11.0 + torchvision 0.12.0

The remaining dependencies are specified in the file `requirements.txt` and can be easily installed via the command:

```p
pip install -r requirements.txt
```

To prepare the involved dataset, an optional parameter `--download` should be specified in the running command. The program will download the required files automatically. This functionality currently doesn't support the dataset TinyImages which is used in semi-supervised learning setting. One may need to manually download 500K TinyImages data from this [repository](https://github.com/yaircarmon/semisup-adv) and move it into `/data` directory.

# Dependencies

* The training script is based on [IDBH](https://github.com/TreeLLi/DA-Alone-Improves-AT)
* the code of Wide ResNet is a revised version of [wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)
* the code of PreAct ResNet is from [Alleviate-Robust_Overfitting](https://github.com/VITA-Group/Alleviate-Robust-Overfitting)
* the code of AWP is modified from [AWP](https://github.com/csdongxian/AWP)

# Training

To adversarially train a PreAct ResNet18 on CIFAR10 using PGD10, run:

```python
python src/train.py -a paresnet --depth 18 --max_iter 10
```

To adversarially train a WideResNet34-10 on CIFAR10 using PGD10, run:

```python
python src/train.py -a wresnet --depth 34 --width 10 --max_iter 10
```

To adversarially train a WideResNet34-10 on CIFAR10 using PGD10 using the proposed method ISEAT, run:

```python
python src/train.py -a wresnet --depth 34 --width 10 --max_iter 10 -ag 7e-3 --lam 0 0.1
```

There are also a lot of hyper-parameters allowed to be specified in the running command in order to control the training. The common hyper-parameters, shared between `src/train.py` and `src/adversary.py` are stored in the `src/config/config.py` and the task-specific hyper-parameters are defined in the corresponding configuration file in the `src/config` folder. Please refer to the specific configuration file for the details of the default and the available options.

# Evaluation

For each training, the checkpoints will be saved in `model/trained/{log}` where {log} is the name of the experiment logbook (by default, is `log`). Each instance of training is tagged with a unique identifier, found in the logbook `output/log/{log}.json`, and that id is later used to load the well-trained model for the evaluation.

To evaluate the robustness of the "best" checkpoint against PGD50, run:

```
python src/adversary.py 0000 -v pgd -a PGD --max_iter 50
```

Similarly against AutoAttack (AA), run:

```
python src/adversary.py 0000 -v pgd -a AA
```

where "0000" should be replaced the real identifier to be evaluated.