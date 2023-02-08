# PD-Quant
PyTorch implementation of *PD-Quant: Post-Training Quantization Based on Prediction Difference Metric*

## Usage
### 1. Download pre-trained FP model.
The pre-trained FP models in our experiment comes from [BRECQ](https://github.com/yhhhli/BRECQ), they can be downloaded in [link](https://github.com/yhhhli/BRECQ/releases/tag/v1.0).
And modify the path of the pre-trained model in ```hubconf.py```.

### 2. Installation.
```
python >= 3.7.13
numpy >= 1.21.6
torch >= 1.11.0
torchvision >= 0.12.0
```

### 3. Run experiments
You can run ```run_script.py``` for different models including ResNet18, ResNet50, RegNet600, RegNet3200, MobilenetV2, and MNasNet.

Take ResNet18 as an example:
```
python run_script.py resnet18
```

## Results

| Methods |  Bits (W/A) | Res18    |Res50 | MNV2 | Reg600M | Reg3.2G | MNasx2 |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   Full Prec. |   32/32 |  71.01 | 76.63 | 72.62 | 73.52 | 78.46 | 76.52 |
|PD-Quant| 4/4 | 69.30 | 75.09 | 68.33 | 71.04 | 76.57 | 73.30 |
|PD-Quant| 2/4 | 65.07 | 70.92 | 55.27 | 64.00 | 72.43 | 63.33| 
|PD-Quant| 4/2 | 58.65 | 64.18 | 20.40 | 51.29 | 62.76 | 38.89 |
|PD-Quant| 2/2 | 53.08 | 56.98 | 14.17 | 40.92 | 55.13 | 28.03| 

## Reference
```
@article{liu2022pd,
  title={PD-Quant: Post-Training Quantization based on Prediction Difference Metric},
  author={Liu, Jiawei and Niu, Lin and Yuan, Zhihang and Yang, Dawei and Wang, Xinggang and Liu, Wenyu},
  journal={arXiv preprint arXiv:2212.07048},
  year={2022}
}
```

## Thanks
Our code is based on [QDROP](https://github.com/wimh966/QDrop) by @wimh966.