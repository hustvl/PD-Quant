# PD-Quant
Official PyTorch implementation of *PD-Quant: Post-Training Quantization Based on Prediction Difference Metric*

---------------------------------------------------
We propose a post-training quantization method PD-Quant, which uses the information of differences between network prediction before and after quantization to determine the quantization parameters. 
PD-Quant also adjusts the distribution of activations to mitigate the overfitting problem.
Thanks to QDrop's SOTA performance in PTQ, our implementation of PD-Quant is based on [QDrop](https://github.com/wimh966/QDrop).

## 1. Download pre-trained FP model.
The pre-trained FP models in our experiment comes from [BRECQ](https://github.com/yhhhli/BRECQ), they can be downloaded in [link](https://github.com/yhhhli/BRECQ/releases/tag/v1.0).
And modify the path of the pre-trained model in `hubconf.py`.

## 2. Installation.
```
python >= 3.7.0
numpy >= 1.21.0
torch >= 1.8.1
torchvision >= 0.9.1
```

## 3. Run experiments
For ResNet-18 and ResNet-50:
```
python main_imagenet.py --data_path $imagenet$ --arch $resnet18/resnet50$ --n_bits_w $Wb$ --n_bits_a $Ab$ --weight 0.01 --pd --lamb_r 0.1 --T 4.0 --dc --lamb_c 0.02 --bn_lr 1e-3
```
For RegNetX-600MF and RegNetX-3200MF:
```
python main_imagenet.py --data_path $imagenet$ --arch $regnetx_600m/regnetx_3200m$ --n_bits_w $Wb$ --n_bits_a $Ab$ --weight 0.01 --pd --lamb_r 0.1 --T 4.0 --dc --lamb_c 0.01 --bn_lr 1e-3
```
For MobileNetV2:
```
python main_imagenet.py --data_path $imagenet$ --arch mobilenetv2 --n_bits_w $Wb$ --n_bits_a $Ab$ --weight 0.1 --pd --lamb_r 0.1 --T 1.0 --dc --lamb_c 0.005 --bn_lr 1e-3
```
For MNasNet:
```
python main_imagenet.py --data_path $imagenet$ --arch mnasnet --n_bits_w $Wb$ --n_bits_a $Ab$ --weight 0.2 --pd --lamb_r 0.1 --T 1.0 --dc --lamb_c 0.001 --bn_lr 1e-3
```

## 4. Results
Experimental results of PD-Quant on ImageNet.
| Bits (W/A) | Res18 | Res50 |  MNV2 | Reg600M | Reg3.2G | MNasx |
| ---------- | ----- | ----- | ----- | ------- | ------- | ----- |
|   32/32    | 71.01 | 76.63 | 72.62 |  73.52  |  78.46  | 76.52 |
|    4/4     | 69.30 | 75.09 | 68.33 |  71.04  |  76.57  | 73.30 |
|    2/4     | 65.07 | 70.92 | 55.27 |  64.00  |  72.43  | 63.33 |
|    4/2     | 58.65 | 64.18 | 20.40 |  51.29  |  62.76  | 38.89 |
|    2/2     | 53.08 | 56.98 | 14.17 |  40.92  |  55.13  | 28.03 |
