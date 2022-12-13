# PD-Quant
Official PyTorch implementation of *PD-Quant: Post-Training Quantization Based on Prediction Difference Metric*

---------------------------------------------------
We propose a post-training quantization method PD-Quant, which uses the information of differences between network prediction before and after quantization to determine the quantization parameters. 
PD-Quant also adjusts the distribution of activations to mitigate the overfitting problem.
Thanks to QDrop's SOTA performance in PTQ, our implementation of PD-Quant is based on [QDrop](https://github.com/wimh966/QDrop).