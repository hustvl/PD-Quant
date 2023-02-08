from .quant_layer import QuantModule
from .data_utils import save_inp_oup_data, save_dc_fp_data


def get_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True):
    cached_inps = save_inp_oup_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    return cached_inps

def get_dc_fp_init(model, block, cali_data, batch_size, input_prob: bool = False, keep_gpu: bool=True, lamb=50, bn_lr=1e-3):
    cached_outs, cached_outputs, cached_sym = save_dc_fp_data(model, block, cali_data, batch_size, input_prob=input_prob, keep_gpu=keep_gpu, lamb=lamb, bn_lr=bn_lr)
    return cached_outs, cached_outputs, cached_sym

def set_weight_quantize_params(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)
            '''caculate the step size and zero point for weight quantizer'''
            module.weight_quantizer(module.weight)
            module.weight_quantizer.set_inited(True)

def save_quantized_weight(model):
    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight.data = module.weight_quantizer(module.weight)
