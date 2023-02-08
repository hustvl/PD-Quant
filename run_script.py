import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    args = parser.parse_args()
    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]
    
    if args.exp_name == "resnet18":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch resnet18 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02")
            time.sleep(0.5)

    if args.exp_name == "resnet50":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch resnet50 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02")
            time.sleep(0.5)

    if args.exp_name == "regnetx_600m":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_600m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01")
            time.sleep(0.5)
    
    if args.exp_name == "regnetx_3200m":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_3200m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01")
            time.sleep(0.5)
    
    if args.exp_name == "mobilenetv2":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mobilenetv2 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.1 --T 1.0 --lamb_c 0.005")
            time.sleep(0.5)
    
    if args.exp_name == "mnasnet":
        for i in range(4):
            os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mnasnet --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.2 --T 1.0 --lamb_c 0.001")
            time.sleep(0.5)

    