"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
from pathlib import Path
import torch


from thop import profile  # 导入 thop 库
#*********************************************************************#
def count_parameters(model):
    """计算模型的参数数量"""
    return sum(p.numel() for p in model.parameters())

def print_model_info(model, input_size=(1, 3, 224, 224)):
    """打印模型的参数数量和 GFLOPs"""
    input_tensor = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    gflops = flops / (10**9)
    print(f"模型参数数量: {params:.2f}")
    print(f"模型 GFLOPs: {gflops:.2f}")
#**********************************************************************#


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    
    # 创建输出目录
    cfg.output_dir = Path('/gz-data/REDETR/rtdetr_pytorch/tensorboard')
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)